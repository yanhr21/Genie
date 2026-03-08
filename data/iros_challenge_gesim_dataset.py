"""
Dataset for GE-Sim (Cosmos) training on IROS 2026 AgiBotWorldChallenge data.

Each sample is a flat directory:
  data_root/train/{sample_id}/head_color.mp4
  data_root/train/{sample_id}/proprio_stats.h5
  data_root/train/{sample_id}/head_extrinsic_params_aligned.json
  data_root/train/{sample_id}/head_intrinsic_params.json

Returns video tensor, cond_to_concat (traj + ray maps), and text caption.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import math
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import IterableDataset, get_worker_info
from einops import rearrange
import torchvision.transforms as transforms
try:
    import decord
    decord.bridge.set_bridge("numpy")
except Exception:
    decord = None
from moviepy.editor import VideoFileClip

from utils.get_traj_maps import get_traj_maps, simple_radius_gen_func
from utils.get_ray_maps import get_ray_maps
from utils import zero_rank_print


class IROSChallengeGESimDataset(Dataset):
    """
    GE-Sim training dataset for IROS 2026 WorldModel track.

    Returns:
        video: (C, 1, T, H, W)  — single view, T = n_previous + chunk
        cond_to_concat: (9, 1, T, H, W) — 3ch traj + 6ch ray maps
        caption: str
    """

    def __init__(
        self,
        data_roots,
        split="train",
        sample_size=(384, 512),
        sample_n_frames=900,
        preprocess="resize",
        valid_cam=None,
        chunk=25,
        action_chunk=None,
        n_previous=1,
        rollout_chunks=1,
        previous_pick_mode="random",
        random_crop=False,
        use_unified_prompt=True,
        unified_prompt="A robotic arm performing a manipulation task, best quality, consistent and smooth motion.",
        **kwargs,
    ):
        zero_rank_print(f"[IROSGESimDataset] loading from {data_roots}, split={split}")

        if isinstance(data_roots, str):
            data_roots = [data_roots]

        self.valid_cam = ["head"]
        self.use_unified_prompt = use_unified_prompt
        self.unified_prompt = unified_prompt
        self.random_crop = random_crop
        self._known_bad_indices = set()
        self._num_frames_cache = {}

        # Collect samples
        self.samples = []
        dropped_missing = 0
        required_files = (
            "head_color.mp4",
            "proprio_stats.h5",
            "head_extrinsic_params_aligned.json",
            "head_intrinsic_params.json",
        )
        for data_root in data_roots:
            split_dir = os.path.join(data_root, split)
            if not os.path.isdir(split_dir):
                zero_rank_print(f"[IROSGESimDataset] WARNING: {split_dir} not found")
                continue
            for sample_id in sorted(os.listdir(split_dir)):
                sample_dir = os.path.join(split_dir, sample_id)
                if not os.path.isdir(sample_dir):
                    continue
                if any(not os.path.isfile(os.path.join(sample_dir, fname)) for fname in required_files):
                    dropped_missing += 1
                    continue
                self.samples.append(sample_dir)

        self.length = len(self.samples)
        zero_rank_print(
            f"[IROSGESimDataset] found {self.length} valid samples (dropped_missing={dropped_missing})"
        )

        # Chunking — for GE-Sim, action_chunk == chunk (no temporal stride for actions)
        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert self.chunk * self.video_temporal_stride == self.action_chunk
        self.rollout_chunks = max(1, int(rollout_chunks))

        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size  # (H, W)
        self.n_previous = n_previous
        self.previous_pick_mode = previous_pick_mode

        # Transforms
        if preprocess == "resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        elif preprocess == "center_crop_resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),
                transforms.CenterCrop(sample_size),
            ])
        else:
            raise NotImplementedError(f"Unknown preprocess: {preprocess}")

        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def get_total_timesteps(self, sample_dir):
        cached = self._num_frames_cache.get(sample_dir, None)
        if cached is not None:
            return cached
        ext_path = os.path.join(sample_dir, "head_extrinsic_params_aligned.json")
        with open(ext_path, "r") as f:
            data = json.load(f)
        n_frames = len(data)
        self._num_frames_cache[sample_dir] = n_frames
        return n_frames

    def _compose_action(self, h5_path, slices):
        """
        Compose 16-dim EE pose vector from h5.

        Layout: [left_pos(3), left_quat(4), left_gripper(1),
                 right_pos(3), right_quat(4), right_gripper(1)]
        """
        import h5py
        with h5py.File(h5_path, "r") as f:
            end_pos = np.array(f["state/end/position"], dtype=np.float32)       # (T_total, 2, 3)
            end_ori = np.array(f["state/end/orientation"], dtype=np.float32)     # (T_total, 2, 4)
            eff_pos = np.array(f["state/effector/position"], dtype=np.float32)   # (T_total, 2)

        end_pos = end_pos[slices]
        end_ori = end_ori[slices]
        eff_pos = eff_pos[slices]

        T = len(slices)
        action = np.zeros((T, 16), dtype=np.float32)
        action[:, 0:3] = end_pos[:, 0, :]       # left arm position
        action[:, 3:7] = end_ori[:, 0, :]       # left arm quaternion
        action[:, 7] = eff_pos[:, 0]             # left gripper
        action[:, 8:11] = end_pos[:, 1, :]      # right arm position
        action[:, 11:15] = end_ori[:, 1, :]     # right arm quaternion
        action[:, 15] = eff_pos[:, 1]            # right gripper

        return torch.from_numpy(action)  # (T, 16)

    def _load_extrinsics(self, sample_dir, slices):
        """Load c2b extrinsics for head camera. Returns (1, T, 4, 4) tensor."""
        ext_path = os.path.join(sample_dir, "head_extrinsic_params_aligned.json")
        with open(ext_path, "r") as f:
            ext_data = json.load(f)

        c2bs = []
        for i in slices:
            # Clamp index to valid range
            idx = min(i, len(ext_data) - 1)
            item = ext_data[idx]
            c2b = np.eye(4, dtype=np.float32)
            c2b[:3, :3] = np.array(item["extrinsic"]["rotation_matrix"], dtype=np.float32)
            c2b[:3, 3] = np.array(item["extrinsic"]["translation_vector"], dtype=np.float32)
            c2bs.append(c2b)
        c2bs = np.stack(c2bs, axis=0)  # (T, 4, 4)
        return torch.from_numpy(c2bs).unsqueeze(0)  # (1, T, 4, 4)

    def _load_intrinsic(self, sample_dir):
        """Load intrinsic for head camera. Returns (1, 3, 3) tensor."""
        int_path = os.path.join(sample_dir, "head_intrinsic_params.json")
        with open(int_path, "r") as f:
            info = json.load(f)["intrinsic"]
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = info["fx"]
        intrinsic[1, 1] = info["fy"]
        intrinsic[0, 2] = info["ppx"]
        intrinsic[1, 2] = info["ppy"]
        return torch.from_numpy(intrinsic).unsqueeze(0)  # (1, 3, 3)

    def _read_video_frames(self, sample_dir, frame_indices):
        """Read specific frames from head_color.mp4. Returns (C, T, H, W) tensor in [0,1]."""
        video_path = os.path.join(sample_dir, "head_color.mp4")
        if decord is not None:
            video_reader = decord.VideoReader(video_path, num_threads=1)
            max_idx = max(0, len(video_reader) - 1)
            safe_indices = np.clip(np.asarray(frame_indices, dtype=np.int64), 0, max_idx)
            frames = video_reader.get_batch(safe_indices).asnumpy()
        else:
            video_reader = VideoFileClip(video_path)
            fps = float(video_reader.fps)
            max_idx = max(0, int(math.floor(video_reader.duration * fps)) - 1)
            frames = []
            for idx in frame_indices:
                safe_idx = min(max(int(idx), 0), max_idx)
                frames.append(video_reader.get_frame(float(safe_idx) / fps))
            video_reader.close()
        # (T, H, W, C) -> (C, T, H, W)
        video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).contiguous()
        video = video.float() / 255.0
        return video

    def _get_frame_indices(self, total_frames):
        """
        Sample frame indices: n_previous memory frames + chunk future frames.

        For training, we randomly sample a contiguous chunk from the video,
        with memory frames preceding the chunk (possibly subsampled).
        """
        # With temporal stride for video frames.
        # rollout_chunks > 1 means we sample one longer contiguous timeline:
        # n_previous + rollout_chunks * action_chunk.
        total_action_needed = self.n_previous + self.action_chunk * self.rollout_chunks

        # Random start for the action chunk
        max_start = max(0, total_frames - total_action_needed)
        start = random.randint(0, max_start)

        # Action indices (dense, every frame)
        action_indices = list(range(start, min(start + total_action_needed, total_frames)))
        # Pad if we don't have enough
        while len(action_indices) < total_action_needed:
            action_indices.append(action_indices[-1])

        # Video frame indices: n_previous memory + rollout_chunks*chunk future (with stride)
        mem_indices = action_indices[:self.n_previous]
        future_action_indices = action_indices[self.n_previous:]
        future_video_indices = future_action_indices[self.video_temporal_stride - 1::self.video_temporal_stride]
        # Pad if needed
        total_future_video = self.chunk * self.rollout_chunks
        while len(future_video_indices) < total_future_video:
            future_video_indices.append(future_video_indices[-1])
        future_video_indices = future_video_indices[:total_future_video]

        frame_indices = mem_indices + future_video_indices
        return frame_indices, action_indices

    def _compute_cond_to_concat(self, actions, extrinsics, intrinsic, sample_size):
        """
        Compute traj maps (3ch) + ray maps (6ch) = cond_to_concat (9ch).

        Args:
            actions: (T, 16) tensor
            extrinsics: (1, T, 4, 4) c2b tensor
            intrinsic: (1, 3, 3) tensor
            sample_size: (H, W)

        Returns:
            cond_to_concat: (9, 1, T, H, W) tensor
        """
        h, w = sample_size

        # Scale intrinsic to target resolution
        # We need the original image resolution to properly scale intrinsics.
        # Since we don't know the original resolution here, we use the intrinsic as-is
        # and assume it corresponds to the original video resolution.
        # We'll scale intrinsic after knowing the original size (done in get_batch).

        w2c = torch.linalg.inv(extrinsics)  # (1, T, 4, 4)

        # get_traj_maps expects: pose (T, 16), w2c (v, t, 4, 4), c2w (v, t, 4, 4), intrinsic (v, 3, 3), sample_size
        trajs = get_traj_maps(
            actions, w2c, extrinsics, intrinsic, sample_size,
            radius_gen_func=simple_radius_gen_func
        )  # (3, v, T, H, W) = (3, 1, T, H, W)
        trajs = trajs * 2 - 1  # normalize to [-1, 1]

        # get_ray_maps expects: intrinsic (vt, 3, 3), c2w (vt, 4, 4), H, W
        v, t = extrinsics.shape[0], extrinsics.shape[1]
        intrinsic_expanded = intrinsic.unsqueeze(1).repeat(1, t, 1, 1).reshape(-1, 3, 3)  # (vt, 3, 3)
        c2w_flat = extrinsics.reshape(-1, 4, 4)  # (vt, 4, 4)

        rays_o, rays_d = get_ray_maps(intrinsic_expanded, c2w_flat, h, w)
        # rays_o, rays_d: (vt, H, W, 3)
        rays = torch.cat((rays_o, rays_d), dim=-1).reshape(v, t, h, w, 6)
        rays = rays.permute(4, 0, 1, 2, 3)  # (6, v, t, h, w)

        cond_to_concat = torch.cat((trajs, rays), dim=0)  # (9, v, t, h, w)

        return cond_to_concat

    def get_batch(self, idx):
        sample_dir = self.samples[idx]
        h5_file = os.path.join(sample_dir, "proprio_stats.h5")

        total_ext = self.get_total_timesteps(sample_dir)
        total_frames = total_ext  # assume extrinsic count == frame count

        frame_indices, action_indices = self._get_frame_indices(total_frames)

        # Read video frames
        video = self._read_video_frames(sample_dir, frame_indices)  # (C, T, H, W) [0,1]
        ori_h, ori_w = video.shape[2], video.shape[3]

        # Resize video
        video = self.pixel_transforms_resize(video)  # (C, T, H_new, W_new)
        # Normalize to [-1, 1]
        _, _, new_h, new_w = video.shape
        video_flat = video.permute(1, 0, 2, 3)  # (T, C, H, W)
        video_flat = self.pixel_transforms_norm(video_flat)
        video = video_flat.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Add view dimension: (C, 1, T, H, W)
        video = video.unsqueeze(1)

        # Compose actions
        actions = self._compose_action(h5_file, action_indices)  # (T_action, 16)

        # Load camera parameters
        extrinsics = self._load_extrinsics(sample_dir, action_indices)  # (1, T_action, 4, 4)
        intrinsic = self._load_intrinsic(sample_dir)  # (1, 3, 3)

        # Scale intrinsic to target resolution
        scale_w = new_w / ori_w
        scale_h = new_h / ori_h
        intrinsic_scaled = intrinsic.clone()
        intrinsic_scaled[:, 0, 0] *= scale_w
        intrinsic_scaled[:, 0, 2] *= scale_w
        intrinsic_scaled[:, 1, 1] *= scale_h
        intrinsic_scaled[:, 1, 2] *= scale_h

        # Compute cond_to_concat using action_indices (dense timeline)
        cond_to_concat = self._compute_cond_to_concat(
            actions, extrinsics, intrinsic_scaled, (new_h, new_w)
        )  # (9, 1, T_action, H, W)

        # Now subsample cond_to_concat to match video frame_indices
        # frame_indices maps to: [mem(n_previous)] + [future(chunk with stride)]
        # action_indices maps to: [mem(n_previous)] + [future(action_chunk dense)]
        # We need cond_to_concat at frame_indices positions relative to action_indices
        relative_indices = []
        for fi in frame_indices:
            # Find position of fi in action_indices
            pos = action_indices.index(fi) if fi in action_indices else 0
            relative_indices.append(pos)
        cond_to_concat = cond_to_concat[:, :, relative_indices, :, :]  # (9, 1, T_video, H, W)

        frame_indices_t = torch.tensor(frame_indices, dtype=torch.int32)
        action_indices_t = torch.tensor(action_indices, dtype=torch.int32)
        return video, cond_to_concat, actions, sample_dir, frame_indices_t, action_indices_t

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        max_retries = 64
        last_err = None
        for _ in range(max_retries):
            if idx in self._known_bad_indices:
                idx = random.randint(0, self.length - 1)
                continue
            try:
                video, cond_to_concat, actions, sample_dir, frame_indices, action_indices = self.get_batch(idx)
                return dict(
                    video=video,               # (C, 1, T, H, W)
                    cond_to_concat=cond_to_concat,  # (9, 1, T, H, W)
                    actions=actions,           # (T_action, 16)
                    sample_dir=sample_dir,
                    frame_indices=frame_indices,
                    action_indices=action_indices,
                )
            except Exception as err:
                last_err = err
                self._known_bad_indices.add(idx)
                if len(self._known_bad_indices) % 50 == 1:
                    zero_rank_print(
                        f"[IROSGESimDataset] marked bad samples={len(self._known_bad_indices)} (latest_idx={idx})"
                    )
                idx = random.randint(0, self.length - 1)
        raise RuntimeError(
            f"[IROSGESimDataset] failed to fetch a valid sample after {max_retries} retries. "
            f"bad={len(self._known_bad_indices)}/{self.length}. last_err={repr(last_err)}"
        )


class IROSChallengeGESimIterableDataset(IterableDataset):
    """
    Stream-style iterable wrapper around IROSChallengeGESimDataset.

    - Shards samples across (world_size * num_workers) to avoid duplicated reads.
    - Supports deterministic per-epoch reshuffle via set_epoch().
    """

    def __init__(
        self,
        stream_shuffle=True,
        stream_seed=42,
        stream_infinite=False,
        stream_random_sample=True,
        **kwargs,
    ):
        super().__init__()
        self.inner = IROSChallengeGESimDataset(**kwargs)
        self.stream_shuffle = bool(stream_shuffle)
        self.stream_seed = int(stream_seed)
        self.stream_infinite = bool(stream_infinite)
        self.stream_random_sample = bool(stream_random_sample)
        self._epoch = 0

    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _get_shard_info(self):
        # By default we shard only by dataloader workers and let Accelerate shard by rank.
        # If needed for standalone non-Accelerate runs, enable rank sharding via env:
        #   GESIM_STREAM_SHARD_BY_RANK=1
        shard_by_rank = os.environ.get("GESIM_STREAM_SHARD_BY_RANK", "0") == "1"
        if shard_by_rank and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        return shard_id, num_shards

    def __iter__(self):
        total = len(self.inner)
        if total <= 0:
            return

        shard_id, num_shards = self._get_shard_info()
        samples_per_shard = max(1, total // max(num_shards, 1))
        epoch = self._epoch

        while True:
            rng = random.Random(self.stream_seed + epoch * 1000003 + shard_id)
            if self.stream_random_sample:
                for _ in range(samples_per_shard):
                    idx = rng.randrange(total)
                    yield self.inner[idx]
            else:
                indices = list(range(total))
                if self.stream_shuffle:
                    rng.shuffle(indices)
                for i in range(samples_per_shard):
                    idx = indices[(shard_id + i * num_shards) % total]
                    yield self.inner[idx]

            if not self.stream_infinite:
                break
            epoch += 1
