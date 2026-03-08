"""
Dataset adapter for IROS 2025/2026 AgiBotWorld Challenge (WorldModel track).

Each sample is a flat directory containing:
  - head_color.mp4
  - proprio_stats.h5
  - head_extrinsic_params_aligned.json
  - head_intrinsic_params.json

This adapter wraps GE's data utilities to match the expected interface.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import traceback
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import math
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip

from data.utils.get_actions import parse_h5
from data.utils.statistics import StatisticInfo
from utils import zero_rank_print


class IROSChallengeDataset(Dataset):
    """
    Dataset for IROS AgiBotWorld Challenge (WorldModel track).

    Flat directory layout per sample:
        data_root/{split}/{sample_id}/head_color.mp4
        data_root/{split}/{sample_id}/proprio_stats.h5
        data_root/{split}/{sample_id}/head_extrinsic_params_aligned.json
        data_root/{split}/{sample_id}/head_intrinsic_params.json
    """

    def __init__(
        self,
        data_roots,
        split="train",
        sample_size=(192, 256),
        sample_n_frames=900,
        preprocess="resize",
        valid_cam=None,
        chunk=9,
        action_chunk=None,
        n_previous=4,
        previous_pick_mode="random",
        random_crop=False,
        action_type="absolute",
        action_space="joint",
        stat_file=None,
        use_unified_prompt=True,
        unified_prompt="A robotic arm performing a manipulation task.",
        domains=None,
        **kwargs,
    ):
        zero_rank_print(f"[IROSChallengeDataset] loading data from {data_roots}, split={split}")

        if isinstance(data_roots, str):
            data_roots = [data_roots]

        self.action_type = action_type
        self.action_space = action_space
        self.random_crop = random_crop
        self.valid_cam = ["head"]  # IROS challenge only has head camera
        self.use_unified_prompt = use_unified_prompt
        self.unified_prompt = unified_prompt

        # Collect all samples (trust directory structure to avoid slow stat calls on CEPH)
        self.samples = []
        for data_root in data_roots:
            split_dir = os.path.join(data_root, split)
            if not os.path.isdir(split_dir):
                zero_rank_print(f"[IROSChallengeDataset] WARNING: {split_dir} not found, skipping")
                continue
            for sample_id in sorted(os.listdir(split_dir)):
                sample_dir = os.path.join(split_dir, sample_id)
                self.samples.append(sample_dir)

        self.length = len(self.samples)
        zero_rank_print(f"[IROSChallengeDataset] found {self.length} samples in split={split}")

        # Chunking
        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert self.chunk * self.video_temporal_stride == self.action_chunk

        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size

        if n_previous > 1:
            self.n_previous = n_previous
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = self.sample_n_frames - self.chunk
            self.previous_pick_mode = "uniform"

        # Transforms
        if preprocess == "center_crop_resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),
                transforms.CenterCrop(sample_size),
            ])
        elif preprocess == "resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        else:
            raise NotImplementedError(f"Unknown preprocess: {preprocess}")

        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        # Action statistics
        self.StatisticInfo = StatisticInfo
        if stat_file is not None and os.path.exists(stat_file):
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)

    def get_total_timesteps(self, sample_dir):
        ext_path = os.path.join(sample_dir, "head_extrinsic_params_aligned.json")
        with open(ext_path, "r") as f:
            data = json.load(f)
        return len(data)

    def get_frame_indexes(self, total_frames):
        chunk_end = random.randint(self.action_chunk, total_frames + self.action_chunk)
        indexes = np.array(list(range(chunk_end - self.sample_n_frames, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames - 1).tolist()
        video_end = indexes[-self.action_chunk:]
        mem_candidates = [
            indexes[int(i)] for i in range(0, self.sample_n_frames - self.action_chunk)
        ]

        if self.previous_pick_mode == "uniform":
            mem_indexes = [
                mem_candidates[int(i)]
                for i in np.linspace(0, len(mem_candidates) - 1, self.n_previous).tolist()
            ]
        elif self.previous_pick_mode == "random":
            mem_indexes = [
                mem_candidates[i]
                for i in sorted(
                    np.random.choice(
                        list(range(0, len(mem_candidates) - 1)),
                        size=self.n_previous - 1,
                        replace=False,
                    ).tolist()
                )
            ] + [mem_candidates[-1]]
        else:
            raise NotImplementedError(f"Unsupported previous_pick_mode: {self.previous_pick_mode}")

        frame_indexes = mem_indexes + video_end[self.video_temporal_stride - 1 :: self.video_temporal_stride]
        action_indexes = mem_indexes + video_end
        return frame_indexes, action_indexes

    def get_action_bias_std(self, domain_name):
        key = domain_name + "_" + self.action_space
        return (
            torch.tensor(self.StatisticInfo[key]["mean"]).unsqueeze(0),
            torch.tensor(self.StatisticInfo[key]["std"]).unsqueeze(0) + 1e-6,
        )

    def get_action(self, h5_file, slices, domain_name="agibotworld"):
        action, delta_action = parse_h5(
            h5_file, slices=slices, delta_act_sidx=0, action_space=self.action_space
        )
        act_meanv, act_stdv = self.get_action_bias_std(domain_name)

        state = torch.FloatTensor(action[self.n_previous - 1 : self.n_previous])
        state = (state - act_meanv) / act_stdv

        if self.action_type == "absolute":
            action = torch.FloatTensor(action)
            action = (action - act_meanv) / act_stdv
            return action, state
        elif self.action_type == "delta":
            delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name + "_delta")
            delta_action = torch.FloatTensor(delta_action)
            delta_action = (delta_action - delta_act_meanv) / delta_act_stdv
            return delta_action, state
        else:
            raise NotImplementedError(f"Unsupported action_type: {self.action_type}")

    def seek_mp4(self, sample_dir, slices):
        """Read frames from head_color.mp4"""
        video_path = os.path.join(sample_dir, "head_color.mp4")
        video_reader = VideoFileClip(video_path)
        fps = video_reader.fps
        video = []
        for idx in slices:
            video.append(video_reader.get_frame(float(idx) / fps))
        video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
        video = video.float() / 255.0
        video_reader.close()
        return [video]  # single view, wrapped in list

    def get_intrin_and_extrin(self, sample_dir, slices):
        """Get intrinsic (1x3x3) and extrinsic c2ws (1xTx4x4) for head camera."""
        # Intrinsic
        int_path = os.path.join(sample_dir, "head_intrinsic_params.json")
        with open(int_path, "r") as f:
            info = json.load(f)["intrinsic"]
        intrinsic = torch.eye(3, dtype=torch.float)
        intrinsic[0, 0] = info["fx"]
        intrinsic[1, 1] = info["fy"]
        intrinsic[0, 2] = info["ppx"]
        intrinsic[1, 2] = info["ppy"]

        # Extrinsic
        ext_path = os.path.join(sample_dir, "head_extrinsic_params_aligned.json")
        with open(ext_path, "r") as f:
            ext_data = json.load(f)
        c2ws = []
        for i in slices:
            item = ext_data[i]
            c2w = torch.eye(4, dtype=torch.float)
            c2w[:3, :3] = torch.FloatTensor(item["extrinsic"]["rotation_matrix"])
            c2w[:3, -1] = torch.FloatTensor(item["extrinsic"]["translation_vector"])
            c2ws.append(c2w)
        c2ws = torch.stack(c2ws, dim=0)

        # Stack for single view: (1, 3, 3) and (1, T, 4, 4)
        return intrinsic.unsqueeze(0), c2ws.unsqueeze(0)

    def transform_video(self, videos):
        """Resize video frames. Input: list of (C, T, H, W) tensors."""
        new_videos = []
        for video in videos:
            video = self.pixel_transforms_resize(video)
            new_videos.append(video)
        # Stack as (C, V, T, H, W) where V=1
        return torch.stack(new_videos, dim=1)

    def normalize_video(self, video):
        """Normalize video from [0,1] to [-1,1]. Input: (C, V, T, H, W)."""
        c, v, t, h, w = video.shape
        video = self.pixel_transforms_norm(
            video.permute(1, 2, 0, 3, 4).reshape(-1, c, h, w)
        ).reshape(v, t, c, h, w).permute(2, 0, 1, 3, 4)
        return video

    def get_batch(self, idx):
        sample_dir = self.samples[idx]
        h5_file = os.path.join(sample_dir, "proprio_stats.h5")
        total_frames = self.get_total_timesteps(sample_dir)

        caption = self.unified_prompt

        vid_indexes, action_indexes = self.get_frame_indexes(total_frames)
        action, state = self.get_action(h5_file, action_indexes)

        videos = self.seek_mp4(sample_dir, vid_indexes)
        videos = self.transform_video(videos)
        videos = self.normalize_video(videos)

        return videos, action, state, caption

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, actions, state, caption = self.get_batch(idx)
                break
            except Exception:
                traceback.print_exc()
                idx = random.randint(0, self.length - 1)

        sample = dict(
            video=video,
            actions=actions,
            state=state,
            caption=caption,
        )
        return sample
