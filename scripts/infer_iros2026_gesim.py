"""
GE-Sim inference script for IROS 2026 WorldModel track (val/test).

Autoregressive chunked generation:
- Val: 1 first frame + 130 to generate (h5 has 131 timesteps, GT has frame_00001..frame_00130)
- Test: 1 first frame + 54 to generate (h5 has 55 timesteps, no GT)

Uses the Cosmos GE-Sim pipeline with traj+ray conditioning.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import math
import numpy as np
import torch
import torch.distributed as dist
import cv2
import time
from tqdm import tqdm
from yaml import load, Loader
from einops import rearrange
from pathlib import Path

from utils import import_custom_class, save_video
from utils.data_utils import get_text_conditions
from utils.get_traj_maps import get_traj_maps, simple_radius_gen_func
from utils.get_ray_maps import get_ray_maps
from utils.jepa_utils import (
    load_jepa_backend,
)
from utils.vjepa2_official_utils import is_official_vjepa2_ac_ckpt
from models.pipeline.jepa_guided_pipeline import JEPAGuidedPipeline


def _read_first_frame(sample_dir):
    """
    Read initial RGB frame for autoregressive rollout.

    Priority:
    1) frame.png (challenge input)
    2) head_color.mp4 first frame (fallback for broken/corrupted png)
    """
    frame_path = os.path.join(sample_dir, "frame.png")
    if os.path.isfile(frame_path):
        # Retry a few times to mitigate transient network/fs decode failures.
        for _ in range(3):
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is not None:
                return frame_bgr[:, :, ::-1]  # BGR -> RGB
            time.sleep(0.03)

        # Fallback: decode from bytes buffer.
        try:
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
            frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                return frame_bgr[:, :, ::-1]  # BGR -> RGB
        except Exception:
            pass

        # Fallback: PIL decoder (often more tolerant than OpenCV builds).
        try:
            from PIL import Image
            with Image.open(frame_path) as im:
                return np.array(im.convert("RGB"))
        except Exception:
            pass

    video_path = os.path.join(sample_dir, "head_color.mp4")
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok and frame_bgr is not None:
            return frame_bgr[:, :, ::-1]  # BGR -> RGB

    raise RuntimeError(
        f"failed to read initial frame from both frame.png and head_color.mp4: {sample_dir}"
    )


def load_config(config_file):
    cd = load(open(config_file, "r"), Loader=Loader)
    args = argparse.Namespace(**cd)
    return args


def prepare_model(args, dtype=torch.bfloat16, device="cuda:0", force_jepa_pipeline=False):
    """Load all model components (same pattern as infer_gesim.py)."""

    tokenizer_class = import_custom_class(
        args.tokenizer_class, getattr(args, "tokenizer_class_path", "transformers")
    )
    textenc_class = import_custom_class(
        args.textenc_class, getattr(args, "textenc_class_path", "transformers")
    )
    from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters

    cond_models = load_condition_models(
        tokenizer_class, textenc_class,
        args.pretrained_model_name_or_path if not hasattr(args, "tokenizer_pretrained_model_name_or_path") else args.tokenizer_pretrained_model_name_or_path,
        load_weights=args.load_weights
    )
    tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
    text_encoder = text_encoder.to(device, dtype=dtype).eval()

    vae_class = import_custom_class(
        args.vae_class, getattr(args, "vae_class_path", "transformers")
    )
    if getattr(args, 'vae_path', False):
        vae = load_vae_models(vae_class, args.vae_path).to(device, dtype=dtype).eval()
    else:
        vae = load_latent_models(vae_class, args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
    from typing import List
    if isinstance(vae.latents_mean, List):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if isinstance(vae.latents_std, List):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if vae is not None:
        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

    diffusion_model_class = import_custom_class(
        args.diffusion_model_class, getattr(args, "diffusion_model_class_path", "transformers")
    )
    diffusion_model = load_diffusion_model(
        model_cls=diffusion_model_class,
        model_dir=args.diffusion_model['model_path'],
        load_weights=args.load_weights and getattr(args, "load_diffusion_model_weights", True),
        **args.diffusion_model['config']
    ).to(device, dtype=dtype)
    total_params = count_model_parameters(diffusion_model)
    print(f'Total parameters for transformer model: {total_params}')

    diffusion_scheduler_class = import_custom_class(
        args.diffusion_scheduler_class, getattr(args, "diffusion_scheduler_class_path", "diffusers")
    )
    if hasattr(diffusion_scheduler_class, "from_pretrained") and os.path.exists(os.path.join(args.pretrained_model_name_or_path, "scheduler")):
        scheduler = diffusion_scheduler_class.from_pretrained(os.path.join(args.pretrained_model_name_or_path, 'scheduler'))
    elif hasattr(args, "diffusion_scheduler_args"):
        scheduler = diffusion_scheduler_class(**args.diffusion_scheduler_args)
    else:
        scheduler = diffusion_scheduler_class()

    if force_jepa_pipeline:
        pipeline_class = JEPAGuidedPipeline
    else:
        pipeline_class = import_custom_class(
            args.pipeline_class, getattr(args, "pipeline_class_path", "diffusers")
        )
    pipe = pipeline_class(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=diffusion_model
    )

    return tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe


def compose_action_from_h5(h5_path):
    """
    Compose 16-dim EE pose vector from h5.

    Returns: (T, 16) numpy array
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        end_pos = np.array(f["state/end/position"], dtype=np.float32)       # (T, 2, 3)
        end_ori = np.array(f["state/end/orientation"], dtype=np.float32)     # (T, 2, 4)
        eff_pos = np.array(f["state/effector/position"], dtype=np.float32)   # (T, 2)

    T = end_pos.shape[0]
    action = np.zeros((T, 16), dtype=np.float32)
    action[:, 0:3] = end_pos[:, 0, :]
    action[:, 3:7] = end_ori[:, 0, :]
    action[:, 7] = eff_pos[:, 0]
    action[:, 8:11] = end_pos[:, 1, :]
    action[:, 11:15] = end_ori[:, 1, :]
    action[:, 15] = eff_pos[:, 1]
    return action


def load_extrinsics(ext_path, T_total):
    """
    Load extrinsics from JSON. If only 1 entry, replicate for all T.
    Returns: (1, T, 4, 4) tensor (v=1)
    """
    with open(ext_path, "r") as f:
        ext_data = json.load(f)

    c2bs = []
    for item in ext_data:
        c2b = np.eye(4, dtype=np.float32)
        c2b[:3, :3] = np.array(item["extrinsic"]["rotation_matrix"], dtype=np.float32)
        c2b[:3, 3] = np.array(item["extrinsic"]["translation_vector"], dtype=np.float32)
        c2bs.append(c2b)
    c2bs = np.stack(c2bs, axis=0)  # (N_entries, 4, 4)

    if c2bs.shape[0] == 1:
        # Replicate static camera for all timesteps
        c2bs = np.repeat(c2bs, T_total, axis=0)
    elif c2bs.shape[0] < T_total:
        # Pad with last entry
        pad = np.repeat(c2bs[-1:], T_total - c2bs.shape[0], axis=0)
        c2bs = np.concatenate([c2bs, pad], axis=0)

    return torch.from_numpy(c2bs[:T_total]).unsqueeze(0)  # (1, T, 4, 4)


def load_intrinsic(int_path):
    """Load intrinsic. Returns: (1, 3, 3) tensor."""
    with open(int_path, "r") as f:
        info = json.load(f)["intrinsic"]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = info["fx"]
    intrinsic[1, 1] = info["fy"]
    intrinsic[0, 2] = info["ppx"]
    intrinsic[1, 2] = info["ppy"]
    return torch.from_numpy(intrinsic).unsqueeze(0)  # (1, 3, 3)


def compute_cond_to_concat(actions, extrinsics, intrinsic, sample_size):
    """
    Compute traj maps (3ch) + ray maps (6ch) = 9ch conditioning.

    Args:
        actions: (T, 16) tensor
        extrinsics: (1, T, 4, 4) c2b tensor
        intrinsic: (1, 3, 3) tensor (already scaled to target res)
        sample_size: (H, W)

    Returns:
        cond_to_concat: (9, 1, T, H, W) tensor — (c, v, t, h, w)
    """
    h, w = sample_size
    w2c = torch.linalg.inv(extrinsics)

    trajs = get_traj_maps(
        actions, w2c, extrinsics, intrinsic, sample_size,
        radius_gen_func=simple_radius_gen_func
    )  # (3, v, T, H, W) = (3, 1, T, H, W)
    trajs = trajs * 2 - 1

    v, t = extrinsics.shape[0], extrinsics.shape[1]
    intrinsic_expanded = intrinsic.unsqueeze(1).repeat(1, t, 1, 1).reshape(-1, 3, 3)
    c2w_flat = extrinsics.reshape(-1, 4, 4)
    rays_o, rays_d = get_ray_maps(intrinsic_expanded, c2w_flat, h, w)
    rays = torch.cat((rays_o, rays_d), dim=-1).reshape(v, t, h, w, 6)
    rays = rays.permute(4, 0, 1, 2, 3)  # (6, v, t, h, w)

    cond_to_concat = torch.cat((trajs, rays), dim=0)  # (9, v, t, h, w)
    return cond_to_concat


def scan_samples(data_root, split):
    """
    Scan val/test samples (nested layout: info_dataset/{task_id}/{episode_id}/).
    Returns list of sample directories.
    """
    info_dir = os.path.join(data_root, split, "info_dataset")
    samples = []
    for task_id in sorted(os.listdir(info_dir)):
        task_dir = os.path.join(info_dir, task_id)
        if not os.path.isdir(task_dir):
            continue
        for ep_id in sorted(os.listdir(task_dir)):
            ep_dir = os.path.join(task_dir, ep_id)
            if not os.path.isdir(ep_dir):
                continue
            samples.append({
                "dir": ep_dir,
                "task_id": task_id,
                "episode_id": ep_id,
            })
    return samples


def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    return distributed, rank, world_size, local_rank


def load_gt_frames(data_root, task_id, episode_id):
    """Load GT frames for validation. Returns list of numpy arrays (H, W, 3) in [0, 255]."""
    gt_dir = os.path.join(data_root, "validation", "gt_dataset", task_id, episode_id, "video")
    if not os.path.isdir(gt_dir):
        return None
    files = sorted(os.listdir(gt_dir))
    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(gt_dir, fname))[:, :, ::-1]  # BGR -> RGB
        frames.append(img)
    return frames


def compute_psnr(pred, gt):
    """Compute PSNR between two uint8 images."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(pred, gt):
    """Compute SSIM between two images (simplified)."""
    from skimage.metrics import structural_similarity
    return structural_similarity(pred, gt, channel_axis=2, data_range=255)


def build_jepa_infer_cfg(args, cmd_args=None):
    cfg = dict(getattr(args, "jepa", {}) or {})
    infer_cfg = dict(cfg.get("infer", {}) or {})
    if cmd_args is not None:
        if getattr(cmd_args, "jepa_ckpt", None):
            cfg["ckpt_path"] = cmd_args.jepa_ckpt
        if getattr(cmd_args, "jepa_guidance_strength", None) is not None:
            infer_cfg["guidance_strength"] = cmd_args.jepa_guidance_strength
        if getattr(cmd_args, "jepa_guidance_every_n", None) is not None:
            infer_cfg["guidance_every_n"] = cmd_args.jepa_guidance_every_n
        if getattr(cmd_args, "jepa_guidance_start_pct", None) is not None:
            infer_cfg["guidance_start_pct"] = cmd_args.jepa_guidance_start_pct
        if getattr(cmd_args, "jepa_guidance_end_pct", None) is not None:
            infer_cfg["guidance_end_pct"] = cmd_args.jepa_guidance_end_pct
        if getattr(cmd_args, "jepa_rerank_candidates", None) is not None:
            infer_cfg["rerank_candidates"] = cmd_args.jepa_rerank_candidates
        if getattr(cmd_args, "jepa_rerank_seed", None) is not None:
            infer_cfg["rerank_seed"] = cmd_args.jepa_rerank_seed
    cfg["infer"] = infer_cfg
    return cfg


def infer_sample(
    pipe, args, sample_dir, sample_size, n_previous, chunk, device,
    num_frames_to_generate, prompt, negative_prompt, first_frame_only=True,
    jepa_cfg=None, jepa_helper=None, jepa_prompt_embeds=None, jepa_prompt_attention_mask=None,
):
    """
    Run autoregressive chunked inference for one sample.

    Returns: list of generated frames as numpy arrays (H, W, 3) in [0, 255]
    """
    jepa_backend_type = getattr(jepa_helper, "backend_type", None)
    h5_path = os.path.join(sample_dir, "proprio_stats.h5")
    ext_path = os.path.join(sample_dir, "head_extrinsic_params_aligned.json")
    int_path = os.path.join(sample_dir, "head_intrinsic_params.json")

    # Read first frame
    first_frame = _read_first_frame(sample_dir)
    ori_h, ori_w = first_frame.shape[:2]
    first_frame_resized = cv2.resize(first_frame, (sample_size[1], sample_size[0]))  # (H, W, 3)

    # Normalize to [-1, 1], shape (C, T, H, W)
    img = first_frame_resized.astype(np.float32) / 255.0 * 2.0 - 1.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))  # (3, H, W)

    # Memory frames: replicate for n_previous
    obs = img.unsqueeze(1).repeat(1, n_previous, 1, 1)  # (C, n_prev, H, W)
    obs = obs.unsqueeze(0)  # (1, C, n_prev, H, W) — (v, c, t, h, w)

    v, c, t_mem, h_target, w_target = obs.shape

    # Load actions (all timesteps)
    actions = compose_action_from_h5(h5_path)
    T_total = actions.shape[0]

    # Load camera parameters
    extrinsics = load_extrinsics(ext_path, T_total)  # (1, T_total, 4, 4)
    intrinsic = load_intrinsic(int_path)  # (1, 3, 3)

    # Scale intrinsic to target resolution
    scale_w = sample_size[1] / ori_w
    scale_h = sample_size[0] / ori_h
    intrinsic_scaled = intrinsic.clone()
    intrinsic_scaled[:, 0, 0] *= scale_w
    intrinsic_scaled[:, 0, 2] *= scale_w
    intrinsic_scaled[:, 1, 1] *= scale_h
    intrinsic_scaled[:, 1, 2] *= scale_h

    actions_t = torch.from_numpy(actions).float()
    extrinsics_t = extrinsics.float()

    # Compute full cond_to_concat for all timesteps
    cond_to_concat = compute_cond_to_concat(
        actions_t, extrinsics_t, intrinsic_scaled, (h_target, w_target)
    )  # (9, 1, T_total, H, W)

    generated_chunks = []
    generated_count = 0
    if first_frame_only:
        future_cursor = 1
        mem_indices = [0 for _ in range(n_previous)]
    else:
        future_cursor = n_previous
        mem_indices = list(range(n_previous))

    ichunk = 0
    while generated_count < num_frames_to_generate and future_cursor < T_total:
        future_end = min(future_cursor + chunk, T_total)
        future_indices = list(range(future_cursor, future_end))
        actual_chunk = min(len(future_indices), num_frames_to_generate - generated_count)
        if actual_chunk <= 0:
            break

        cond_indices = mem_indices + future_indices
        expected_t = n_previous + chunk
        if len(cond_indices) < expected_t:
            cond_indices = cond_indices + [cond_indices[-1]] * (expected_t - len(cond_indices))
        else:
            cond_indices = cond_indices[:expected_t]
        cond_indices_t = torch.tensor(cond_indices, dtype=torch.long)
        ichunk_cond_to_concat = cond_to_concat.index_select(2, cond_indices_t)

        infer_jepa_cfg = dict((jepa_cfg or {}).get("infer", {}) or {})
        rerank_candidates = max(1, int(infer_jepa_cfg.get("rerank_candidates", 1)))
        rerank_seed = int(infer_jepa_cfg.get("rerank_seed", 2026))
        guidance_strength = float(infer_jepa_cfg.get("guidance_strength", 0.0))
        guidance_every_n = max(1, int(infer_jepa_cfg.get("guidance_every_n", 3)))
        guidance_start_pct = float(infer_jepa_cfg.get("guidance_start_pct", 0.2))
        guidance_end_pct = float(infer_jepa_cfg.get("guidance_end_pct", 0.8))
        if jepa_backend_type == "official_vjepa2_ac" and guidance_strength > 0.0:
            raise RuntimeError(
                "Official V-JEPA2-AC backend does not support latent guidance in infer_sample(). "
                "Set jepa.infer.guidance_strength=0.0."
            )

        chunk_actions = torch.from_numpy(actions[cond_indices]).float().unsqueeze(0).to(device)
        if chunk_actions.shape[1] < len(cond_indices):
            pad_len = len(cond_indices) - chunk_actions.shape[1]
            chunk_actions = torch.cat([chunk_actions, chunk_actions[:, -1:].repeat(1, pad_len, 1)], dim=1)

        candidate_results = []
        cond_for_pipe = rearrange(ichunk_cond_to_concat, "c v t h w -> v c t h w").to(device)
        obs_device = obs.to(device)
        for candidate_idx in range(rerank_candidates):
            pipe_kwargs = dict(
                video=obs_device.permute(0, 2, 1, 3, 4),
                cond_to_concat=cond_for_pipe,
                prompt=[prompt],
                negative_prompt=negative_prompt,
                height=h_target,
                width=w_target,
                n_view=v,
                num_frames=chunk,
                num_inference_steps=args.num_inference_step,
                n_prev=n_previous,
                guidance_scale=1.0,
                merge_view_into_width=False,
                output_type="pt",
                postprocess_video=False,
            )
            if rerank_candidates > 1:
                generator = torch.Generator(device=device)
                generator.manual_seed(rerank_seed + 1009 * ichunk + candidate_idx)
                pipe_kwargs["generator"] = generator
            if jepa_backend_type == "legacy" and hasattr(pipe, "set_jepa_modules") and guidance_strength > 0.0:
                pipe_kwargs.update(
                    actions=chunk_actions,
                    jepa_guidance_strength=guidance_strength,
                    jepa_guidance_every_n=guidance_every_n,
                    jepa_guidance_start_pct=guidance_start_pct,
                    jepa_guidance_end_pct=guidance_end_pct,
                )

            preds = pipe.infer(**pipe_kwargs)["frames"]
            preds = torch.clamp(preds, min=-1, max=1)
            pred_keep = preds[:, :, :actual_chunk].contiguous()

            score = 0.0
            if rerank_candidates > 1 and jepa_helper is not None and jepa_prompt_embeds is not None:
                try:
                    score_video = torch.cat((obs_device, pred_keep), dim=2)
                    score = float(
                        jepa_helper.score_consistency(
                            video=score_video,
                            actions=chunk_actions,
                            mem_size=n_previous,
                            frame_stride=int(infer_jepa_cfg.get("frame_stride", (jepa_cfg or {}).get("frame_stride", 1))),
                            cond_to_concat=cond_for_pipe,
                            prompt_embeds=jepa_prompt_embeds,
                            prompt_attention_mask=jepa_prompt_attention_mask,
                            n_view=v,
                            require_grad=False,
                        ).detach().cpu().item()
                    )
                except Exception:
                    score = float("inf")
            candidate_results.append((score, pred_keep.detach().cpu()))

        candidate_results.sort(key=lambda x: x[0])
        pred_keep = candidate_results[0][1]
        generated_chunks.append(pred_keep)

        # Keep recent contiguous memory frames for the next rollout chunk.
        mem_source = torch.cat((obs, pred_keep), dim=2)
        if mem_source.shape[2] >= n_previous:
            obs = mem_source[:, :, -n_previous:].clone()
        else:
            pad = obs[:, :, :1].repeat(1, 1, n_previous - mem_source.shape[2], 1, 1)
            obs = torch.cat((pad, mem_source), dim=2).clone()

        merged_indices = mem_indices + future_indices[:actual_chunk]
        if len(merged_indices) >= n_previous:
            mem_indices = merged_indices[-n_previous:]
        else:
            mem_indices = [merged_indices[0]] * (n_previous - len(merged_indices)) + merged_indices

        generated_count += actual_chunk
        future_cursor += actual_chunk
        ichunk += 1

    if len(generated_chunks) == 0:
        return []

    # Convert generated frames to numpy
    # gen_video: (v, c, t, h, w) where v=1
    gen_video = torch.cat(generated_chunks, dim=2)[0]  # (c, t, h, w)
    gen_video = gen_video[:, :num_frames_to_generate]
    # Convert from [-1, 1] to [0, 255]
    gen_video = ((gen_video + 1) / 2 * 255).clamp(0, 255).byte()
    gen_video = gen_video.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)

    generated_frames = [gen_video[i] for i in range(gen_video.shape[0])]
    return generated_frames


def main():
    parser = argparse.ArgumentParser(description="GE-Sim inference for IROS 2026 val/test")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--split', type=str, default='validation', choices=['validation', 'test'])
    parser.add_argument('--data_root', type=str, default='/apdcephfs_nj10/share_301739632/yhr/AgiBotWorld/data/AgiBotWorldChallenge-2026/WorldModel/iros_challenge_2026_acwm')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--video_output_root', type=str, default='',
                        help='Root path to save challenge-format mp4 outputs: {root}/{split}/{task}/{episode}/head_color.mp4. If empty, uses {output_dir}/challenge_mp4')
    parser.add_argument('--checkpoint', type=str, default=None, help='Override diffusion model checkpoint')
    parser.add_argument('--max_samples', type=int, default=-1, help='Max samples to process (-1 = all)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt', type=str, default='A robotic arm performing a manipulation task, best quality, consistent and smooth motion.')
    parser.add_argument('--jepa_ckpt', type=str, default='', help='Optional JEPA checkpoint for guidance/rerank')
    parser.add_argument('--jepa_guidance_strength', type=float, default=None)
    parser.add_argument('--jepa_guidance_every_n', type=int, default=None)
    parser.add_argument('--jepa_guidance_start_pct', type=float, default=None)
    parser.add_argument('--jepa_guidance_end_pct', type=float, default=None)
    parser.add_argument('--jepa_rerank_candidates', type=int, default=None)
    parser.add_argument('--jepa_rerank_seed', type=int, default=None)
    cmd_args = parser.parse_args()
    distributed, rank, world_size, local_rank = get_dist_info()

    if distributed and cmd_args.device.startswith("cuda"):
        cmd_args.device = f"cuda:{local_rank}"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    if distributed and dist.is_available() and (not dist.is_initialized()):
        backend = "nccl" if cmd_args.device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    args = load_config(cmd_args.config_file)

    # Override checkpoint if specified
    if cmd_args.checkpoint is not None:
        args.diffusion_model['model_path'] = cmd_args.checkpoint
        args.load_weights = True

    sample_size = tuple(args.data['train']['sample_size'])  # (H, W)
    n_previous = args.data['train']['n_previous']
    chunk = args.data['train']['chunk']

    jepa_cfg = build_jepa_infer_cfg(args, cmd_args)
    jepa_ckpt = str(jepa_cfg.get("ckpt_path", "") or "")
    force_jepa_pipeline = bool(jepa_ckpt) and (not is_official_vjepa2_ac_ckpt(jepa_ckpt))

    print(f"[rank {rank}/{world_size}] Loading models on device={cmd_args.device}...")
    tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe = prepare_model(
        args, device=cmd_args.device, force_jepa_pipeline=force_jepa_pipeline
    )

    jepa_helper = None
    jepa_prompt_embeds = None
    jepa_prompt_attention_mask = None
    jepa_backend_type = None
    if jepa_ckpt:
        try:
            backend, loaded_jepa_cfg = load_jepa_backend(
                jepa_ckpt,
                device=torch.device(cmd_args.device),
                dtype=diffusion_model.dtype,
            )
            jepa_backend_type = str(backend.get("backend_type", "legacy"))
            if jepa_backend_type == "legacy" and hasattr(pipe, "set_jepa_modules"):
                pipe.set_jepa_modules(
                    backend["frame_pooler"],
                    backend["dynamics_predictor"],
                    extract_layer=int(loaded_jepa_cfg.get("extract_layer", jepa_cfg.get("extract_layer", 14))),
                    frame_stride=int(
                        (jepa_cfg.get("infer", {}) or {}).get(
                            "frame_stride",
                            loaded_jepa_cfg.get("frame_stride", jepa_cfg.get("frame_stride", 1)),
                        )
                    ),
                )
            if jepa_backend_type == "legacy":
                from utils.jepa_utils import JEPADynamicsHelper

                jepa_helper = JEPADynamicsHelper(
                    vae=vae,
                    transformer=diffusion_model,
                    scheduler=scheduler,
                    frame_pooler=backend["frame_pooler"],
                    dynamics_predictor=backend["dynamics_predictor"],
                    weight_dtype=diffusion_model.dtype,
                    sigma_conditioning=float(getattr(args, "sigma_conditioning", 0.0001)),
                    noise_to_first_frame=float(getattr(args, "noise_to_first_frame", 0.05)),
                    pixel_wise_timestep=bool(getattr(args, "pixel_wise_timestep", False)),
                    extract_layer=int(loaded_jepa_cfg.get("extract_layer", jepa_cfg.get("extract_layer", 14))),
                )
            else:
                jepa_helper = backend["helper"]
            text_cond = get_text_conditions(tokenizer, text_encoder, prompt=cmd_args.prompt)
            jepa_prompt_embeds = text_cond["prompt_embeds"].to(cmd_args.device, dtype=diffusion_model.dtype)
            jepa_prompt_attention_mask = text_cond["prompt_attention_mask"].to(cmd_args.device)
            print(f"JEPA enabled ({jepa_backend_type}): ckpt={jepa_ckpt}")
        except Exception as e:
            print(f"JEPA disabled due to load failure: {repr(e)}")
            jepa_helper = None
            jepa_cfg = {}

    if not cmd_args.video_output_root:
        cmd_args.video_output_root = os.path.join(cmd_args.output_dir, "challenge_mp4")

    print(f"[rank {rank}] Scanning {cmd_args.split} samples...")
    samples_all = scan_samples(cmd_args.data_root, cmd_args.split)
    print(f"[rank {rank}] Found total {len(samples_all)} samples")
    print(f"[rank {rank}] Saving challenge mp4 outputs to: {cmd_args.video_output_root}")

    if cmd_args.max_samples > 0:
        samples_all = samples_all[:cmd_args.max_samples]

    if distributed:
        samples = samples_all[rank::world_size]
    else:
        samples = samples_all
    print(f"[rank {rank}] Assigned {len(samples)} / {len(samples_all)} samples")

    os.makedirs(cmd_args.output_dir, exist_ok=True)

    negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

    all_metrics = []

    for si, sample_info in enumerate(tqdm(samples, desc="Generating")):
        sample_dir = sample_info["dir"]
        task_id = sample_info["task_id"]
        ep_id = sample_info["episode_id"]

        # Determine number of frames to generate
        import h5py
        h5_path = os.path.join(sample_dir, "proprio_stats.h5")
        with h5py.File(h5_path, "r") as f:
            T_total = f["state/end/position"].shape[0]
        num_frames_to_generate = T_total - 1  # subtract the first frame

        print(f"[rank {rank}] [{si+1}/{len(samples)}] {task_id}/{ep_id}: generating {num_frames_to_generate} frames")

        generated_frames = infer_sample(
            pipe, args, sample_dir, sample_size, n_previous, chunk,
            cmd_args.device, num_frames_to_generate, cmd_args.prompt, negative_prompt,
            first_frame_only=True,
            jepa_cfg=jepa_cfg,
            jepa_helper=jepa_helper,
            jepa_prompt_embeds=jepa_prompt_embeds,
            jepa_prompt_attention_mask=jepa_prompt_attention_mask,
        )

        # Save generated frames
        sample_out_dir = os.path.join(cmd_args.output_dir, task_id, ep_id, "video")
        os.makedirs(sample_out_dir, exist_ok=True)
        for fi, frame in enumerate(generated_frames):
            fname = f"frame_{fi+1:05d}.png"
            cv2.imwrite(os.path.join(sample_out_dir, fname), frame[:, :, ::-1])  # RGB -> BGR

        # Save mp4 in challenge folder layout: {video_output_root}/{split}/{task_id}/{episode_id}/head_color.mp4
        video_base_dir = os.path.join(cmd_args.video_output_root, cmd_args.split, task_id, ep_id)
        os.makedirs(video_base_dir, exist_ok=True)
        mp4_path = os.path.join(video_base_dir, "head_color.mp4")
        h_frame, w_frame = generated_frames[0].shape[:2]
        import av
        container = av.open(mp4_path, mode="w")
        stream = container.add_stream("h264", rate=30)
        stream.width = w_frame
        stream.height = h_frame
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for frame in generated_frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Evaluate against GT (val only)
        if cmd_args.split == "validation":
            gt_frames = load_gt_frames(cmd_args.data_root, task_id, ep_id)
            if gt_frames is not None and len(gt_frames) > 0:
                n_eval = min(len(generated_frames), len(gt_frames))
                psnr_vals = []
                ssim_vals = []
                for fi in range(n_eval):
                    pred = cv2.resize(generated_frames[fi], (gt_frames[fi].shape[1], gt_frames[fi].shape[0]))
                    psnr_vals.append(compute_psnr(pred, gt_frames[fi]))
                    try:
                        ssim_vals.append(compute_ssim(pred, gt_frames[fi]))
                    except Exception:
                        pass

                sample_metrics = {
                    "task_id": task_id,
                    "episode_id": ep_id,
                    "n_generated": len(generated_frames),
                    "n_gt": len(gt_frames),
                    "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else 0,
                    "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else 0,
                }
                all_metrics.append(sample_metrics)
                print(f"[rank {rank}]   PSNR: {sample_metrics['psnr_mean']:.2f}, SSIM: {sample_metrics['ssim_mean']:.4f}")

    # Write metrics
    if all_metrics:
        rank_metrics_path = os.path.join(cmd_args.output_dir, f"metrics_rank{rank:03d}.json")
        summary = {
            "split": cmd_args.split,
            "n_samples": len(all_metrics),
            "avg_psnr": float(np.mean([m["psnr_mean"] for m in all_metrics])),
            "avg_ssim": float(np.mean([m["ssim_mean"] for m in all_metrics])),
            "jepa_ckpt": jepa_ckpt or None,
            "jepa_guidance_strength": float(jepa_cfg.get("infer", {}).get("guidance_strength", 0.0)) if jepa_cfg else 0.0,
            "jepa_rerank_candidates": int(jepa_cfg.get("infer", {}).get("rerank_candidates", 1)) if jepa_cfg else 1,
            "rank": rank,
            "world_size": world_size,
            "per_sample": all_metrics,
        }
        with open(rank_metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[rank {rank}] Overall PSNR: {summary['avg_psnr']:.2f}, SSIM: {summary['avg_ssim']:.4f}")
        print(f"[rank {rank}] Metrics saved to {rank_metrics_path}")

    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged_metrics = []
        if distributed:
            for i in range(world_size):
                rank_metrics_path = os.path.join(cmd_args.output_dir, f"metrics_rank{i:03d}.json")
                if not os.path.isfile(rank_metrics_path):
                    continue
                with open(rank_metrics_path, "r") as f:
                    rank_summary = json.load(f)
                merged_metrics.extend(rank_summary.get("per_sample", []))
        else:
            merged_metrics = all_metrics

        if merged_metrics:
            metrics_path = os.path.join(cmd_args.output_dir, "metrics.json")
            summary = {
                "split": cmd_args.split,
                "n_samples": len(merged_metrics),
                "avg_psnr": float(np.mean([m["psnr_mean"] for m in merged_metrics])),
                "avg_ssim": float(np.mean([m["ssim_mean"] for m in merged_metrics])),
                "jepa_ckpt": jepa_ckpt or None,
                "jepa_guidance_strength": float(jepa_cfg.get("infer", {}).get("guidance_strength", 0.0)) if jepa_cfg else 0.0,
                "jepa_rerank_candidates": int(jepa_cfg.get("infer", {}).get("rerank_candidates", 1)) if jepa_cfg else 1,
                "world_size": world_size,
                "per_sample": merged_metrics,
            }
            with open(metrics_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n[rank 0] Merged Overall PSNR: {summary['avg_psnr']:.2f}, SSIM: {summary['avg_ssim']:.4f}")
            print(f"[rank 0] Merged metrics saved to {metrics_path}")
    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
