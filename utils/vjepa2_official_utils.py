import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_official_vjepa2_ac_ckpt(ckpt_path: str) -> bool:
    name = os.path.basename(str(ckpt_path or "")).lower()
    return name in {"vjepa2-ac-vitg.pt", "vjepa2-ac-vitg.pth", "vjepa2-ac-vitg.bin"}


def _ensure_official_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1] / "third_party" / "vjepa2_official"
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Official V-JEPA2 repo not found: {repo_root}")
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def _clean_backbone_key(state_dict: Dict) -> Dict:
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = val
    return cleaned


def _normalize_video_to_01(video: torch.Tensor) -> torch.Tensor:
    video = video.float()
    if float(video.amin().item()) < -0.25:
        video = (video + 1.0) / 2.0
    return video.clamp(0.0, 1.0)


def _resize_short_side_center_crop(video: torch.Tensor, crop_size: int) -> torch.Tensor:
    bsz, channels, steps, height, width = video.shape
    short_side = min(height, width)
    resize_short = int(round(float(crop_size) * 256.0 / 224.0))
    scale = float(resize_short) / max(1.0, float(short_side))
    new_h = max(crop_size, int(round(height * scale)))
    new_w = max(crop_size, int(round(width * scale)))

    frames = video.permute(0, 2, 1, 3, 4).reshape(bsz * steps, channels, height, width)
    frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

    top = max(0, (new_h - crop_size) // 2)
    left = max(0, (new_w - crop_size) // 2)
    frames = frames[:, :, top : top + crop_size, left : left + crop_size]

    mean = frames.new_tensor(IMAGENET_MEAN).view(1, channels, 1, 1)
    std = frames.new_tensor(IMAGENET_STD).view(1, channels, 1, 1)
    frames = (frames - mean) / std
    return frames.view(bsz, steps, channels, crop_size, crop_size).permute(0, 2, 1, 3, 4).contiguous()


def _normalize_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _quat_inverse_xyzw(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(dim=-1)
    inv = torch.stack((-x, -y, -z, w), dim=-1)
    return _normalize_quat_xyzw(inv)


def _quat_multiply_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_to_euler_xyz_xyzw(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat_xyzw(quat)
    x, y, z, w = quat.unbind(dim=-1)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(sinp.clamp(-1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


def _select_primary_arm(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if actions.shape[-1] < 16:
        raise ValueError(f"Official V-JEPA2-AC backend expects 16-d AgiBot actions, got {actions.shape}")

    left_pos = actions[:, :, 0:3]
    left_quat = actions[:, :, 3:7]
    left_grip = actions[:, :, 7:8]
    right_pos = actions[:, :, 8:11]
    right_quat = actions[:, :, 11:15]
    right_grip = actions[:, :, 15:16]

    left_motion = (left_pos[:, 1:] - left_pos[:, :-1]).norm(dim=-1).mean(dim=1)
    right_motion = (right_pos[:, 1:] - right_pos[:, :-1]).norm(dim=-1).mean(dim=1)
    choose_left = (left_motion >= right_motion).view(-1, 1, 1)

    pos = torch.where(choose_left, left_pos, right_pos)
    quat = torch.where(choose_left, left_quat, right_quat)
    grip = torch.where(choose_left, left_grip, right_grip)
    return pos, quat, grip


def _build_official_states_actions(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pos, quat, grip = _select_primary_arm(actions.float())
    quat = _normalize_quat_xyzw(quat)
    euler = _quat_to_euler_xyz_xyzw(quat)
    states = torch.cat((pos, euler, grip), dim=-1)

    pos_diff = pos[:, 1:] - pos[:, :-1]
    rel_quat = _quat_multiply_xyzw(quat[:, 1:], _quat_inverse_xyzw(quat[:, :-1]))
    ang_diff = _quat_to_euler_xyz_xyzw(rel_quat)
    grip_diff = grip[:, 1:] - grip[:, :-1]
    actions_delta = torch.cat((pos_diff, ang_diff, grip_diff), dim=-1)
    return states, actions_delta


class OfficialVJEPA2ACHelper:
    def __init__(self, encoder, predictor, config: Dict, *, weight_dtype: torch.dtype):
        self.encoder = encoder.eval()
        self.predictor = predictor.eval()
        self.config = dict(config or {})
        self.weight_dtype = weight_dtype
        self.backend_type = "official_vjepa2_ac"
        self.supports_guidance = False
        self.normalize_reps = bool(self.config.get("normalize_reps", True))
        self.loss_exp = float(self.config.get("loss_exp", 1.0))
        self.auto_steps = int(self.config.get("auto_steps", 2))
        self.crop_size = int(self.config.get("crop_size", 256))
        self.max_num_frames = int(self.config.get("num_frames", 64))

    def uses_rollout_buffer(self) -> bool:
        return False

    def get_store_buffer_indices(self):
        return tuple()

    def _subsample_sequence(
        self,
        video: torch.Tensor,
        actions: torch.Tensor,
        *,
        mem_size: int,
        frame_stride: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != video.shape[0]:
            if actions.shape[0] == 1:
                actions = actions.repeat(video.shape[0], 1, 1)
            else:
                raise ValueError(f"actions batch {actions.shape[0]} does not match video batch {video.shape[0]}")

        start_idx = max(0, int(mem_size) - 1)
        video = video[:, :, start_idx:]
        actions = actions[:, start_idx:]

        step = max(1, int(frame_stride))
        if step > 1:
            idx = torch.arange(0, video.shape[2], step=step, device=video.device, dtype=torch.long)
            video = video.index_select(2, idx)
            actions = actions.index_select(1, idx)

        if video.shape[2] > self.max_num_frames:
            idx = torch.linspace(
                0,
                video.shape[2] - 1,
                steps=self.max_num_frames,
                device=video.device,
            ).round().long()
            video = video.index_select(2, idx)
            actions = actions.index_select(1, idx)

        return video.contiguous(), actions.contiguous()

    def _encode_frame_tokens(self, video: torch.Tensor, *, require_grad: bool) -> torch.Tensor:
        video = _resize_short_side_center_crop(_normalize_video_to_01(video), self.crop_size)
        bsz, channels, steps, height, width = video.shape
        encoder_in = video.permute(0, 2, 1, 3, 4).reshape(bsz * steps, channels, 1, height, width)
        encoder_in = encoder_in.repeat(1, 1, 2, 1, 1).to(dtype=self.weight_dtype)
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            tokens = self.encoder(encoder_in)
            tokens = tokens.view(bsz, steps, -1, tokens.size(-1))
            if self.normalize_reps:
                tokens = F.layer_norm(tokens, (tokens.size(-1),))
        return tokens

    def extract_frame_repr(
        self,
        *,
        video: torch.Tensor,
        cond_to_concat=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        mem_size: int = 0,
        n_view: int = 1,
        frame_rate: int = 30,
        require_grad: bool = False,
    ) -> torch.Tensor:
        del cond_to_concat, prompt_embeds, prompt_attention_mask, mem_size, n_view, frame_rate
        dummy_actions = video.new_zeros(video.shape[0], video.shape[2], 16)
        video, _ = self._subsample_sequence(video, dummy_actions, mem_size=0, frame_stride=1)
        frame_tokens = self._encode_frame_tokens(video, require_grad=require_grad)
        return frame_tokens.mean(dim=2)

    def score_consistency(
        self,
        *,
        video: torch.Tensor,
        actions: torch.Tensor,
        mem_size: int,
        frame_stride: int = 1,
        cond_to_concat=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        n_view: int = 1,
        frame_repr=None,
        latent_video=None,
        require_grad: bool = False,
    ) -> torch.Tensor:
        del cond_to_concat, prompt_embeds, prompt_attention_mask, n_view, frame_repr, latent_video
        video, actions = self._subsample_sequence(
            video=video,
            actions=actions,
            mem_size=mem_size,
            frame_stride=frame_stride,
        )
        if video.shape[2] <= 1 or actions.shape[1] <= 1:
            return video.new_zeros(())

        states, actions_delta = _build_official_states_actions(actions)
        frame_tokens = self._encode_frame_tokens(video, require_grad=require_grad)
        bsz, steps, tokens_per_frame, dim = frame_tokens.shape
        flat_tokens = frame_tokens.flatten(1, 2)

        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            z_tf = self.predictor(
                flat_tokens[:, :-tokens_per_frame].to(dtype=self.weight_dtype),
                actions_delta.to(dtype=self.weight_dtype),
                states[:, :-1].to(dtype=self.weight_dtype),
            )
            if self.normalize_reps:
                z_tf = F.layer_norm(z_tf, (z_tf.size(-1),))
            target_tf = flat_tokens[:, tokens_per_frame : tokens_per_frame + z_tf.size(1)].detach()
            teacher_loss = torch.mean(torch.abs(z_tf.float() - target_tf.float()) ** self.loss_exp) / self.loss_exp

            rollout_steps = min(max(1, self.auto_steps), steps - 1)
            if rollout_steps <= 1:
                return teacher_loss

            z_roll = torch.cat((flat_tokens[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]), dim=1)
            for step_idx in range(1, rollout_steps):
                z_next = self.predictor(
                    z_roll.to(dtype=self.weight_dtype),
                    actions_delta[:, : step_idx + 1].to(dtype=self.weight_dtype),
                    states[:, : step_idx + 1].to(dtype=self.weight_dtype),
                )[:, -tokens_per_frame:]
                if self.normalize_reps:
                    z_next = F.layer_norm(z_next, (z_next.size(-1),))
                z_roll = torch.cat((z_roll, z_next), dim=1)
            z_ar = z_roll[:, tokens_per_frame:]
            target_ar = flat_tokens[:, tokens_per_frame : tokens_per_frame + z_ar.size(1)].detach()
            rollout_loss = torch.mean(torch.abs(z_ar.float() - target_ar.float()) ** self.loss_exp) / self.loss_exp
        return teacher_loss + rollout_loss


def load_official_vjepa2_ac_backend(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[OfficialVJEPA2ACHelper, Dict]:
    if not is_official_vjepa2_ac_ckpt(ckpt_path):
        raise ValueError(f"Unsupported official V-JEPA2 checkpoint: {ckpt_path}")
    _ensure_official_repo_on_path()

    from src.models import ac_predictor as vit_ac_predictor
    from src.models import vision_transformer as vit_encoder

    encoder = vit_encoder.vit_giant_xformers(
        patch_size=16,
        img_size=(256, 256),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )
    predictor = vit_ac_predictor.vit_ac_predictor(
        img_size=(256, 256),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        action_embed_dim=7,
        is_frame_causal=True,
        use_rope=True,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "encoder" not in state_dict or "predictor" not in state_dict:
        raise ValueError(f"Official V-JEPA2-AC checkpoint missing encoder/predictor keys: {ckpt_path}")

    encoder.load_state_dict(_clean_backbone_key(state_dict["encoder"]), strict=False)
    predictor.load_state_dict(_clean_backbone_key(state_dict["predictor"]), strict=True)

    encoder = encoder.to(device=device, dtype=dtype).eval()
    predictor = predictor.to(device=device, dtype=dtype).eval()
    encoder.requires_grad_(False)
    predictor.requires_grad_(False)

    config = {
        "backend_type": "official_vjepa2_ac",
        "crop_size": 256,
        "num_frames": 64,
        "action_embed_dim": 7,
        "normalize_reps": True,
        "loss_exp": 1.0,
        "auto_steps": 2,
        "supports_guidance": False,
    }
    return OfficialVJEPA2ACHelper(encoder, predictor, config, weight_dtype=dtype), config
