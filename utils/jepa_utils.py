import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from jepa.frame_pooler import FramePooler
from jepa.dynamics_predictor import LatentDynamicsPredictor
from utils.data_utils import get_latents, gen_noise_from_condition_frame_latent
from utils.geometry_utils import resize_traj_and_ray
from utils.model_utils import forward_pass
from utils.vjepa2_official_utils import (
    is_official_vjepa2_ac_ckpt,
    load_official_vjepa2_ac_backend,
)


def _resolve_pooling_mode(config: Dict, state_dict: Dict) -> str:
    if "pooling_mode" in config:
        return str(config["pooling_mode"]).lower()
    if any(k.startswith("query_tokens") or k.startswith("attn.") for k in state_dict.keys()):
        return "attn"
    return "mean"


def _select_video_state(video_states_buffer, extract_layer: int) -> torch.Tensor:
    if isinstance(video_states_buffer, dict):
        if int(extract_layer) not in video_states_buffer:
            raise KeyError(f"extract_layer={extract_layer} not found in video_states_buffer")
        return video_states_buffer[int(extract_layer)]
    return video_states_buffer[int(extract_layer)]


def load_jepa_checkpoint(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[FramePooler, LatentDynamicsPredictor, Dict]:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"JEPA checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    if "frame_pooler" not in ckpt or "dynamics_predictor" not in ckpt:
        raise ValueError(
            "Incompatible JEPA checkpoint format. Current loader expects keys "
            "'frame_pooler' and 'dynamics_predictor'. Official raw V-JEPA/V-JEPA2 "
            "checkpoints are not directly supported."
        )

    frame_pooler_state = ckpt["frame_pooler"]
    frame_pooler = FramePooler(
        dit_dim=int(config.get("dit_dim", 2048)),
        out_dim=int(config.get("out_dim", config.get("dim", 512))),
        pooling_mode=_resolve_pooling_mode(config, frame_pooler_state),
        num_queries=int(config.get("num_queries", 4)),
        num_heads=int(config.get("num_heads", 8)),
    )
    frame_pooler.load_state_dict(frame_pooler_state, strict=False)
    frame_pooler = frame_pooler.to(device=device, dtype=dtype).eval()
    frame_pooler.requires_grad_(False)

    dynamics_predictor = LatentDynamicsPredictor(
        dim=int(config.get("dim", config.get("out_dim", 512))),
        action_dim=int(config.get("action_dim", 16)),
        num_heads=int(config.get("predictor_num_heads", config.get("num_heads", 8))),
        num_layers=int(config.get("predictor_num_layers", 2)),
        mlp_ratio=float(config.get("predictor_mlp_ratio", 4.0)),
        causal_actions=bool(config.get("causal_actions", True)),
    )
    dynamics_predictor.load_state_dict(ckpt["dynamics_predictor"], strict=False)
    dynamics_predictor = dynamics_predictor.to(device=device, dtype=dtype).eval()
    dynamics_predictor.requires_grad_(False)

    return frame_pooler, dynamics_predictor, config


def load_jepa_backend(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[str, Any], Dict]:
    if is_official_vjepa2_ac_ckpt(ckpt_path):
        helper, config = load_official_vjepa2_ac_backend(ckpt_path, device=device, dtype=dtype)
        return {
            "backend_type": "official_vjepa2_ac",
            "helper": helper,
            "frame_pooler": None,
            "dynamics_predictor": None,
        }, config

    frame_pooler, dynamics_predictor, config = load_jepa_checkpoint(ckpt_path, device=device, dtype=dtype)
    return {
        "backend_type": "legacy",
        "helper": None,
        "frame_pooler": frame_pooler,
        "dynamics_predictor": dynamics_predictor,
    }, config


def align_actions_to_length(actions: torch.Tensor, target_len: int) -> torch.Tensor:
    target_len = max(1, int(target_len))
    if actions.shape[1] == target_len:
        return actions
    actions_t = actions.float().permute(0, 2, 1)
    actions_t = F.interpolate(actions_t, size=target_len, mode="linear", align_corners=True)
    return actions_t.permute(0, 2, 1).contiguous()


def compute_dynamics_consistency_loss(
    frame_repr: torch.Tensor,
    actions: torch.Tensor,
    dynamics_predictor: LatentDynamicsPredictor,
    mem_size: int,
    frame_stride: int = 1,
) -> torch.Tensor:
    if frame_stride > 1:
        frame_repr = frame_repr[:, ::frame_stride]
        actions = align_actions_to_length(actions, frame_repr.shape[1])
    if frame_repr.shape[1] <= max(1, int(mem_size) + 1):
        return frame_repr.new_zeros(())
    future_repr = frame_repr[:, int(mem_size):]
    future_actions = align_actions_to_length(actions, future_repr.shape[1])[:, : future_repr.shape[1]]
    if future_repr.shape[1] <= 1:
        return frame_repr.new_zeros(())
    predicted = dynamics_predictor(future_repr[:, :-1], future_actions[:, :-1])
    target = future_repr[:, 1:].detach()
    return F.mse_loss(predicted.float(), target.float())


class JEPADynamicsHelper:
    def __init__(
        self,
        vae,
        transformer,
        scheduler,
        frame_pooler: FramePooler,
        dynamics_predictor: LatentDynamicsPredictor,
        *,
        weight_dtype: torch.dtype,
        sigma_conditioning: float = 0.0001,
        noise_to_first_frame: float = 0.05,
        pixel_wise_timestep: bool = False,
        extract_layer: int = 14,
    ):
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.frame_pooler = frame_pooler
        self.dynamics_predictor = dynamics_predictor
        self.weight_dtype = weight_dtype
        self.sigma_conditioning = float(sigma_conditioning)
        self.noise_to_first_frame = float(noise_to_first_frame)
        self.pixel_wise_timestep = bool(pixel_wise_timestep)
        self.extract_layer = int(extract_layer)
        self.scheduler_sigmas = scheduler.sigmas

    def _get_base_transformer(self):
        return self.transformer.module if hasattr(self.transformer, "module") else self.transformer

    def get_store_buffer_indices(self) -> Tuple[int]:
        return (int(self.extract_layer),)

    def uses_rollout_buffer(self) -> bool:
        return True

    def extract_frame_repr_from_buffer(
        self,
        *,
        video_states_buffer,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        base_transformer = self._get_base_transformer()
        layer_repr = _select_video_state(video_states_buffer, self.extract_layer)
        patch_size = base_transformer.config.patch_size
        p_t, p_h, p_w = patch_size
        post_patch_frames = int(num_frames) // p_t
        post_patch_h = int(height) // p_h
        post_patch_w = int(width) // p_w
        return self.frame_pooler(
            layer_repr,
            num_frames=post_patch_frames,
            height=post_patch_h,
            width=post_patch_w,
        )

    def extract_frame_repr_from_latents(
        self,
        *,
        latents_video: torch.Tensor,
        cond_to_concat: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        mem_size: int,
        n_view: int = 1,
        frame_rate: int = 30,
        require_grad: bool = False,
    ) -> torch.Tensor:
        bv, _, latent_frames, latent_height, latent_width = latents_video.shape
        latents = rearrange(latents_video, "bv c f h w -> bv (f h w) c")

        cond_to_concat_flat = None
        if cond_to_concat is not None:
            cond_to_concat_resized = resize_traj_and_ray(
                cond_to_concat,
                mem_size=int(mem_size),
                future_size=int(latent_frames - mem_size),
                height=latent_height,
                width=latent_width,
            )
            cond_to_concat_flat = rearrange(cond_to_concat_resized, "bv c t h w -> bv (t h w) c")

        sigmas = self.scheduler_sigmas.to(device=latents_video.device, dtype=torch.float32)
        indices = torch.full(
            (bv,),
            fill_value=min(len(sigmas) - 1, max(0, int(self.sigma_conditioning * self.scheduler.config.num_train_timesteps))),
            device=latents_video.device,
            dtype=torch.long,
        )
        sigmas = sigmas.index_select(0, indices)
        timesteps = (sigmas * 1000.0).long()

        mem_latents = latents_video[:, :, : int(mem_size)]
        noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
            mem_latents,
            int(latent_frames),
            int(latent_height),
            int(latent_width),
            noise_to_condition_frames=self.noise_to_first_frame,
        )
        if self.pixel_wise_timestep:
            timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
        else:
            timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

        ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
        noisy_latents = (1.0 - ss) * latents + ss * noise
        if cond_to_concat_flat is not None:
            noisy_latents = torch.cat([noisy_latents, cond_to_concat_flat], dim=-1)

        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            pred_all = forward_pass(
                model=self.transformer,
                timesteps=timesteps,
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                num_frames=latent_frames,
                height=latent_height,
                width=latent_width,
                n_view=n_view,
                return_video=False,
                return_action=False,
                video_attention_mask=None,
                condition_mask=conditioning_mask,
                frame_rate=frame_rate,
                store_buffer=True,
                store_buffer_indices=self.get_store_buffer_indices(),
            )["latents"]

        video_states_buffer = pred_all.get("video_states_buffer", None)
        if video_states_buffer is None:
            raise RuntimeError("store_buffer=True did not return video_states_buffer.")
        return self.extract_frame_repr_from_buffer(
            video_states_buffer=video_states_buffer,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
        )

    def extract_frame_repr(
        self,
        *,
        video: torch.Tensor,
        cond_to_concat: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        mem_size: int,
        n_view: int = 1,
        frame_rate: int = 30,
        require_grad: bool = False,
    ) -> torch.Tensor:
        bv, _, _, raw_height, raw_width = video.shape
        mem = video[:, :, :mem_size]
        future = video[:, :, mem_size:]
        raw_future_frames = future.shape[2]
        if raw_future_frames <= 0:
            raise ValueError("JEPA extraction requires at least one future frame.")

        temporal_down = int(self.vae.temporal_compression_ratio)
        spatial_down = int(self.vae.spatial_compression_ratio)
        latent_frames = raw_future_frames // temporal_down + 1 + int(mem_size)
        latent_height = raw_height // spatial_down
        latent_width = raw_width // spatial_down

        mem_latents, future_latents = get_latents(self.vae, mem, future)
        mem_latents = rearrange(
            mem_latents,
            "(bv m) (h w) c -> bv c m h w",
            bv=bv,
            m=int(mem_size),
            h=latent_height,
        )
        future_latents = rearrange(
            future_latents,
            "bv (f h w) c -> bv c f h w",
            bv=bv,
            h=latent_height,
            w=latent_width,
        )
        latents_video = torch.cat((mem_latents, future_latents), dim=2)
        return self.extract_frame_repr_from_latents(
            latents_video=latents_video,
            cond_to_concat=cond_to_concat,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            mem_size=mem_size,
            n_view=n_view,
            frame_rate=frame_rate,
            require_grad=require_grad,
        )

    def score_consistency(
        self,
        *,
        video: Optional[torch.Tensor],
        actions: torch.Tensor,
        mem_size: int,
        frame_stride: int = 1,
        cond_to_concat: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        n_view: int = 1,
        frame_repr: Optional[torch.Tensor] = None,
        latent_video: Optional[torch.Tensor] = None,
        require_grad: bool = False,
    ) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        ref_batch = frame_repr.shape[0] if frame_repr is not None else (
            latent_video.shape[0] if latent_video is not None else video.shape[0]
        )
        if actions.shape[0] != ref_batch:
            if actions.shape[0] * max(1, n_view) == ref_batch:
                actions = actions.repeat_interleave(max(1, n_view), dim=0)
            elif actions.shape[0] == 1:
                actions = actions.repeat(ref_batch, 1, 1)
        if frame_repr is None:
            if latent_video is not None:
                frame_repr = self.extract_frame_repr_from_latents(
                    latents_video=latent_video,
                    cond_to_concat=cond_to_concat,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    mem_size=mem_size,
                    n_view=n_view,
                    require_grad=require_grad,
                )
            else:
                if video is None:
                    raise ValueError("video must be provided when frame_repr/latent_video are unavailable")
                frame_repr = self.extract_frame_repr(
                    video=video,
                    cond_to_concat=cond_to_concat,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    mem_size=mem_size,
                    n_view=n_view,
                    require_grad=require_grad,
                )
        return compute_dynamics_consistency_loss(
            frame_repr=frame_repr,
            actions=actions,
            dynamics_predictor=self.dynamics_predictor,
            mem_size=mem_size,
            frame_stride=frame_stride,
        )
