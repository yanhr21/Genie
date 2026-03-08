"""
JEPA-guided GE-Sim inference pipeline.
"""

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import is_torch_xla_available, logging

from models.pipeline.gesim_pipeline import GeSimCosmos2Pipeline, retrieve_timesteps
from utils.geometry_utils import resize_traj_and_ray
from jepa.frame_pooler import FramePooler
from jepa.dynamics_predictor import LatentDynamicsPredictor


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


class JEPAGuidedPipeline(GeSimCosmos2Pipeline):
    def __init__(
        self,
        text_encoder,
        tokenizer,
        transformer,
        vae,
        scheduler,
        safety_checker=None,
    ):
        super().__init__(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )
        self.frame_pooler = None
        self.dynamics_predictor = None
        self.jepa_config = {}

    def set_jepa_modules(
        self,
        frame_pooler: FramePooler,
        dynamics_predictor: LatentDynamicsPredictor,
        extract_layer: int = 14,
        frame_stride: int = 1,
    ):
        self.frame_pooler = frame_pooler.eval()
        self.dynamics_predictor = dynamics_predictor.eval()
        self.jepa_config["extract_layer"] = int(extract_layer)
        self.jepa_config["frame_stride"] = max(1, int(frame_stride))

    def _compute_jepa_guidance(
        self,
        latents: torch.Tensor,
        conditioning_latents: torch.Tensor,
        cond_to_concat_resized: torch.Tensor,
        prompt_embeds: torch.Tensor,
        actions: torch.Tensor,
        current_t: torch.Tensor,
        cond_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        n_view: int,
        n_prev: int,
        n_fut: int,
        guidance_strength: float,
        transformer_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.frame_pooler is None or self.dynamics_predictor is None:
            return latents

        extract_layer = int(self.jepa_config.get("extract_layer", 14))
        guidance_frame_stride = max(1, int(self.jepa_config.get("frame_stride", 1)))
        patch_size = self.transformer.config.patch_size

        latents_for_guidance = latents.detach().clone().requires_grad_(True)
        c_in = 1 - current_t
        guided_latent = latents_for_guidance * c_in
        guided_latent = torch.cat([conditioning_latents, guided_latent], dim=2)

        guided_latent = torch.cat(
            [guided_latent, cond_to_concat_resized.to(device=guided_latent.device, dtype=guided_latent.dtype)],
            dim=1,
        )
        guided_latent = guided_latent.to(transformer_dtype)

        t_total_lat = n_prev + n_fut
        h_lat = latents.shape[-2]
        w_lat = latents.shape[-1]

        sigma_conditioning = 0.0001
        t_conditioning = torch.tensor(
            sigma_conditioning / (sigma_conditioning + 1),
            device=latents.device,
            dtype=transformer_dtype,
        )
        cond_indicator = latents.new_zeros(1, 1, t_total_lat, 1, 1)
        cond_indicator[:, :, :n_prev] = 1.0
        timestep = current_t.view(1, 1, 1, 1, 1).expand(
            latents.size(0), -1, t_total_lat, -1, -1
        ).clone().to(transformer_dtype)
        timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep

        output = self.transformer(
            hidden_states=guided_latent,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            fps=None,
            condition_mask=cond_mask,
            padding_mask=padding_mask,
            return_dict=False,
            height=h_lat,
            width=w_lat,
            n_view=n_view,
            num_frames=t_total_lat,
            return_video=False,
            store_buffer=True,
            store_buffer_indices=[extract_layer],
        )[0]

        video_states_buffer = output["video_states_buffer"]
        layer_repr = video_states_buffer[extract_layer]

        p_t, p_h, p_w = patch_size
        post_patch_frames = t_total_lat // p_t
        post_patch_height = h_lat // p_h
        post_patch_width = w_lat // p_w

        frame_repr = self.frame_pooler(
            layer_repr,
            num_frames=post_patch_frames,
            height=post_patch_height,
            width=post_patch_width,
        )
        future_repr = frame_repr[:, n_prev:]

        if actions is None or actions.shape[1] < future_repr.shape[1]:
            return latents

        actions_fut = actions[:, n_prev:].to(device=latents.device, dtype=torch.float32)
        if actions_fut.shape[1] != future_repr.shape[1]:
            actions_fut = actions_fut.permute(0, 2, 1)
            actions_fut = F.interpolate(actions_fut, size=future_repr.shape[1], mode="linear", align_corners=True)
            actions_fut = actions_fut.permute(0, 2, 1)
        if guidance_frame_stride > 1:
            future_repr = future_repr[:, ::guidance_frame_stride]
            actions_fut = actions_fut[:, ::guidance_frame_stride]
        if future_repr.shape[1] <= 1:
            return latents

        predicted = self.dynamics_predictor(future_repr[:, :-1], actions_fut[:, :-1])
        consistency_loss = F.mse_loss(predicted, future_repr[:, 1:])
        grad = torch.autograd.grad(consistency_loss, latents_for_guidance, retain_graph=False)[0]
        grad_norm = grad.norm(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-8)
        latents = latents - guidance_strength * (grad / grad_norm)
        return latents

    @torch.no_grad()
    def infer(
        self,
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,
        cond_to_concat: List[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        fps: int = 16,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        sigma_conditioning: float = 0.0001,
        n_view: int = 3,
        n_prev: int = 4,
        merge_view_into_width: bool = False,
        postprocess_video: bool = True,
        show_progress: bool = False,
        actions: Optional[torch.Tensor] = None,
        jepa_guidance_strength: float = 1.0,
        jepa_guidance_start_pct: float = 0.2,
        jepa_guidance_end_pct: float = 0.8,
        jepa_guidance_every_n: int = 3,
        **kwargs,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        assert num_videos_per_prompt == 1
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        if self.scheduler.config.final_sigmas_type == "sigma_min":
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        if image is not None:
            video = self.video_processor.preprocess(image, height, width).unsqueeze(2)
        else:
            video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=vae_dtype)

        num_channels_latents = self.vae.z_dim
        if video.shape[2] > n_prev:
            video = video[:, :, :n_prev]
        latents, conditioning_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask = self.prepare_latents(
            video,
            batch_size * n_view,
            num_channels_latents,
            height,
            width,
            num_frames,
            self.do_classifier_free_guidance,
            torch.float32,
            device,
            generator,
            latents,
        )

        cond_mask = cond_mask.to(transformer_dtype)
        unconditioning_latents = None
        if self.do_classifier_free_guidance:
            uncond_mask = uncond_mask.to(transformer_dtype)
            unconditioning_latents = conditioning_latents

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        sigma_conditioning_t = torch.tensor(sigma_conditioning, dtype=torch.float32, device=device)
        t_conditioning = sigma_conditioning_t / (sigma_conditioning_t + 1)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        jepa_enabled = (
            self.frame_pooler is not None
            and self.dynamics_predictor is not None
            and actions is not None
            and jepa_guidance_strength > 0
        )
        guide_start_step = int(num_inference_steps * jepa_guidance_start_pct)
        guide_end_step = int(num_inference_steps * jepa_guidance_end_pct)
        n_fut = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        cond_to_concat = cond_to_concat.to(device=conditioning_latents.device, dtype=conditioning_latents.dtype)
        cond_to_concat_resized = resize_traj_and_ray(
            cond_to_concat,
            mem_size=n_prev,
            future_size=n_fut,
            height=conditioning_latents.shape[-2],
            width=conditioning_latents.shape[-1],
        )

        if not show_progress:
            self.set_progress_bar_config(disable=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]
                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t

                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2) + conditioning_latents.size(2), -1, -1
                )

                cond_latent = latents * c_in
                cond_latent = torch.cat([conditioning_latents, cond_latent], dim=2)
                cond_latent = torch.cat([cond_latent, cond_to_concat_resized.to(device=cond_latent.device, dtype=cond_latent.dtype)], dim=1)
                cond_latent = cond_latent.to(transformer_dtype)
                cond_timestep = (cond_indicator * t_conditioning + (1 - cond_indicator) * timestep).to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=cond_latent,
                    timestep=cond_timestep,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps,
                    condition_mask=cond_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                    height=cond_latent.shape[-2],
                    width=cond_latent.shape[-1],
                    n_view=n_view,
                    num_frames=n_fut + n_prev,
                    return_video=True,
                )[0]["video"]

                noise_pred = noise_pred[:, :, n_prev:]
                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)

                if self.do_classifier_free_guidance:
                    uncond_latent = latents * c_in
                    uncond_latent = torch.cat([unconditioning_latents, uncond_latent], dim=2)
                    uncond_latent = torch.cat(
                        [uncond_latent, cond_to_concat.to(device=cond_latent.device, dtype=cond_latent.dtype)],
                        dim=1,
                    )
                    uncond_latent = uncond_latent.to(transformer_dtype)
                    uncond_timestep = (uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep).to(transformer_dtype)
                    noise_pred_uncond = self.transformer(
                        hidden_states=uncond_latent,
                        timestep=uncond_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        fps=fps,
                        condition_mask=uncond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                        n_view=n_view,
                    )[0]
                    noise_pred_uncond = noise_pred_uncond[:, :, n_prev:]
                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                noise_pred_velocity = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(noise_pred_velocity, t, latents, return_dict=False)[0]

                should_guide = (
                    jepa_enabled
                    and guide_start_step <= i <= guide_end_step
                    and (i - guide_start_step) % max(1, int(jepa_guidance_every_n)) == 0
                )
                if should_guide:
                    with torch.enable_grad():
                        latents = self._compute_jepa_guidance(
                            latents=latents,
                            conditioning_latents=conditioning_latents,
                            cond_to_concat_resized=cond_to_concat_resized,
                            prompt_embeds=prompt_embeds,
                            actions=actions,
                            current_t=current_t,
                            cond_mask=cond_mask,
                            padding_mask=padding_mask,
                            n_view=n_view,
                            n_prev=n_prev,
                            n_fut=n_fut,
                            guidance_strength=jepa_guidance_strength,
                            transformer_dtype=transformer_dtype,
                        )

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type != "latent":
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            if merge_view_into_width:
                video = rearrange(video, "(b v) c t h w -> b c t h (v w)", v=n_view)
            if postprocess_video:
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return {"frames": video}
