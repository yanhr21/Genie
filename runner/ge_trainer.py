import os, random, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import nullcontext

from datetime import datetime, timedelta
import argparse
import json
import importlib
import cv2
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib

from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
import torch.nn.functional as F
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset as TorchIterableDataset
from utils import init_logging, import_custom_class, save_video

# ----------------------------------------------------
from utils.data_utils import (
    _normalize_latents,
    get_latents,
    get_text_conditions,
    gen_noise_from_condition_frame_latent,
    randn_tensor,
    apply_color_jitter_to_video,
    unpack_latents,
)
from utils.geometry_utils import resize_traj_and_ray
from utils.jepa_utils import (
    JEPADynamicsHelper,
    load_jepa_backend,
)
from models.pipeline.jepa_guided_pipeline import JEPAGuidedPipeline

# ----------------------------------------------------
from utils.extra_utils import act_metric

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
logger = get_logger("wm_runner")
logger.setLevel(LOG_LEVEL)


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None



class Trainer:

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')
        self.state = State()

        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        self.jepa_backend_type = None
        self.jepa_frame_pooler = None
        self.jepa_dynamics_predictor = None
        self.jepa_helper = None
        self.jepa_config = {}

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if self.state.accelerator.is_main_process:

            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for k, v in args_dict.items():
                args_dict[k] = str(v)
            with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)
            
            if to_log:
                self.writer = SummaryWriter(log_dir=self.save_folder)
            else:
                self.writer = None

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

        if bool(getattr(self.args, "use_rank_log_file", False)):
            init_logging(self.save_folder, rank=self.state.accelerator.process_index)
        self._debug_first_batch_dumped = False


    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_find_unused_parameters = bool(getattr(self.args, "ddp_find_unused_parameters", False))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused_parameters)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))  # 或 self.args.world_size
            grad_accum = self.args.gradient_accumulation_steps

            train_batch_size = per_device_bs * world_size * grad_accum
            self.args.deepspeed["train_batch_size"] = train_batch_size
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype


    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)
        

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir

    def _default_prompt(self, split="train"):
        prompt = None
        try:
            split_cfg = self.args.data.get(split, {}) if isinstance(self.args.data, dict) else {}
            if isinstance(split_cfg, dict):
                prompt = split_cfg.get("unified_prompt", None)
        except Exception:
            prompt = None
        if not prompt:
            prompt = "A robotic arm performing a manipulation task, best quality, consistent and smooth motion."
        return prompt

    def _get_jepa_cfg(self) -> Dict[str, Any]:
        cfg = getattr(self.args, "jepa", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _prepare_jepa_modules(self):
        cfg = self._get_jepa_cfg()
        ckpt_path = str(cfg.get("ckpt_path", "") or "")
        if not ckpt_path:
            logger.info("JEPA disabled: no ckpt_path configured.")
            return
        try:
            backend, loaded_cfg = load_jepa_backend(
                ckpt_path,
                device=self.state.accelerator.device,
                dtype=self.state.weight_dtype,
            )
            self.jepa_backend_type = str(backend.get("backend_type", "legacy"))
            self.jepa_config = dict(loaded_cfg or {})
            self.jepa_config.update(cfg)
            self.jepa_frame_pooler = backend.get("frame_pooler", None)
            self.jepa_dynamics_predictor = backend.get("dynamics_predictor", None)
            if self.jepa_backend_type == "official_vjepa2_ac":
                self.jepa_helper = backend["helper"]
            else:
                self.jepa_helper = JEPADynamicsHelper(
                    vae=self.vae,
                    transformer=self.diffusion_model,
                    scheduler=self.scheduler,
                    frame_pooler=self.jepa_frame_pooler,
                    dynamics_predictor=self.jepa_dynamics_predictor,
                    weight_dtype=self.state.weight_dtype,
                    sigma_conditioning=float(getattr(self.args, "sigma_conditioning", 0.0001)),
                    noise_to_first_frame=float(getattr(self.args, "noise_to_first_frame", 0.05)),
                    pixel_wise_timestep=bool(getattr(self.args, "pixel_wise_timestep", False)),
                    extract_layer=int(self.jepa_config.get("extract_layer", 14)),
                )
            logger.info(f"JEPA enabled ({self.jepa_backend_type}) from checkpoint: {ckpt_path}")
        except Exception as e:
            self.jepa_backend_type = None
            logger.warning(f"Failed to initialize JEPA modules from {ckpt_path}: {repr(e)}")
            self.jepa_frame_pooler = None
            self.jepa_dynamics_predictor = None
            self.jepa_helper = None
            self.jepa_config = {}

    def _attach_jepa_to_pipe(self, pipe):
        if pipe is None or self.jepa_backend_type != "legacy":
            return
        if hasattr(pipe, "set_jepa_modules"):
            extract_layer = int(self.jepa_config.get("extract_layer", self._get_jepa_cfg().get("extract_layer", 14)))
            frame_stride = int(
                self.jepa_config.get(
                    "frame_stride",
                    self._get_jepa_cfg().get("infer", {}).get("frame_stride", self._get_jepa_cfg().get("frame_stride", 1)),
                )
            )
            pipe.set_jepa_modules(
                self.jepa_frame_pooler,
                self.jepa_dynamics_predictor,
                extract_layer=extract_layer,
                frame_stride=frame_stride,
            )

    @staticmethod
    def _normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if x.numel() == 0:
            return x
        x_min = float(x.amin().item())
        x_max = float(x.amax().item())
        if x_max - x_min < 1e-8:
            return torch.zeros_like(x)
        x = (x - x_min) / (x_max - x_min)
        return x * 2.0 - 1.0

    @staticmethod
    def _build_first_frame_only_indices(total_t: int, mem_size: int) -> List[int]:
        total_t = max(1, int(total_t))
        mem_size = max(1, int(mem_size))
        future_len = max(0, total_t - mem_size)
        indices = [0] * mem_size
        indices.extend(min(i, total_t - 1) for i in range(1, 1 + future_len))
        if len(indices) < total_t:
            indices.extend([indices[-1]] * (total_t - len(indices)))
        return indices[:total_t]

    def _apply_first_frame_only_timeline(self, tensor: torch.Tensor, mem_size: int) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
            return tensor
        total_t = int(tensor.shape[2])
        idx = self._build_first_frame_only_indices(total_t=total_t, mem_size=mem_size)
        idx_t = torch.tensor(idx, device=tensor.device, dtype=torch.long)
        return tensor.index_select(2, idx_t)

    def _decode_video_latents_for_loss(self, latents: torch.Tensor) -> torch.Tensor:
        scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
        latents = _normalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            scaling_factor,
            reverse=True,
        )
        video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        return torch.nan_to_num(video.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

    @staticmethod
    def _spatial_edge_map(video: torch.Tensor) -> torch.Tensor:
        gray = video.mean(dim=1, keepdim=True)
        grad_x = gray[:, :, :, :, 1:] - gray[:, :, :, :, :-1]
        grad_y = gray[:, :, :, 1:, :] - gray[:, :, :, :-1, :]
        pad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0))
        pad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        return torch.cat([pad_x, pad_y], dim=1)

    def _compute_rollout_holistic_loss(
        self,
        pred_future_latents: torch.Tensor,
        gt_future_video: torch.Tensor,
    ):
        enabled = bool(getattr(self.args, "rollout_holistic_loss", False))
        rgb_weight = float(getattr(self.args, "rollout_holistic_rgb_weight", 0.0))
        temporal_weight = float(getattr(self.args, "rollout_holistic_temporal_weight", 0.0))
        edge_weight = float(getattr(self.args, "rollout_holistic_edge_weight", 0.0))
        decode_downsample = max(1, int(getattr(self.args, "rollout_holistic_decode_downsample", 1)))
        frame_stride = max(1, int(getattr(self.args, "rollout_holistic_frame_stride", 1)))

        zero = pred_future_latents.new_zeros(())
        if (not enabled) or (rgb_weight <= 0.0 and temporal_weight <= 0.0 and edge_weight <= 0.0):
            return zero, {}

        pred_latents = pred_future_latents.float()
        if decode_downsample > 1:
            scale = (1.0, 1.0 / decode_downsample, 1.0 / decode_downsample)
            pred_latents = F.interpolate(pred_latents, scale_factor=scale, mode="trilinear", align_corners=False)

        pred_video = self._decode_video_latents_for_loss(pred_latents)
        gt_video = torch.nan_to_num(gt_future_video.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        if gt_video.shape[2:] != pred_video.shape[2:]:
            gt_video = F.interpolate(gt_video, size=pred_video.shape[2:], mode="trilinear", align_corners=False)

        if frame_stride > 1:
            pred_video = pred_video[:, :, ::frame_stride]
            gt_video = gt_video[:, :, ::frame_stride]

        rgb_loss = zero
        if rgb_weight > 0.0:
            rgb_loss = F.l1_loss(pred_video, gt_video)

        temporal_loss = zero
        if temporal_weight > 0.0 and pred_video.shape[2] > 1 and gt_video.shape[2] > 1:
            pred_delta = pred_video[:, :, 1:] - pred_video[:, :, :-1]
            gt_delta = gt_video[:, :, 1:] - gt_video[:, :, :-1]
            temporal_loss = F.l1_loss(pred_delta, gt_delta)

        edge_loss = zero
        if edge_weight > 0.0:
            edge_loss = F.l1_loss(self._spatial_edge_map(pred_video), self._spatial_edge_map(gt_video))

        total = rgb_weight * rgb_loss + temporal_weight * temporal_loss + edge_weight * edge_loss
        stats = {
            "holistic_rgb": float(rgb_loss.detach().item()) if rgb_weight > 0.0 else 0.0,
            "holistic_temporal": float(temporal_loss.detach().item()) if temporal_weight > 0.0 else 0.0,
            "holistic_edge": float(edge_loss.detach().item()) if edge_weight > 0.0 else 0.0,
            "holistic_total": float(total.detach().item()),
        }
        return total, stats

    @staticmethod
    def _compute_psnr(pred_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        pred = pred_frame.astype(np.float32)
        gt = gt_frame.astype(np.float32)
        mse = float(np.mean((pred - gt) ** 2))
        if mse <= 1e-10:
            return 99.0
        return float(20.0 * math.log10(255.0 / math.sqrt(mse)))

    def _compute_video_metrics(self, pred_frames: np.ndarray, gt_dir: str) -> Dict[str, Any]:
        gt_path = Path(gt_dir)
        gt_files = sorted(list(gt_path.glob("*.png")) + list(gt_path.glob("*.jpg")) + list(gt_path.glob("*.jpeg")))
        if len(gt_files) == 0 or pred_frames.size == 0:
            return {}

        n = min(int(pred_frames.shape[0]), len(gt_files))
        all_psnr = []
        late_psnr = []
        last_quarter_psnr = []
        pred_delta_list = []
        gt_delta_list = []
        prev_pred = None
        prev_gt = None
        for i in range(n):
            gt = cv2.imread(str(gt_files[i]), cv2.IMREAD_COLOR)
            if gt is None:
                continue
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = pred_frames[i]
            if gt.shape[:2] != pred.shape[:2]:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
            score = self._compute_psnr(pred, gt)
            all_psnr.append(score)
            if i >= n // 2:
                late_psnr.append(score)
            if i >= (3 * n) // 4:
                last_quarter_psnr.append(score)
            if prev_pred is not None and prev_gt is not None:
                pred_delta_list.append(pred.astype(np.float32) - prev_pred.astype(np.float32))
                gt_delta_list.append(gt.astype(np.float32) - prev_gt.astype(np.float32))
            prev_pred = pred
            prev_gt = gt

        motion_l1 = None
        if len(pred_delta_list) > 0:
            pred_delta = np.stack(pred_delta_list, axis=0)
            gt_delta = np.stack(gt_delta_list, axis=0)
            motion_l1 = float(np.mean(np.abs(pred_delta - gt_delta)) / 255.0)

        return {
            "frames": int(len(all_psnr)),
            "avg_psnr": float(np.mean(all_psnr)) if len(all_psnr) > 0 else None,
            "late_half_psnr": float(np.mean(late_psnr)) if len(late_psnr) > 0 else None,
            "last_quarter_psnr": float(np.mean(last_quarter_psnr)) if len(last_quarter_psnr) > 0 else None,
            "motion_l1": motion_l1,
        }

    def _compute_jepa_rollout_loss(
        self,
        video: Optional[torch.Tensor],
        cond_to_concat: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        actions: Optional[torch.Tensor],
        mem_size: int,
        n_view: int = 1,
        frame_repr: Optional[torch.Tensor] = None,
        latent_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.jepa_helper is None or actions is None:
            ref = frame_repr if frame_repr is not None else latent_video
            if ref is None:
                ref = video
            return ref.new_zeros(())
        frame_stride = int(self._get_jepa_cfg().get("train", {}).get("frame_stride", 1))
        if self.jepa_backend_type == "official_vjepa2_ac" and video is None:
            if latent_video is None:
                raise ValueError("Official V-JEPA2 backend requires decoded RGB video for JEPA loss.")
            video = self._decode_video_latents_for_loss(latent_video)
        return self.jepa_helper.score_consistency(
            video=video,
            actions=actions,
            mem_size=mem_size,
            frame_stride=frame_stride,
            cond_to_concat=cond_to_concat,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            n_view=n_view,
            frame_repr=frame_repr,
            latent_video=latent_video,
            require_grad=True,
        )

    def _compute_jepa_validation_metrics(
        self,
        sample_dir: str,
        pred_frames: np.ndarray,
        gt_dir: str,
        sample_size: tuple,
        mem_size: int,
        first_frame_only: bool,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        if self.jepa_helper is None or pred_frames.size == 0:
            return {}
        try:
            from scripts.infer_iros2026_gesim import (
                _read_first_frame,
                compose_action_from_h5,
                load_extrinsics,
                load_intrinsic,
                compute_cond_to_concat,
            )
        except Exception as e:
            logger.warning(f"[Validation-Full] JEPA metric helper import failed: {repr(e)}")
            return {}

        gt_path = Path(gt_dir)
        gt_files = sorted(list(gt_path.glob("*.png")) + list(gt_path.glob("*.jpg")) + list(gt_path.glob("*.jpeg")))
        if len(gt_files) == 0:
            return {}

        n_eval = min(len(gt_files), int(pred_frames.shape[0]))
        if n_eval <= 0:
            return {}

        first_frame = _read_first_frame(sample_dir)
        ori_h, ori_w = first_frame.shape[:2]
        first_frame = cv2.resize(first_frame, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_AREA)

        gt_frames = []
        for idx in range(n_eval):
            gt = cv2.imread(str(gt_files[idx]), cv2.IMREAD_COLOR)
            if gt is None:
                continue
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = cv2.resize(gt, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_AREA)
            gt_frames.append(gt)
        n_eval = min(n_eval, len(gt_frames))
        if n_eval <= 0:
            return {}

        device = self.state.accelerator.device
        mem_np = np.repeat(first_frame[None, ...], mem_size, axis=0)
        pred_full = np.concatenate([mem_np, pred_frames[:n_eval]], axis=0)
        gt_full = np.concatenate([mem_np, np.stack(gt_frames[:n_eval], axis=0)], axis=0)

        pred_video = torch.from_numpy(pred_full).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
        gt_video = torch.from_numpy(gt_full).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
        pred_video = pred_video / 255.0 * 2.0 - 1.0
        gt_video = gt_video / 255.0 * 2.0 - 1.0

        h5_path = os.path.join(sample_dir, "proprio_stats.h5")
        ext_path = os.path.join(sample_dir, "head_extrinsic_params_aligned.json")
        int_path = os.path.join(sample_dir, "head_intrinsic_params.json")
        actions = compose_action_from_h5(h5_path)
        t_total = actions.shape[0]
        extrinsics = load_extrinsics(ext_path, t_total)
        intrinsic = load_intrinsic(int_path)
        scale_w = sample_size[1] / ori_w
        scale_h = sample_size[0] / ori_h
        intrinsic_scaled = intrinsic.clone()
        intrinsic_scaled[:, 0, 0] *= scale_w
        intrinsic_scaled[:, 0, 2] *= scale_w
        intrinsic_scaled[:, 1, 1] *= scale_h
        intrinsic_scaled[:, 1, 2] *= scale_h

        actions_t = torch.from_numpy(actions).float()
        cond_full = compute_cond_to_concat(actions_t, extrinsics.float(), intrinsic_scaled, sample_size)
        if first_frame_only:
            cond_indices = [0 for _ in range(mem_size)] + list(range(1, 1 + n_eval))
        else:
            cond_indices = list(range(mem_size + n_eval))
        cond_indices = [min(max(0, idx), cond_full.shape[2] - 1) for idx in cond_indices]
        cond_idx_t = torch.tensor(cond_indices, dtype=torch.long)
        cond = cond_full.index_select(2, cond_idx_t)
        cond = rearrange(cond, "c v t h w -> v c t h w").to(device)
        actions_seq = torch.from_numpy(actions[cond_indices]).float().unsqueeze(0).to(device)

        pred_repr = self.jepa_helper.extract_frame_repr(
            video=pred_video,
            cond_to_concat=cond,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            mem_size=mem_size,
            n_view=1,
            require_grad=False,
        )
        gt_repr = self.jepa_helper.extract_frame_repr(
            video=gt_video,
            cond_to_concat=cond,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            mem_size=mem_size,
            n_view=1,
            require_grad=False,
        )
        pred_future = pred_repr[:, mem_size:]
        gt_future = gt_repr[:, mem_size:]
        if pred_future.shape[1] == 0 or gt_future.shape[1] == 0:
            return {}

        jepa_dyn = self.jepa_helper.score_consistency(
            video=pred_video,
            actions=actions_seq,
            mem_size=mem_size,
            frame_stride=int(self._get_jepa_cfg().get("metric", {}).get("frame_stride", 1)),
            cond_to_concat=cond,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            n_view=1,
            frame_repr=None if self.jepa_backend_type == "official_vjepa2_ac" else pred_repr,
            require_grad=False,
        )
        repr_l2 = F.mse_loss(pred_future.float(), gt_future.float())
        late_start = pred_future.shape[1] // 2
        last_quarter_start = (3 * pred_future.shape[1]) // 4
        late_repr_l2 = F.mse_loss(pred_future[:, late_start:].float(), gt_future[:, late_start:].float())
        last_quarter_repr_l2 = F.mse_loss(pred_future[:, last_quarter_start:].float(), gt_future[:, last_quarter_start:].float())
        return {
            "jepa_dyn_loss": float(jepa_dyn.detach().item()),
            "jepa_repr_l2": float(repr_l2.detach().item()),
            "jepa_late_half_repr_l2": float(late_repr_l2.detach().item()),
            "jepa_last_quarter_repr_l2": float(last_quarter_repr_l2.detach().item()),
        }

    def _dump_train_batch_decode(self, batch: Dict[str, Any], global_step: int, epoch: int, step: int) -> None:
        if not self.state.accelerator.is_main_process:
            return
        if "video" not in batch:
            logger.warning("[Train-Debug] batch has no `video`, skip debug decode.")
            return

        debug_dir = os.path.join(self.save_folder, "debug_train_decode", f"step_{int(global_step):07d}")
        os.makedirs(debug_dir, exist_ok=True)

        try:
            n_samples = int(getattr(self.args, "debug_train_decode_n_samples", 1))
        except Exception:
            n_samples = 1
        n_samples = max(1, n_samples)
        save_cond = bool(getattr(self.args, "debug_train_decode_save_cond", True))

        fps = int(
            getattr(self.args, "basic_fps", 30)
            / (self.args.data["train"]["action_chunk"] // self.args.data["train"]["chunk"])
        )

        videos = batch["video"]
        if not isinstance(videos, torch.Tensor):
            logger.warning("[Train-Debug] batch['video'] is not tensor, skip debug decode.")
            return

        n = min(n_samples, int(videos.shape[0]))
        cond = batch.get("cond_to_concat", None)

        for i in range(n):
            video_i = videos[i].detach().float().cpu()
            finite_ratio = float(torch.isfinite(video_i).float().mean().item())
            vmin = float(torch.nan_to_num(video_i, nan=0.0, posinf=0.0, neginf=0.0).amin().item())
            vmax = float(torch.nan_to_num(video_i, nan=0.0, posinf=0.0, neginf=0.0).amax().item())
            video_to_save = torch.nan_to_num(video_i, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            video_to_save = rearrange(video_to_save, "c v t h w -> c t h (v w)")
            video_path = os.path.join(debug_dir, f"train_e{epoch:03d}_iter{step:06d}_b{i}_video.mp4")
            try:
                save_video(video_to_save, video_path, fps=fps)
            except Exception as e:
                logger.warning(f"[Train-Debug] failed to save train video sample {i}: {repr(e)}")
                continue

            logger.info(
                f"[Train-Debug] saved sample={i}, step={global_step}, epoch={epoch}, iter={step}, "
                f"video_shape={tuple(video_i.shape)}, finite_ratio={finite_ratio:.6f}, min={vmin:.6f}, max={vmax:.6f}, "
                f"path={video_path}"
            )

            sample_dir_i = None
            frame_indices_i = None
            action_indices_i = None
            sample_dirs = batch.get("sample_dir", None)
            frame_indices = batch.get("frame_indices", None)
            action_indices = batch.get("action_indices", None)
            try:
                if isinstance(sample_dirs, (list, tuple)) and len(sample_dirs) > i:
                    sample_dir_i = str(sample_dirs[i])
                elif isinstance(sample_dirs, str):
                    sample_dir_i = sample_dirs
            except Exception:
                sample_dir_i = None
            try:
                if isinstance(frame_indices, torch.Tensor) and frame_indices.ndim >= 2 and frame_indices.shape[0] > i:
                    frame_indices_i = [int(x) for x in frame_indices[i].detach().cpu().tolist()]
                elif isinstance(frame_indices, (list, tuple)) and len(frame_indices) > i:
                    item = frame_indices[i]
                    if isinstance(item, torch.Tensor):
                        frame_indices_i = [int(x) for x in item.detach().cpu().tolist()]
            except Exception:
                frame_indices_i = None
            try:
                if isinstance(action_indices, torch.Tensor) and action_indices.ndim >= 2 and action_indices.shape[0] > i:
                    action_indices_i = [int(x) for x in action_indices[i].detach().cpu().tolist()]
                elif isinstance(action_indices, (list, tuple)) and len(action_indices) > i:
                    item = action_indices[i]
                    if isinstance(item, torch.Tensor):
                        action_indices_i = [int(x) for x in item.detach().cpu().tolist()]
            except Exception:
                action_indices_i = None
            if sample_dir_i is not None or frame_indices_i is not None or action_indices_i is not None:
                meta = {
                    "sample_dir": sample_dir_i,
                    "frame_indices": frame_indices_i,
                    "action_indices": action_indices_i,
                }
                meta_path = os.path.join(debug_dir, f"train_e{epoch:03d}_iter{step:06d}_b{i}_meta.json")
                try:
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                except Exception as e:
                    logger.warning(f"[Train-Debug] failed to save meta sample {i}: {repr(e)}")

            if save_cond and isinstance(cond, torch.Tensor) and cond.shape[0] > i:
                cond_i = cond[i].detach().float().cpu()
                if cond_i.ndim == 5 and cond_i.shape[0] >= 3:
                    traj = torch.nan_to_num(cond_i[:3], nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                    traj = rearrange(traj, "c v t h w -> c t h (v w)")
                    traj_path = os.path.join(debug_dir, f"train_e{epoch:03d}_iter{step:06d}_b{i}_traj.mp4")
                    try:
                        save_video(traj, traj_path, fps=fps)
                    except Exception as e:
                        logger.warning(f"[Train-Debug] failed to save traj sample {i}: {repr(e)}")

                if cond_i.ndim == 5 and cond_i.shape[0] >= 9:
                    rays_o = self._normalize_to_minus1_1(cond_i[3:6])
                    rays_d = self._normalize_to_minus1_1(cond_i[6:9])
                    rays_o = rearrange(rays_o, "c v t h w -> c t h (v w)")
                    rays_d = rearrange(rays_d, "c v t h w -> c t h (v w)")
                    rays_o_path = os.path.join(debug_dir, f"train_e{epoch:03d}_iter{step:06d}_b{i}_rays_o.mp4")
                    rays_d_path = os.path.join(debug_dir, f"train_e{epoch:03d}_iter{step:06d}_b{i}_rays_d.mp4")
                    try:
                        save_video(rays_o, rays_o_path, fps=fps)
                        save_video(rays_d, rays_d_path, fps=fps)
                    except Exception as e:
                        logger.warning(f"[Train-Debug] failed to save rays sample {i}: {repr(e)}")


    def prepare_dataset(self) -> None:

        logger.info(f"Training Dataset: {self.args.train_data_class}")

        train_dataset_class = import_custom_class(
            self.args.train_data_class, self.args.train_data_class_path
        )
        self.train_dataset = train_dataset_class(**self.args.data['train'])
        is_iterable_train = isinstance(self.train_dataset, TorchIterableDataset)

        dataloader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.dataloader_num_workers,
            multiprocessing_context=None,
            pin_memory=True,
        )
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2
        if not is_iterable_train:
            dataloader_kwargs["shuffle"] = True

        self.train_dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)
        logger.info(f">>>>>>>>>>>>>Total Train Eps: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")
        if is_iterable_train:
            logger.info("[Trainer] train dataset is IterableDataset (stream mode)")


        if 'val' in self.args.data:
            val_cfg = getattr(self.args, "val", {})
            full_video_val = isinstance(val_cfg, dict) and bool(val_cfg.get("full_video", False))
            if full_video_val:
                logger.info("[Trainer] Full-video validation enabled; skip val dataloader construction.")
                self.val_dataset = None
                self.val_dataloader = None
            else:
                self.prepare_val_dataset()


    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        self.val_dataset = val_dataset_class(**self.args.data['val'])

        self.val_index = []
        for _ in range(self.args.batch_size):
            self.val_index.append(random.randint(0, len(self.val_dataset)-1))
        if self.state.accelerator.is_main_process:
            with open(os.path.join(self.save_folder, 'idx.txt'), "w") as file:
                file.write(", ".join(map(str, self.val_index)))

        subset = torch.utils.data.Subset(self.val_dataset, self.val_index)
        self.val_dataloader = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size, shuffle=getattr(self.args, "val_shuffle", False)
        )
        logger.info(f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")


    def prepare_models(self):

        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=True
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        logger.info(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        logger.info(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        logger.info(f'Total parameters for transformer model:{total_params}')


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        scheduler_dir = os.path.join(str(self.args.pretrained_model_name_or_path), "scheduler")
        if hasattr(diffusion_scheduler_class, "from_pretrained") and os.path.isdir(scheduler_dir):
            self.scheduler = diffusion_scheduler_class.from_pretrained(scheduler_dir)
            logger.info(f"Loaded diffusion scheduler from pretrained config: {scheduler_dir}")
        elif hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
            logger.info("Loaded diffusion scheduler from `diffusion_scheduler_args`.")
        else:
            self.scheduler = diffusion_scheduler_class()
            logger.info("Loaded diffusion scheduler with class defaults.")

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )
        self._prepare_jepa_modules()


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        components_to_disable_grads = []
            
        for component in components_to_disable_grads:
            if component is not None:
                component.requires_grad_(False)

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True


    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        train_mode = self.args.train_mode

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        diffusion_model_trainable_params = []
        if train_mode == 'action_only':
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "video_only":
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' not in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "all" or train_mode == 'action_full':
            for name, param in self.diffusion_model.named_parameters():
                param.requires_grad = True
                diffusion_model_trainable_params.append(param)
        elif train_mode == "lora" or train_mode == "video_lora":
            # LoRA mode for Cosmos GE-Sim: only train AdaLN LoRA projections.
            lora_trainable_keywords = getattr(
                self.args,
                "lora_trainable_keywords",
                ["norm1.linear_", "norm2.linear_", "norm3.linear_", "norm_out.linear_"],
            )
            lora_exclude_keywords = getattr(self.args, "lora_exclude_keywords", [])
            if isinstance(lora_trainable_keywords, str):
                lora_trainable_keywords = [lora_trainable_keywords]
            if isinstance(lora_exclude_keywords, str):
                lora_exclude_keywords = [lora_exclude_keywords]

            trainable_param_names = []
            for name, param in self.diffusion_model.named_parameters():
                is_lora_target = any(key in name for key in lora_trainable_keywords)
                is_excluded = any(key in name for key in lora_exclude_keywords)
                param.requires_grad = bool(is_lora_target and not is_excluded)
                if param.requires_grad:
                    diffusion_model_trainable_params.append(param)
                    trainable_param_names.append(name)

            if len(diffusion_model_trainable_params) == 0:
                raise ValueError(
                    f"No trainable parameters matched LoRA keywords: {lora_trainable_keywords}"
                )
            logger.info(
                f"LoRA mode enabled. keywords={lora_trainable_keywords}, "
                f"exclude={lora_exclude_keywords}, matched={len(trainable_param_names)} tensors"
            )
            logger.info(f"LoRA trainable tensors (first 20): {trainable_param_names[:20]}")
        else:
            raise NotImplementedError

        num_trainable_params = sum(p.numel() for p in diffusion_model_trainable_params)
        logger.info(f'Total trainable parameters: {num_trainable_params}')

        diffusion_model_parameters_with_lr = {
            "params": diffusion_model_trainable_params,
            "lr": self.state.learning_rate,
        }
        params_to_optimize = [diffusion_model_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in diffusion_model_trainable_params)

        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        

    def prepare_for_training(self):
        self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        if self.jepa_backend_type == "legacy" and self.jepa_helper is not None:
            self.jepa_helper.transformer = self.diffusion_model


    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)

    def _broadcast_int_from_main(self, value: int) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return int(value)

        device = self.state.accelerator.device
        if self.state.accelerator.is_main_process:
            tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
        else:
            tensor = torch.zeros(1, device=device, dtype=torch.long)
        dist.broadcast(tensor, src=0)
        return int(tensor.item())


    def train(self):
        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        
        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        # Run an initial validation at step 0 before any optimization step.
        if getattr(self.args, "steps_to_val", 0) and int(self.args.steps_to_val) > 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model_save_dir = os.path.join(self.save_folder, "Validation_step_0")
                self.validate(accelerator, model_save_dir, global_step=0, n_view=1, n_chunk=1)
            accelerator.wait_for_everyone()

        # loss spikes
        anomalies = []
        is_video_train_mode = self.args.train_mode in {"all", "video_only", "lora", "video_lora"}
        is_action_train_mode = self.args.train_mode in {"all", "action_only", "action_full"}

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)

            self.diffusion_model.train()

            running_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                if (
                    accelerator.is_main_process
                    and bool(getattr(self.args, "debug_train_decode", False))
                    and bool(getattr(self.args, "debug_train_decode_first_batch", True))
                    and (not self._debug_first_batch_dumped)
                    and epoch == 0
                    and step == 0
                ):
                    self._dump_train_batch_decode(batch=batch, global_step=global_step, epoch=epoch, step=step)
                    self._debug_first_batch_dumped = True

                with accelerator.accumulate(self.diffusion_model):
                    
                    video = batch['video']
                    sim_type = str(getattr(self.args, "sim_type", "default")).lower()
                    use_gesim_cond = sim_type in {"gesim", "ge-sim", "sim"}
                    cond_to_concat = None
                    jepa_actions_full = None

                    # shape: {b, c, v, t, h, w}; ranging from -1 to 1
                    video = video.to(accelerator.device, dtype=weight_dtype).contiguous()
                    batch_size, c, n_view, _, h, w = video.shape
                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')
                    if use_gesim_cond:
                        if "cond_to_concat" not in batch:
                            raise KeyError("GE-Sim mode requires `cond_to_concat` in batch.")
                        cond_to_concat = batch["cond_to_concat"].to(
                            accelerator.device, dtype=weight_dtype
                        ).contiguous()
                        cond_to_concat = rearrange(cond_to_concat, "b c v t h w -> (b v) c t h w")
                    if "actions" in batch and isinstance(batch["actions"], torch.Tensor):
                        jepa_actions_full = batch["actions"].to(accelerator.device, dtype=weight_dtype).contiguous()

                    # here we use color jitter to the video, with different views or different batches different jitter
                    if self.args.use_color_jitter:
                        video = apply_color_jitter_to_video(video)

                    mem_size = int(self.args.data['train']['n_previous'])
                    chunk_size = int(self.args.data['train']['chunk'])
                    train_first_frame_only_prob = float(getattr(self.args, "train_first_frame_only_prob", 0.0))
                    train_first_frame_only = False
                    if (
                        train_first_frame_only_prob > 0.0
                        and is_video_train_mode
                        and (not is_action_train_mode)
                        and (not self.args.return_action)
                    ):
                        local_train_fff = 1 if random.random() < train_first_frame_only_prob else 0
                        train_first_frame_only = self._broadcast_int_from_main(local_train_fff) > 0
                        if train_first_frame_only:
                            video = self._apply_first_frame_only_timeline(video, mem_size=mem_size)
                            if cond_to_concat is not None:
                                cond_to_concat = self._apply_first_frame_only_timeline(cond_to_concat, mem_size=mem_size)

                    total_frames_in_batch = int(video.shape[2])
                    max_rollout_chunks = max(1, (total_frames_in_batch - mem_size) // max(chunk_size, 1))
                    self_forcing_enabled = bool(getattr(self.args, "self_forcing", False))
                    self_forcing_rollout_chunks = max(1, int(getattr(self.args, "self_forcing_rollout_chunks", 1)))
                    self_forcing_prob = float(getattr(self.args, "self_forcing_prob", 1.0))
                    self_forcing_detach_mem = bool(getattr(self.args, "self_forcing_detach_mem", True))
                    self_forcing_random_step = bool(getattr(self.args, "self_forcing_random_step", False))
                    self_forcing_last_step_only = bool(getattr(self.args, "self_forcing_last_step_only", False))

                    sf_chunks = 1
                    sf_rollout_candidate = (
                        self_forcing_enabled
                        and is_video_train_mode
                        and (not is_action_train_mode)
                        and (not self.args.return_action)
                        and self_forcing_rollout_chunks > 1
                        and max_rollout_chunks > 1
                    )
                    if sf_rollout_candidate:
                        sf_enable = 1 if random.random() < self_forcing_prob else 0
                        sf_enable = self._broadcast_int_from_main(sf_enable)
                        if sf_enable > 0:
                            sf_chunks = min(self_forcing_rollout_chunks, max_rollout_chunks)

                    sf_exit_chunk = 0
                    if sf_chunks > 1 and self_forcing_random_step:
                        if self_forcing_last_step_only:
                            local_exit_chunk = sf_chunks - 1
                        else:
                            local_exit_chunk = random.randint(0, sf_chunks - 1)
                        sf_exit_chunk = self._broadcast_int_from_main(local_exit_chunk)

                    video_full = video
                    cond_to_concat_full = cond_to_concat
                    if sf_chunks > 1:
                        # First chunk uses GT memory; later chunks use self-forcing memory.
                        end0 = mem_size + chunk_size
                        video = video_full[:, :, :end0]
                        if cond_to_concat_full is not None:
                            cond_to_concat = cond_to_concat_full[:, :, :end0]

                    mem = video[:,:,:mem_size]
                    future_video = video[:,:,mem_size:]

                    if self.args.return_action:
                        future_video = future_video[:,:,:1].repeat(1,1,self.args.data['train']['chunk'],1,1)

                    # get the shape params
                    _, _, raw_frames, raw_height, raw_width = future_video.shape

                    latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
                    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
                    latent_width = raw_width // self.SPATIAL_DOWN_RATIO

                    dropout_factor = torch.rand(batch_size).to(accelerator.device, dtype=weight_dtype)
                    dropout_mask_prompt = dropout_factor < self.args.caption_dropout_p
                    dropout_mask_prompt = dropout_mask_prompt.unsqueeze(1).unsqueeze(2)

                    mem_latents, future_video_latents = get_latents(
                        self.vae, mem, future_video
                    )

                    mem_latents = rearrange(mem_latents, '(b v m) (h w) c -> (b v) c m h w', b=batch_size, m=mem_size, h=latent_height)
                    future_video_latents = rearrange(future_video_latents, '(b v) (f h w) c -> (b v) c f h w',b=batch_size,h=latent_height,w=latent_width)
                    latents = torch.cat((mem_latents, future_video_latents), dim=2)

                    video_attention_mask = None
                    latents = rearrange(latents, 'bv c f h w -> bv (f h w) c')
                    cond_to_concat_flat = None
                    if use_gesim_cond and cond_to_concat is not None:
                        # GE-Sim adds trajectory+ray condition channels (9ch) at latent resolution.
                        n_fut = latent_frames - mem_size
                        cond_to_concat_resized = resize_traj_and_ray(
                            cond_to_concat,
                            mem_size=mem_size,
                            future_size=n_fut,
                            height=latent_height,
                            width=latent_width,
                        )
                        cond_to_concat_flat = rearrange(
                            cond_to_concat_resized, "bv c t h w -> bv (t h w) c"
                        )

                    captions = batch.get('caption', None)
                    if captions is None:
                        captions = [self._default_prompt("train")] * batch_size
                    elif isinstance(captions, str):
                        captions = [captions] * batch_size
                    else:
                        captions = list(captions)
                        if len(captions) < batch_size:
                            fill = captions[-1] if len(captions) > 0 else self._default_prompt("train")
                            captions = captions + [fill] * (batch_size - len(captions))
                        elif len(captions) > batch_size:
                            captions = captions[:batch_size]
                    text_conds = get_text_conditions(self.tokenizer,self.text_encoder,captions)
                    prompt_embeds = text_conds['prompt_embeds']
                    prompt_attention_mask = text_conds['prompt_attention_mask']
                    prompt_embeds = self.uncond_prompt_embeds.repeat(batch_size,1,1)*dropout_mask_prompt + \
                                    prompt_embeds*~dropout_mask_prompt

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    action_weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    # 0-1, 0 -> most noisy, 1 -> almost clean
                    action_indices = (action_weights * self.scheduler.config.num_train_timesteps).long()
                    action_sigmas = scheduler_sigmas[action_indices]
                    action_timesteps = (action_sigmas * 1000.0).long()

                    if self.args.return_action and self.args.noisy_video:
                        weights = torch.full_like(action_weights, 0.0).unsqueeze(1).repeat(1,n_view)
                    else:
                        weights = action_weights.unsqueeze(1).repeat(1,n_view)

                    weights = rearrange(weights, 'b v -> (b v)')
                    indices = (weights * self.scheduler.config.num_train_timesteps).long()
                    sigmas = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()

                    if self.args.return_action:
                        if getattr(self.args, "add_state", False):
                            # NOTE add states from the batch:
                            act_state = batch['state']
                            if act_state.shape[1] != 1:
                                act_state = act_state[:, mem_size-1:mem_size]
                            act_state = act_state.to(accelerator.device, dtype=weight_dtype).contiguous()
                        else:
                            act_state = None
                            

                        actions = batch['actions'][:, -self.args.data['train']['action_chunk']:].to(accelerator.device, dtype=weight_dtype).contiguous()   # shape b,t,c
                        action_dim = actions.shape[-1]

                        noise_actions = randn_tensor(actions.shape, device=accelerator.device, dtype=weight_dtype)

                        # here we get action_timesteps, shape (b,) originally, target shape (b, l) 
                        action_timesteps = action_timesteps.unsqueeze(-1).repeat(1, actions.shape[1])
                        action_ss= action_sigmas.reshape(-1, 1, 1).repeat(1, 1, actions.shape[-1])

                        noisy_actions = (1.0 - action_ss) * actions + action_ss * noise_actions

                        action_weights = compute_loss_weighting_for_sd3(
                            weighting_scheme=self.args.flow_weighting_scheme, sigmas=action_sigmas
                        ).reshape(-1, 1, 1).repeat(1, 1, actions.size(-1))
                    else:
                        actions = None
                        action_timesteps = None
                        noisy_actions = None
                        act_state = None

                    # shape:  bv, l, c and bv, l
                    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                        mem_latents, latent_frames, latent_height, latent_width, noise_to_condition_frames=self.args.noise_to_first_frame
                    )  # set initial frames noise to 0
                    if self.args.pixel_wise_timestep:
                        # shape: bv, thw
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        # shape: bv, t
                        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

                    # shape: bv,1,c
                    ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    if self.args.return_action and self.args.noisy_video:
                        ss = torch.full_like(ss, 1.0)

                    noisy_latents_core = (1.0 - ss) * latents + ss * noise
                    noisy_latents = noisy_latents_core
                    if cond_to_concat_flat is not None:
                        noisy_latents = torch.cat([noisy_latents, cond_to_concat_flat], dim=-1)

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss, shape bv,1,c
                    weights = compute_loss_weighting_for_sd3(
                        weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas
                    ).reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))

                    run_chunk0_with_grad = True
                    if (
                        sf_chunks > 1
                        and self_forcing_random_step
                        and sf_exit_chunk > 0
                        and is_video_train_mode
                        and (not is_action_train_mode)
                        and (not self.args.return_action)
                    ):
                        run_chunk0_with_grad = False

                    chunk0_ctx = nullcontext() if run_chunk0_with_grad else torch.no_grad()
                    need_jepa_buffer_chunk0 = bool(
                        self.jepa_helper is not None
                        and sf_chunks <= 1
                        and is_video_train_mode
                        and bool(getattr(self.jepa_helper, "uses_rollout_buffer", lambda: False)())
                        and float(self._get_jepa_cfg().get("train", {}).get("loss_weight", 0.0)) > 0.0
                    )
                    with chunk0_ctx:
                        pred_all = forward_pass(
                            model=self.diffusion_model, 
                            timesteps=timesteps, 
                            noisy_latents=noisy_latents,
                            prompt_embeds=prompt_embeds, 
                            prompt_attention_mask=prompt_attention_mask,
                            num_frames=latent_frames,
                            height=latent_height,
                            width=latent_width,
                            n_view=n_view,
                            action_states=noisy_actions,
                            action_timestep=action_timesteps,
                            return_video=self.args.return_video or self.args.return_action,
                            return_action=self.args.return_action,
                            video_attention_mask=video_attention_mask,
                            history_action_state=act_state,
                            condition_mask=conditioning_mask,
                            store_buffer=need_jepa_buffer_chunk0,
                            store_buffer_indices=self.jepa_helper.get_store_buffer_indices() if need_jepa_buffer_chunk0 else None,
                        )['latents']

                    if is_video_train_mode:
                        pred = pred_all['video']
                        target = noise - latents
                        loss_video_raw = weights.float() * (pred.float() - target.float()).pow(2)
                        loss_video_raw = loss_video_raw * (
                            1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss_video_raw.size(-1))
                        )
                        # Average loss across channel dimension
                        loss_video_raw = loss_video_raw.mean(list(range(1, loss_video_raw.ndim)))
                        # Average loss across batch dimension
                        loss_video_raw = loss_video_raw.mean()
                        if run_chunk0_with_grad:
                            loss_video = loss_video_raw
                        else:
                            loss_video = torch.zeros((), device=latents.device, dtype=latents.dtype)
                    else:
                        loss_video = 0.

                    if is_action_train_mode:
                        target_action = noise_actions - actions
                        loss_action = action_weights.float() * (pred_all['action'].float() - target_action.float()).pow(2)    # shape b,l,c
                        loss_action = loss_action.mean()
                    else:
                        loss_action = 0.

                    loss_video_holistic = torch.zeros((), device=latents.device, dtype=latents.dtype)
                    loss_video_jepa = torch.zeros((), device=latents.device, dtype=latents.dtype)
                    holistic_logs = {}

                    if sf_chunks > 1 and is_video_train_mode:
                        rollout_losses = []
                        rollout_holistic_losses = []
                        rollout_jepa_losses = []
                        rollout_holistic_logs = []
                        if run_chunk0_with_grad:
                            rollout_losses.append(loss_video)

                        pred_latents = noisy_latents_core - ss * pred
                        pred_latents = unpack_latents(
                            pred_latents, num_frames=latent_frames, height=latent_height, width=latent_width
                        )
                        prev_mem_latents = pred_latents[:, :, -mem_size:, :, :]
                        if self_forcing_detach_mem:
                            prev_mem_latents = prev_mem_latents.detach()

                        for rollout_idx in range(1, sf_chunks):
                            if self_forcing_random_step and rollout_idx > sf_exit_chunk:
                                break

                            frame_start = rollout_idx * chunk_size
                            frame_end = frame_start + mem_size + chunk_size
                            if frame_end > video_full.shape[2]:
                                break

                            video_chunk = video_full[:, :, frame_start:frame_end]
                            mem_chunk = video_chunk[:, :, :mem_size]
                            future_chunk = video_chunk[:, :, mem_size:]

                            _, _, raw_frames_sf, raw_height_sf, raw_width_sf = future_chunk.shape
                            latent_frames_sf = raw_frames_sf // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
                            latent_height_sf = raw_height_sf // self.SPATIAL_DOWN_RATIO
                            latent_width_sf = raw_width_sf // self.SPATIAL_DOWN_RATIO

                            mem_latents_gt_sf, future_latents_sf = get_latents(
                                self.vae, mem_chunk, future_chunk
                            )
                            mem_latents_gt_sf = rearrange(
                                mem_latents_gt_sf,
                                "(b v m) (h w) c -> (b v) c m h w",
                                b=batch_size,
                                m=mem_size,
                                h=latent_height_sf,
                            )
                            future_latents_sf = rearrange(
                                future_latents_sf,
                                "(b v) (f h w) c -> (b v) c f h w",
                                b=batch_size,
                                h=latent_height_sf,
                                w=latent_width_sf,
                            )

                            mem_latents_sf = prev_mem_latents
                            if mem_latents_sf.shape[2] != mem_size:
                                mem_latents_sf = mem_latents_gt_sf
                            latents_sf = torch.cat((mem_latents_sf, future_latents_sf), dim=2)
                            latents_sf = rearrange(latents_sf, "bv c f h w -> bv (f h w) c")

                            cond_to_concat_flat_sf = None
                            if use_gesim_cond and cond_to_concat_full is not None:
                                cond_chunk = cond_to_concat_full[:, :, frame_start:frame_end]
                                n_fut_sf = latent_frames_sf - mem_size
                                cond_to_concat_resized_sf = resize_traj_and_ray(
                                    cond_chunk,
                                    mem_size=mem_size,
                                    future_size=n_fut_sf,
                                    height=latent_height_sf,
                                    width=latent_width_sf,
                                )
                                cond_to_concat_flat_sf = rearrange(
                                    cond_to_concat_resized_sf, "bv c t h w -> bv (t h w) c"
                                )

                            sample_weights_sf = compute_density_for_timestep_sampling(
                                weighting_scheme=self.args.flow_weighting_scheme,
                                batch_size=batch_size,
                                logit_mean=self.args.flow_logit_mean,
                                logit_std=self.args.flow_logit_std,
                                mode_scale=self.args.flow_mode_scale,
                            )
                            sample_weights_sf = sample_weights_sf.unsqueeze(1).repeat(1, n_view)
                            sample_weights_sf = rearrange(sample_weights_sf, "b v -> (b v)")
                            indices_sf = (sample_weights_sf * self.scheduler.config.num_train_timesteps).long()
                            sigmas_sf = scheduler_sigmas[indices_sf]
                            timesteps_sf = (sigmas_sf * 1000.0).long()

                            noise_sf, conditioning_mask_sf, cond_indicator_sf = gen_noise_from_condition_frame_latent(
                                mem_latents_sf,
                                latent_frames_sf,
                                latent_height_sf,
                                latent_width_sf,
                                noise_to_condition_frames=self.args.noise_to_first_frame,
                            )
                            if self.args.pixel_wise_timestep:
                                timesteps_sf = timesteps_sf.unsqueeze(-1) * (1 - conditioning_mask_sf)
                            else:
                                timesteps_sf = timesteps_sf.unsqueeze(-1) * (1 - cond_indicator_sf)

                            ss_sf = sigmas_sf.reshape(-1, 1, 1).repeat(1, 1, latents_sf.size(-1))
                            noisy_latents_core_sf = (1.0 - ss_sf) * latents_sf + ss_sf * noise_sf
                            noisy_latents_sf = noisy_latents_core_sf
                            if cond_to_concat_flat_sf is not None:
                                noisy_latents_sf = torch.cat([noisy_latents_sf, cond_to_concat_flat_sf], dim=-1)

                            loss_weights_sf = compute_loss_weighting_for_sd3(
                                weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas_sf
                            ).reshape(-1, 1, 1).repeat(1, 1, latents_sf.size(-1))

                            run_grad_this_chunk = (not self_forcing_random_step) or (rollout_idx == sf_exit_chunk)
                            rollout_ctx = nullcontext() if run_grad_this_chunk else torch.no_grad()
                            need_jepa_buffer_sf = bool(
                                run_grad_this_chunk
                                and self.jepa_helper is not None
                                and bool(getattr(self.jepa_helper, "uses_rollout_buffer", lambda: False)())
                                and float(self._get_jepa_cfg().get("train", {}).get("loss_weight", 0.0)) > 0.0
                            )
                            with rollout_ctx:
                                pred_all_sf = forward_pass(
                                    model=self.diffusion_model,
                                    timesteps=timesteps_sf,
                                    noisy_latents=noisy_latents_sf,
                                    prompt_embeds=prompt_embeds,
                                    prompt_attention_mask=prompt_attention_mask,
                                    num_frames=latent_frames_sf,
                                    height=latent_height_sf,
                                    width=latent_width_sf,
                                    n_view=n_view,
                                    action_states=None,
                                    action_timestep=None,
                                    return_video=True,
                                    return_action=False,
                                    video_attention_mask=None,
                                    history_action_state=None,
                                    condition_mask=conditioning_mask_sf,
                                    store_buffer=need_jepa_buffer_sf,
                                    store_buffer_indices=self.jepa_helper.get_store_buffer_indices() if need_jepa_buffer_sf else None,
                                )["latents"]
                            pred_sf = pred_all_sf["video"]
                            target_sf = noise_sf - latents_sf
                            loss_video_sf = loss_weights_sf.float() * (pred_sf.float() - target_sf.float()).pow(2)
                            loss_video_sf = loss_video_sf * (
                                1 - conditioning_mask_sf.unsqueeze(-1).repeat(1, 1, loss_video_sf.size(-1))
                            )
                            loss_video_sf = loss_video_sf.mean(list(range(1, loss_video_sf.ndim))).mean()
                            if run_grad_this_chunk:
                                rollout_losses.append(loss_video_sf)

                            pred_latents_sf = noisy_latents_core_sf - ss_sf * pred_sf
                            pred_latents_sf = unpack_latents(
                                pred_latents_sf,
                                num_frames=latent_frames_sf,
                                height=latent_height_sf,
                                width=latent_width_sf,
                            )
                            prev_mem_latents = pred_latents_sf[:, :, -mem_size:, :, :]
                            if self_forcing_detach_mem:
                                prev_mem_latents = prev_mem_latents.detach()

                            if run_grad_this_chunk:
                                pred_future_latents_sf = pred_latents_sf[:, :, mem_size:, :, :]
                                loss_video_holistic_sf, holistic_logs_sf = self._compute_rollout_holistic_loss(
                                    pred_future_latents=pred_future_latents_sf,
                                    gt_future_video=future_chunk,
                                )
                                rollout_holistic_losses.append(loss_video_holistic_sf)
                                rollout_holistic_logs.append(holistic_logs_sf)
                                jepa_train_cfg = self._get_jepa_cfg().get("train", {})
                                jepa_train_weight = float(jepa_train_cfg.get("loss_weight", 0.0)) if isinstance(jepa_train_cfg, dict) else 0.0
                                if self.jepa_helper is not None and jepa_train_weight > 0.0 and jepa_actions_full is not None:
                                    cond_chunk_for_jepa = None
                                    if use_gesim_cond and cond_to_concat_full is not None:
                                        cond_chunk_for_jepa = cond_to_concat_full[:, :, frame_start:frame_end]
                                    frame_repr_sf = None
                                    if need_jepa_buffer_sf and pred_all_sf.get("video_states_buffer", None) is not None:
                                        frame_repr_sf = self.jepa_helper.extract_frame_repr_from_buffer(
                                            video_states_buffer=pred_all_sf["video_states_buffer"],
                                            num_frames=latent_frames_sf,
                                            height=latent_height_sf,
                                            width=latent_width_sf,
                                        )
                                    loss_video_jepa_sf = self._compute_jepa_rollout_loss(
                                        video=None,
                                        cond_to_concat=cond_chunk_for_jepa,
                                        prompt_embeds=prompt_embeds,
                                        prompt_attention_mask=prompt_attention_mask,
                                        actions=jepa_actions_full[:, frame_start:frame_end],
                                        mem_size=mem_size,
                                        n_view=n_view,
                                        frame_repr=frame_repr_sf,
                                        latent_video=pred_latents_sf,
                                    ) * jepa_train_weight
                                    rollout_jepa_losses.append(loss_video_jepa_sf)

                            if self_forcing_random_step and rollout_idx >= sf_exit_chunk:
                                break

                        if len(rollout_losses) == 0:
                            loss_video = torch.zeros((), device=latents.device, dtype=latents.dtype)
                        elif self_forcing_random_step:
                            loss_video = rollout_losses[-1]
                        else:
                            loss_video = torch.stack(rollout_losses).mean()
                        if len(rollout_holistic_losses) > 0:
                            if self_forcing_random_step:
                                loss_video_holistic = rollout_holistic_losses[-1]
                                holistic_logs = rollout_holistic_logs[-1]
                            else:
                                loss_video_holistic = torch.stack(rollout_holistic_losses).mean()
                                merged_holistic_logs = {}
                                valid_log_count = 0
                                for item in rollout_holistic_logs:
                                    if not item:
                                        continue
                                    valid_log_count += 1
                                    for k, v in item.items():
                                        merged_holistic_logs[k] = merged_holistic_logs.get(k, 0.0) + float(v)
                                if valid_log_count > 0:
                                    holistic_logs = {k: v / valid_log_count for k, v in merged_holistic_logs.items()}
                        if len(rollout_jepa_losses) > 0:
                            loss_video_jepa = rollout_jepa_losses[-1] if self_forcing_random_step else torch.stack(rollout_jepa_losses).mean()

                    action_loss_scale = getattr(self.args, "action_loss_scale", 1.0)

                    loss = loss_video + loss_video_holistic + loss_video_jepa + action_loss_scale * loss_action

                    loss_is_finite = accelerator.reduce(
                        torch.isfinite(loss.detach()).to(dtype=loss.dtype),
                        reduction="min",
                    )
                    if float(loss_is_finite.item()) < 1.0:
                        raise FloatingPointError(f"Non-finite loss detected at global_step={global_step}")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                

                loss = accelerator.reduce(loss.detach(), reduction='mean')
                if is_action_train_mode:
                    loss_action = accelerator.reduce(loss_action.detach(), reduction='mean')
                if is_video_train_mode:
                    loss_video = accelerator.reduce(loss_video.detach(), reduction='mean')
                loss_video_holistic = accelerator.reduce(loss_video_holistic.detach(), reduction='mean')
                loss_video_jepa = accelerator.reduce(loss_video_jepa.detach(), reduction='mean')

                running_loss += loss.item()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    try:
                        debug_every = int(getattr(self.args, "debug_train_decode_every", 0))
                    except Exception:
                        debug_every = 0
                    if (
                        accelerator.is_main_process
                        and bool(getattr(self.args, "debug_train_decode", False))
                        and debug_every > 0
                        and global_step % debug_every == 0
                    ):
                        self._dump_train_batch_decode(batch=batch, global_step=global_step, epoch=epoch, step=step)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                if "sf_chunks" in locals():
                    logs["sf_chunks"] = int(sf_chunks)
                if "sf_exit_chunk" in locals() and int(sf_chunks) > 1:
                    logs["sf_exit_chunk"] = int(sf_exit_chunk)
                if "train_first_frame_only" in locals():
                    logs["train_fff"] = int(bool(train_first_frame_only))
                if float(loss_video_holistic.detach().item()) > 0.0:
                    logs["holistic"] = float(loss_video_holistic.detach().item())
                if float(loss_video_jepa.detach().item()) > 0.0:
                    logs["jepa_phys"] = float(loss_video_jepa.detach().item())
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if accelerator.sync_gradients:
                    if global_step % self.args.steps_to_log == 0:
                        if accelerator.is_main_process:
                            if self.writer is not None:
                                self.writer.add_scalar("Training Loss", loss.item(), global_step)
                                if is_action_train_mode:
                                    self.writer.add_scalar("Action loss", loss_action.mean().item(), global_step)
                                if is_video_train_mode:
                                    self.writer.add_scalar("Video loss", loss_video.item(), global_step)
                                self.writer.add_scalar("Holistic loss", loss_video_holistic.item(), global_step)
                                self.writer.add_scalar("JEPA physics loss", loss_video_jepa.item(), global_step)
                                if "train_first_frame_only" in locals():
                                    self.writer.add_scalar("Train first-frame-only", float(bool(train_first_frame_only)), global_step)

                    if global_step % self.args.steps_to_val == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            model_save_dir = os.path.join(self.save_folder, f"Validation_step_{global_step}")
                            self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)
                        accelerator.wait_for_everyone()

                    if global_step % self.args.steps_to_save == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            model_to_save = unwrap_model(accelerator, self.diffusion_model)
                            dtype = (
                                torch.float16
                                if self.args.mixed_precision == "fp16"
                                else torch.bfloat16
                                if self.args.mixed_precision == "bf16"
                                else torch.float32
                            )

                            model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
                            model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                            del model_to_save

                    if global_step >= self.state.train_steps:
                        logger.info(">>> max train step reached")
                        break
                        
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else torch.bfloat16
                if self.args.mixed_precision == "bf16"
                else torch.float32
            )

            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

        del self.diffusion_model, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()


    def validate(self, accelerator, model_save_dir, global_step, n_view=1, n_chunk=30, image=None, prompt=None, cap=None, path=None, gt_actions=None, to_log=True):

        os.makedirs(model_save_dir,exist_ok=True)
        jepa_old_transformer = None
        if self.jepa_backend_type == "legacy" and self.jepa_helper is not None:
            jepa_old_transformer = self.jepa_helper.transformer
            self.jepa_helper.transformer = unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model

        jepa_infer_cfg = self._get_jepa_cfg().get("infer", {})
        requested_jepa_guidance = isinstance(jepa_infer_cfg, dict) and float(jepa_infer_cfg.get("guidance_strength", 0.0)) > 0.0
        if self.jepa_backend_type == "official_vjepa2_ac" and requested_jepa_guidance:
            raise RuntimeError(
                "Official V-JEPA2-AC backend does not support latent guidance in validate(). "
                "Set jepa.infer.guidance_strength=0.0."
            )
        use_jepa_guided_pipe = (
            self.jepa_backend_type == "legacy"
            and self.jepa_helper is not None
            and requested_jepa_guidance
        )
        pipe_cls = JEPAGuidedPipeline if use_jepa_guided_pipe else self.pipeline_class
        pipe = pipe_cls(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model,
        )
        self._attach_jepa_to_pipe(pipe)

        sim_type = str(getattr(self.args, "sim_type", "default")).lower()
        val_cfg = getattr(self.args, "val", {})
        full_video_val = isinstance(val_cfg, dict) and bool(val_cfg.get("full_video", False))
        val_first_frame_only = isinstance(val_cfg, dict) and bool(val_cfg.get("first_frame_only", False))
        n_previous = int(self.args.data['train']['n_previous'])
        if sim_type in {"gesim", "ge-sim", "sim"} and full_video_val:
            ok = self._run_full_video_validation(pipe, model_save_dir, global_step)
            if ok:
                if self.jepa_backend_type == "legacy" and self.jepa_helper is not None:
                    self.jepa_helper.transformer = jepa_old_transformer
                return
            if self.val_dataloader is None:
                logger.warning("[Validation] full-video validation failed and val_dataloader is unavailable; skip this validation step.")
                if self.jepa_backend_type == "legacy" and self.jepa_helper is not None:
                    self.jepa_helper.transformer = jepa_old_transformer
                return

        batch = next(iter(self.val_dataloader))
        if val_first_frame_only:
            image = batch['video'][:, :, :, :1].clone().repeat(1, 1, 1, n_previous, 1, 1)
        else:
            image = batch['video'][:, :, :, :n_previous].clone()  # shape b,c,v,t,h,w
        gt_video = batch['video']
        b, c, v, t, h, w = image.shape
        negative_prompt = ''

        batch_size = 1
        prompt = batch.get('caption', None)
        if prompt is None:
            prompt = [self._default_prompt("val")] * batch_size
        elif isinstance(prompt, str):
            prompt = [prompt] * batch_size
        else:
            prompt = list(prompt)
            if len(prompt) < batch_size:
                fill = prompt[-1] if len(prompt) > 0 else self._default_prompt("val")
                prompt = prompt + [fill] * (batch_size - len(prompt))
            elif len(prompt) > batch_size:
                prompt = prompt[:batch_size]

        image = image[:batch_size]

        image = rearrange(image, 'b c v t h w -> (b v) c t h w')
        num_denois_steps = self.args.num_inference_step

        if self.args.return_action and getattr(self.args, "add_state", False):
            history_action_state = batch['state'][:batch_size]
            if history_action_state.shape[1] > 1:
                history_action_state = history_action_state[:, n_previous-1:n_previous, :]
            history_action_state = history_action_state.contiguous()
        else:
            history_action_state = None

        if sim_type in {"gesim", "ge-sim", "sim"}:
            cond_to_concat = batch["cond_to_concat"][:batch_size].to(
                image.device, dtype=image.dtype
            ).contiguous()
            if val_first_frame_only:
                cond_t = int(cond_to_concat.shape[3])
                future_start = 1
                future_end = min(future_start + int(self.args.data["train"]["chunk"]), cond_t)
                cond_indices = [0] * n_previous + list(range(future_start, future_end))
                if len(cond_indices) == 0:
                    cond_indices = [0]
                expected_t = n_previous + int(self.args.data["train"]["chunk"])
                if len(cond_indices) < expected_t:
                    cond_indices = cond_indices + [cond_indices[-1]] * (expected_t - len(cond_indices))
                else:
                    cond_indices = cond_indices[:expected_t]
                cond_idx_t = torch.tensor(cond_indices, device=cond_to_concat.device, dtype=torch.long)
                cond_to_concat = cond_to_concat.index_select(3, cond_idx_t)
            image_for_pipe = rearrange(image, "(b v) c t h w -> (b v) t c h w", b=batch_size, v=v)
            cond_for_pipe = rearrange(cond_to_concat, "b c v t h w -> (b v) c t h w")
            preds_out = pipe.infer(
                video=image_for_pipe,
                cond_to_concat=cond_for_pipe,
                prompt=prompt[:batch_size],
                negative_prompt=negative_prompt,
                num_inference_steps=num_denois_steps,
                guidance_scale=1.0,
                height=h,
                width=w,
                n_view=v,
                n_prev=n_previous,
                num_frames=self.args.data["train"]["chunk"],
                merge_view_into_width=False,
                output_type="pt",
                postprocess_video=False,
                show_progress=False,
            )
            pred_video = preds_out["frames"] if isinstance(preds_out, dict) else preds_out.frames
            preds = {"video": pred_video}
        else:
            preds = pipe.infer(
                image=image,
                prompt=prompt[:batch_size],
                negative_prompt=negative_prompt,
                num_inference_steps=num_denois_steps,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                guidance_scale=1.0,
                height=h,
                width=w,
                n_view=v,
                return_action=self.args.return_action,
                n_prev=n_previous,
                chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
                return_video=self.args.return_video,
                noise_seed=42,
                action_chunk=self.args.data['train']['action_chunk'],
                history_action_state = history_action_state,
                pixel_wise_timestep = self.args.pixel_wise_timestep,
                n_chunk=n_chunk,
                action_dim=self.args.diffusion_model["config"]["action_in_channels"] if self.args.return_action else None,
            )[0]

        cap = 'Validation'
        fps = int(getattr(self.args, "basic_fps", 30) / (self.args.data['train']['action_chunk'] // self.args.data['train']['chunk']))
        save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{cap}_gt.mp4'), fps=fps)

        if self.args.return_video:
            video = preds['video'].data.cpu()
            finite_ratio = torch.isfinite(video).float().mean().item()
            nan_count = int(torch.isnan(video).sum().item())
            inf_count = int(torch.isinf(video).sum().item())
            vmin = float(video.amin().item()) if video.numel() > 0 else float("nan")
            vmax = float(video.amax().item()) if video.numel() > 0 else float("nan")
            logger.info(
                f"[Validation] pred_video stats: shape={tuple(video.shape)}, "
                f"finite_ratio={finite_ratio:.6f}, nan={nan_count}, inf={inf_count}, min={vmin:.6f}, max={vmax:.6f}"
            )
            save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], os.path.join(model_save_dir, f'{cap}.mp4'), fps=fps)

        if to_log:
            self.writer.add_text(f'step_{global_step}/{cap} prompt:', prompt[0], global_step)

        if self.args.return_action:
            # shape t, c
            gt_actions = batch['actions'][:, -self.args.data['train']['action_chunk']:]
            action_dim = gt_actions.shape[-1]

            action_logs = act_metric(
                preds['action'][:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                gt_actions[:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                prefix=cap,
                start_stop_interval=[(0,1),(1,9),(9,25),(25,self.args.data['train']['action_chunk'])]
            )

            if to_log:
                for key, value in action_logs.items():
                    self.writer.add_scalar(key, value, global_step)
        if self.jepa_backend_type == "legacy" and self.jepa_helper is not None:
            self.jepa_helper.transformer = jepa_old_transformer

    @torch.no_grad()
    def _run_full_video_validation(self, pipe, model_save_dir, global_step):
        try:
            from scripts.infer_iros2026_gesim import scan_samples, infer_sample
            import h5py
        except Exception as e:
            logger.warning(f"[Validation-Full] import helper failed: {repr(e)}")
            return False

        val_cfg = getattr(self.args, "val", {})
        data_val_cfg = self.args.data.get("val", {}) if isinstance(self.args.data, dict) else {}

        data_roots = data_val_cfg.get("data_roots", [])
        if isinstance(data_roots, str):
            data_roots = [data_roots]
        if len(data_roots) == 0:
            logger.warning("[Validation-Full] no data_roots configured for data.val.")
            return False

        data_root = data_roots[0]
        split = str(data_val_cfg.get("split", "validation"))
        if split == "val":
            split = "validation"

        info_dir = os.path.join(data_root, split, "info_dataset")
        if not os.path.isdir(info_dir):
            logger.warning(
                f"[Validation-Full] info_dataset not found for split='{split}': {info_dir}"
            )
            return False

        samples = scan_samples(data_root, split)
        if len(samples) == 0:
            logger.warning(f"[Validation-Full] no samples found in {info_dir}")
            return False

        try:
            n_samples = int(val_cfg.get("n_samples", 8)) if isinstance(val_cfg, dict) else 8
        except Exception:
            n_samples = 8
        n_samples = max(1, n_samples)
        n_samples = min(n_samples, len(samples))

        try:
            seed = int(val_cfg.get("seed", 2026)) if isinstance(val_cfg, dict) else 2026
        except Exception:
            seed = 2026

        # Persist a fixed validation subset once, then reuse it for all future steps.
        # This guarantees step-to-step comparability.
        fixed_samples_path = os.path.join(self.save_folder, "val_fixed_samples.json")
        selected_fixed = None
        if os.path.isfile(fixed_samples_path):
            try:
                with open(fixed_samples_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list) and len(loaded) > 0:
                    selected_fixed = []
                    for item in loaded:
                        if not isinstance(item, dict):
                            continue
                        task_id = str(item.get("task_id", ""))
                        episode_id = str(item.get("episode_id", ""))
                        sample_dir = str(item.get("dir", ""))
                        if not sample_dir and task_id and episode_id:
                            sample_dir = os.path.join(data_root, split, "info_dataset", task_id, episode_id)
                        if task_id and episode_id and sample_dir:
                            selected_fixed.append(
                                {"task_id": task_id, "episode_id": episode_id, "dir": sample_dir}
                            )
                if selected_fixed:
                    logger.info(
                        f"[Validation-Full] loaded fixed sample list: {fixed_samples_path} "
                        f"(n={len(selected_fixed)})"
                    )
            except Exception as e:
                logger.warning(
                    f"[Validation-Full] failed to load fixed sample list {fixed_samples_path}: {repr(e)}"
                )
                selected_fixed = None

        if not selected_fixed:
            rng = random.Random(seed)
            picked = rng.sample(samples, n_samples)
            selected_fixed = [
                {"task_id": str(x["task_id"]), "episode_id": str(x["episode_id"]), "dir": str(x["dir"])}
                for x in picked
            ]
            try:
                with open(fixed_samples_path, "w") as f:
                    json.dump(selected_fixed, f, indent=2)
                logger.info(
                    f"[Validation-Full] wrote fixed sample list: {fixed_samples_path} "
                    f"(n={len(selected_fixed)}, seed={seed})"
                )
            except Exception as e:
                logger.warning(
                    f"[Validation-Full] failed to write fixed sample list {fixed_samples_path}: {repr(e)}"
                )

        sample_size = tuple(self.args.data["train"]["sample_size"])
        n_previous = int(self.args.data["train"]["n_previous"])
        chunk = int(self.args.data["train"]["chunk"])
        first_frame_only = True
        if isinstance(val_cfg, dict):
            first_frame_only = bool(val_cfg.get("first_frame_only", True))
        prompt = self._default_prompt("val")
        negative_prompt = ""
        jepa_cfg = self._get_jepa_cfg()
        jepa_prompt_embeds = None
        jepa_prompt_attention_mask = None
        if self.jepa_helper is not None:
            try:
                text_cond = get_text_conditions(self.tokenizer, self.text_encoder, prompt=prompt)
                jepa_prompt_embeds = text_cond["prompt_embeds"].to(self.state.accelerator.device, dtype=self.state.weight_dtype)
                jepa_prompt_attention_mask = text_cond["prompt_attention_mask"].to(self.state.accelerator.device)
            except Exception as e:
                logger.warning(f"[Validation-Full] failed to build JEPA text condition: {repr(e)}")

        video_subdir = "validation_videos"
        if isinstance(val_cfg, dict):
            video_subdir = str(val_cfg.get("video_subdir", video_subdir))
        output_dir = os.path.join(model_save_dir, video_subdir)
        os.makedirs(output_dir, exist_ok=True)

        fps = int(
            getattr(self.args, "basic_fps", 30)
            / (self.args.data["train"]["action_chunk"] // self.args.data["train"]["chunk"])
        )
        device = str(self.state.accelerator.device)
        logger.info(
            f"[Validation-Full] split={split}, n_samples={n_samples}, chunk={chunk}, "
            f"sample_size={sample_size}, output={output_dir}"
        )

        generated = 0
        attempted = 0
        metric_rows = []
        psnr_sum = 0.0
        psnr_frames = 0
        late_half_sum = 0.0
        late_half_count = 0
        last_quarter_sum = 0.0
        last_quarter_count = 0
        motion_sum = 0.0
        motion_count = 0
        jepa_dyn_sum = 0.0
        jepa_dyn_count = 0
        jepa_repr_sum = 0.0
        jepa_repr_count = 0
        jepa_last_quarter_sum = 0.0
        jepa_last_quarter_count = 0
        for sample_info in selected_fixed:
            attempted += 1

            sample_dir = sample_info.get("dir", "")
            task_id = sample_info.get("task_id", "unknown_task")
            episode_id = sample_info.get("episode_id", "unknown_episode")
            if not sample_dir:
                logger.warning(f"[Validation-Full] skip malformed sample: {sample_info}")
                continue
            h5_path = os.path.join(sample_dir, "proprio_stats.h5")

            try:
                with h5py.File(h5_path, "r") as f:
                    total_timesteps = int(f["state/end/position"].shape[0])
                if total_timesteps <= 1:
                    logger.warning(f"[Validation-Full] skip {task_id}/{episode_id}: invalid timesteps={total_timesteps}")
                    continue

                num_frames_to_generate = total_timesteps - 1
                frames = infer_sample(
                    pipe=pipe,
                    args=self.args,
                    sample_dir=sample_dir,
                    sample_size=sample_size,
                    n_previous=n_previous,
                    chunk=chunk,
                    device=device,
                    num_frames_to_generate=num_frames_to_generate,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    first_frame_only=first_frame_only,
                    jepa_cfg=jepa_cfg,
                    jepa_helper=self.jepa_helper,
                    jepa_prompt_embeds=jepa_prompt_embeds,
                    jepa_prompt_attention_mask=jepa_prompt_attention_mask,
                )
                if len(frames) == 0:
                    logger.warning(f"[Validation-Full] skip {task_id}/{episode_id}: no generated frames")
                    continue

                video_np = np.stack(frames, axis=0)  # (t, h, w, c), uint8
                video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).to(torch.float32)
                video_tensor = video_tensor / 255.0 * 2.0 - 1.0

                save_name = f"{task_id}_{episode_id}.mp4"
                save_path = os.path.join(output_dir, save_name)
                save_video(video_tensor, save_path, fps=fps)
                logger.info(
                    f"[Validation-Full] saved {save_name} frames={video_np.shape[0]} size={video_np.shape[1]}x{video_np.shape[2]}"
                )
                gt_dir = os.path.join(data_root, split, "gt_dataset", task_id, episode_id, "video")
                sample_metrics = self._compute_video_metrics(video_np, gt_dir)
                if sample_metrics:
                    if self.jepa_helper is not None and jepa_prompt_embeds is not None:
                        try:
                            sample_metrics.update(
                                self._compute_jepa_validation_metrics(
                                    sample_dir=sample_dir,
                                    pred_frames=video_np,
                                    gt_dir=gt_dir,
                                    sample_size=sample_size,
                                    mem_size=n_previous,
                                    first_frame_only=first_frame_only,
                                    prompt_embeds=jepa_prompt_embeds,
                                    prompt_attention_mask=jepa_prompt_attention_mask,
                                )
                            )
                        except Exception as e:
                            logger.warning(f"[Validation-Full] JEPA metric failed for {task_id}/{episode_id}: {repr(e)}")
                    metric_rows.append(
                        {
                            "task_id": task_id,
                            "episode_id": episode_id,
                            **sample_metrics,
                        }
                    )
                    if sample_metrics.get("avg_psnr") is not None:
                        psnr_sum += float(sample_metrics["avg_psnr"]) * int(sample_metrics["frames"])
                        psnr_frames += int(sample_metrics["frames"])
                    if sample_metrics.get("late_half_psnr") is not None:
                        late_frames = max(1, int(sample_metrics["frames"]) - int(sample_metrics["frames"]) // 2)
                        late_half_sum += float(sample_metrics["late_half_psnr"]) * late_frames
                        late_half_count += late_frames
                    if sample_metrics.get("last_quarter_psnr") is not None:
                        last_frames = max(1, int(sample_metrics["frames"]) - (3 * int(sample_metrics["frames"])) // 4)
                        last_quarter_sum += float(sample_metrics["last_quarter_psnr"]) * last_frames
                        last_quarter_count += last_frames
                    if sample_metrics.get("motion_l1") is not None:
                        motion_sum += float(sample_metrics["motion_l1"])
                        motion_count += 1
                    if sample_metrics.get("jepa_dyn_loss") is not None:
                        jepa_dyn_sum += float(sample_metrics["jepa_dyn_loss"])
                        jepa_dyn_count += 1
                    if sample_metrics.get("jepa_repr_l2") is not None:
                        jepa_repr_sum += float(sample_metrics["jepa_repr_l2"]) * int(sample_metrics["frames"])
                        jepa_repr_count += int(sample_metrics["frames"])
                    if sample_metrics.get("jepa_last_quarter_repr_l2") is not None:
                        last_frames = max(1, int(sample_metrics["frames"]) - (3 * int(sample_metrics["frames"])) // 4)
                        jepa_last_quarter_sum += float(sample_metrics["jepa_last_quarter_repr_l2"]) * last_frames
                        jepa_last_quarter_count += last_frames
                    logger.info(
                        f"[Validation-Full] metrics {task_id}/{episode_id}: "
                        f"avg_psnr={sample_metrics.get('avg_psnr')}, "
                        f"late_half_psnr={sample_metrics.get('late_half_psnr')}, "
                        f"last_quarter_psnr={sample_metrics.get('last_quarter_psnr')}, "
                        f"motion_l1={sample_metrics.get('motion_l1')}, "
                        f"jepa_dyn_loss={sample_metrics.get('jepa_dyn_loss')}, "
                        f"jepa_repr_l2={sample_metrics.get('jepa_repr_l2')}, "
                        f"jepa_last_quarter_repr_l2={sample_metrics.get('jepa_last_quarter_repr_l2')}"
                    )
                generated += 1
            except Exception as e:
                logger.warning(
                    f"[Validation-Full] failed {task_id}/{episode_id}: {repr(e)}",
                    exc_info=True,
                )

        target_n = len(selected_fixed)
        if generated < target_n:
            logger.warning(
                f"[Validation-Full] insufficient valid samples: generated={generated}/{target_n}, "
                f"attempted={attempted}"
            )
        logger.info(
            f"[Validation-Full] generated {generated}/{target_n} videos (attempted={attempted})."
        )
        if len(metric_rows) > 0:
            summary = {
                "global_step": int(global_step),
                "videos": int(len(metric_rows)),
                "weighted_avg_psnr": (psnr_sum / psnr_frames) if psnr_frames > 0 else None,
                "weighted_late_half_psnr": (late_half_sum / late_half_count) if late_half_count > 0 else None,
                "weighted_last_quarter_psnr": (last_quarter_sum / last_quarter_count) if last_quarter_count > 0 else None,
                "avg_motion_l1": (motion_sum / motion_count) if motion_count > 0 else None,
                "avg_jepa_dyn_loss": (jepa_dyn_sum / jepa_dyn_count) if jepa_dyn_count > 0 else None,
                "weighted_jepa_repr_l2": (jepa_repr_sum / jepa_repr_count) if jepa_repr_count > 0 else None,
                "weighted_jepa_last_quarter_repr_l2": (jepa_last_quarter_sum / jepa_last_quarter_count) if jepa_last_quarter_count > 0 else None,
            }
            metrics_payload = {"summary": summary, "per_video": metric_rows}
            metrics_path = os.path.join(model_save_dir, "validation_metrics.json")
            try:
                with open(metrics_path, "w") as f:
                    json.dump(metrics_payload, f, indent=2)
            except Exception as e:
                logger.warning(f"[Validation-Full] failed to write metrics json {metrics_path}: {repr(e)}")

            logger.info(f"[Validation-Full] summary: {json.dumps(summary, indent=2)}")
            if self.writer is not None:
                if summary["weighted_avg_psnr"] is not None:
                    self.writer.add_scalar("Validation/weighted_avg_psnr", summary["weighted_avg_psnr"], global_step)
                if summary["weighted_late_half_psnr"] is not None:
                    self.writer.add_scalar("Validation/weighted_late_half_psnr", summary["weighted_late_half_psnr"], global_step)
                if summary["weighted_last_quarter_psnr"] is not None:
                    self.writer.add_scalar("Validation/weighted_last_quarter_psnr", summary["weighted_last_quarter_psnr"], global_step)
                if summary["avg_motion_l1"] is not None:
                    self.writer.add_scalar("Validation/avg_motion_l1", summary["avg_motion_l1"], global_step)
                if summary["avg_jepa_dyn_loss"] is not None:
                    self.writer.add_scalar("Validation/avg_jepa_dyn_loss", summary["avg_jepa_dyn_loss"], global_step)
                if summary["weighted_jepa_repr_l2"] is not None:
                    self.writer.add_scalar("Validation/weighted_jepa_repr_l2", summary["weighted_jepa_repr_l2"], global_step)
                if summary["weighted_jepa_last_quarter_repr_l2"] is not None:
                    self.writer.add_scalar("Validation/weighted_jepa_last_quarter_repr_l2", summary["weighted_jepa_last_quarter_repr_l2"], global_step)
        return generated > 0
