# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Dict, Any, TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from fastgen.methods import FastGenModel, CausVidModel
from fastgen.utils import expand_like
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.methods.config_kd import KDModelConfig as ModelConfig


class KDModel(FastGenModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the knowledge distillation model.
            This model directly learns the pre-constructed ODE pairs from the teacher model.

            Optional SWIFT target extrapolation (arXiv:2610.25844): when `config.swift_alpha > 0`, the regression
            target `x_T` becomes `x_T + alpha * (x_T - G_weak(x_t, t))` where `G_weak` is a frozen snapshot of the
            student taken at `config.swift_snapshot_iter` by an `EMACallback(type="snapshot")` into the network
            named `config.swift_weak_name`. Before the snapshot exists the plain KD target is used.
        """
        self.swift_alpha = float(getattr(config, "swift_alpha", 0.0) or 0.0)
        self.swift_snapshot_iter = int(getattr(config, "swift_snapshot_iter", 0) or 0)
        self.swift_weak_name = getattr(config, "swift_weak_name", "ema_swift")
        if self.swift_alpha > 0:
            assert self.swift_weak_name.startswith("ema"), "swift_weak_name must start with `ema`"
            # The weak snapshot is held like an EMA network (full copy per rank, not sharded, checkpointed).
            use_ema = config.use_ema
            if isinstance(use_ema, bool):
                use_ema = ["ema"] if use_ema else []
            if self.swift_weak_name not in use_ema:
                config.use_ema = list(use_ema) + [self.swift_weak_name]
        super().__init__(config)
        self.config = config

    def build_model(self):
        super().build_model()
        self.load_student_weights_and_ema()

    @property
    def net_inference(self):
        """Same as FastGenModel.net_inference, but never use the SWIFT weak snapshot for inference."""
        use_ema_for_inference = self.use_ema and not getattr(self, "_is_fsdp", False)
        if use_ema_for_inference:
            for name in self.use_ema:
                if name != self.swift_weak_name:
                    return getattr(self, name)
        return self.net

    def _total_iteration(self, iteration: int) -> int:
        resume_iter = getattr(self, "resume_iter", None)
        return iteration + resume_iter if isinstance(resume_iter, int) else iteration

    def swift_active(self, iteration: int) -> bool:
        """SWIFT is active once the weak snapshot has been taken (at the end of `swift_snapshot_iter`)."""
        return self.swift_alpha > 0 and self._total_iteration(iteration) > self.swift_snapshot_iter

    def swift_target(
        self,
        target: torch.Tensor,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        condition: Any,
        iteration: int,
    ) -> torch.Tensor:
        """Return the KD regression target, extrapolated away from the weak snapshot when SWIFT is active.

        target_swift = target + alpha * (target - G_weak(input_student, t_student))
        """
        if not self.swift_active(iteration):
            return target
        weak = getattr(self, self.swift_weak_name, None)
        if weak is None:
            if iteration == self.swift_snapshot_iter + 1:
                logger.warning(f"SWIFT weak network `{self.swift_weak_name}` not found; using the plain KD target.")
            return target
        with torch.no_grad():
            weak_pred = weak(input_student, t_student, condition=condition, fwd_pred_type="x0")
            return target + self.swift_alpha * (target - weak_pred.to(target.dtype))

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        if self.config.student_sample_steps == 1:
            assert input_student is not None, "input_student must be provided for KDModel"
            return {"gen_rand": gen_data, "input_rand": input_student}
        else:
            noise = torch.randn_like(gen_data, dtype=self.precision)
            gen_rand_func = partial(
                self.generator_fn,
                net=self.net_inference,
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                precision_amp=self.precision_amp_infer,
            )
            return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for knowledge distillation model.

        Important! For multistep KD distillation, t_list must be aligned with the `path`'s timesteps:
        1) Ensure t_list corresponds exactly to path timesteps
        2) Please check the `path_timesteps` item in index.json of the paired dataset
        3) num_inference_steps in denoise path must be 4
        4) student_sample_steps must be either 2 or 4
        5) Current approach assumes uniform spacing: t_list=[t1, t3] → path indices [0, 2]

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        denoised_data = data["real"]
        condition = data["condition"]
        batch_size = denoised_data.shape[0]

        if self.config.student_sample_steps == 1:
            # perform single-step distillation
            if "noise" in data:
                input_student = data["noise"]
            elif "path" in data:
                input_student = data["path"][:, 0, ...]  # the first step is noise
            else:
                raise ValueError("Noise or path must be provided for KDModel")
            t_student = torch.full(
                (batch_size,),
                self.net.noise_scheduler.max_t,
                device=self.device,
                dtype=self.net.noise_scheduler.t_precision,
            )
        else:
            # perform multiple-step distillation
            assert "path" in data, "path must be provided for KDModel"
            denoise_path = data["path"]  # [batch_size, num_inf_steps, C, num_frames, H, W]
            assert denoise_path.shape[1] == 4, "num_inference_steps in denoise path must be 4"
            assert (
                denoise_path.shape[1] % self.config.student_sample_steps == 0
            ), f"student_sample_steps must be either 2 or 4, but got {self.config.student_sample_steps}"

            t_student, t_list_ids = self.net.noise_scheduler.sample_from_t_list(
                batch_size,
                sample_steps=self.config.student_sample_steps,
                t_list=self.config.sample_t_cfg.t_list,
                return_ids=True,
                device=self.device,
            )

            # Important: Ensure t_list corresponds exactly to path timesteps
            # Current approach assumes uniform spacing: t_list=[t1, t3] → path indices [0, 2]
            path_indices = t_list_ids * (denoise_path.shape[1] // self.config.student_sample_steps)
            path_indices = expand_like(path_indices, denoise_path).expand(
                -1, -1, *denoise_path.shape[2:]
            )  # [batch_size, 1, C, num_frames, H, W]
            input_student = torch.gather(denoise_path, 1, path_indices).squeeze(1)  # [batch_size, C, num_frames, H, W]

        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)

        # Compute the l2 loss between the generated data and the (optionally SWIFT-extrapolated) denoised data
        target = self.swift_target(denoised_data, input_student, t_student, condition, iteration)
        loss = 0.5 * F.mse_loss(gen_data, target, reduction="mean")

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "recon_loss": loss,
        }
        if self.swift_alpha > 0:
            # plain (un-extrapolated) regression loss: the unbiased proxy for the student's error w.r.t. the teacher
            loss_map["recon_loss_plain"] = 0.5 * F.mse_loss(gen_data.detach(), denoised_data, reduction="mean")
            loss_map["swift_active"] = torch.tensor(float(self.swift_active(iteration)), device=loss.device)
        outputs = self._get_outputs(gen_data, input_student, condition=condition)

        return loss_map, outputs


class CausalKDModel(KDModel):
    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        noise = torch.randn_like(gen_data, dtype=self.precision)
        context_noise = getattr(self.config, "context_noise", 0)
        # Reuse CausVidModel's autoregressive generation logic
        gen_rand_func = partial(
            CausVidModel.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            context_noise=context_noise,
            precision_amp=self.precision_amp_infer,
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for knowledge distillation model.

         Important! t_list must be the same with the `path`'s timesteps.
         Please check the `path_timesteps` item in index.json of the paired dataset.

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        denoise_path = data["path"]  # shape is [batch_size, num_inf_steps, C, num_frames, H, W]
        denoised_data = data["real"]  # [batch_size, C, num_frames, H, W]
        condition = data["condition"]
        batch_size, num_frames = denoise_path.shape[0], denoise_path.shape[3]
        chunk_size = self.net.chunk_size

        # add noise
        t_inhom, ids = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            num_frames,
            chunk_size,
            sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,  # Note t_list to be aligned the `path`'s timesteps
            device=self.device,
            dtype=denoise_path.dtype,
        )  # [batch_size, num_frames]
        expand_shape = [ids.shape[0], 1, 1, ids.shape[1]] + [1] * max(0, denoise_path.ndim - 4)
        ids = ids.view(expand_shape).expand(-1, -1, *denoise_path.shape[2:])  # [batch_size, 1, C, num_frames, H, W]

        denoise_path_all = torch.cat([denoise_path, denoised_data.unsqueeze(1)], dim=1)  # gather clean data
        noisy_data = torch.gather(denoise_path_all, 1, ids).squeeze(1)  # [batch_size, C, num_frames, H, W]

        # generate data
        gen_data = self.gen_data_from_net(noisy_data, t_inhom, condition=condition)

        # Compute the l2 loss between the generated data and the (optionally SWIFT-extrapolated) denoised data
        target = self.swift_target(denoised_data, noisy_data, t_inhom, condition, iteration)
        loss = 0.5 * F.mse_loss(gen_data, target, reduction="mean")

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "recon_loss": loss,
        }
        if self.swift_alpha > 0:
            loss_map["recon_loss_plain"] = 0.5 * F.mse_loss(gen_data.detach(), denoised_data, reduction="mean")
            loss_map["swift_active"] = torch.tensor(float(self.swift_active(iteration)), device=loss.device)
        outputs = self._get_outputs(gen_data, condition=condition)

        return loss_map, outputs
