# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fastgen.configs.methods.config_kd_causal as config_kd_causal_default
from fastgen.callbacks.ema import EMACallback
from fastgen.configs.data import PathLoaderConfig
from fastgen.configs.net import CausalWan_1_3B_Config
from fastgen.utils import LazyCall as L

""" Configs for causal KD (ODE init for CausVid / Self-Forcing) on Wan-1.3B with SWIFT target extrapolation.

The regression target x_T (pre-computed teacher ODE output) is replaced by
    x_T + swift_alpha * (x_T - G_weak(x_t, t)),
where G_weak is a frozen snapshot of the student taken at `swift_snapshot_iter` (1/4 of the budget) by the
`ema_swift` snapshot callback. Set `swift_alpha = 0` for the plain causal-KD baseline with identical data/schedule.
Monitor `recon_loss_plain` (MSE to the un-extrapolated teacher output), not `recon_loss`, to compare runs.
"""


def create_config():
    config = config_kd_causal_default.create_config()

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000

    config.model.net_optimizer.lr = 7e-5

    config.model.input_shape = [16, 21, 60, 104]
    config.model.net = CausalWan_1_3B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.enable_preprocessors = False
    config.model.precision = "bfloat16"

    # KD with 4 steps (needs to be aligned with the steps in the data)
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # SWIFT: weak = student snapshot at 1/4 of the budget, extrapolation strength 0.25
    config.model.swift_alpha = 0.25
    config.model.swift_snapshot_iter = config.trainer.max_iter // 4
    config.model.swift_weak_name = "ema_swift"
    config.trainer.callbacks.update(
        dict(
            ema_swift=L(EMACallback)(
                type="snapshot",
                snapshot_iter=config.model.swift_snapshot_iter,
                ema_name=config.model.swift_weak_name,
            )
        )
    )

    config.dataloader_train = PathLoaderConfig
    config.dataloader_train.batch_size = 2

    config.log_config.group = "wan_kd_causal_swift"
    return config
