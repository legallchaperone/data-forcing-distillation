# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.callbacks.ema import EMACallback
from fastgen.configs.experiments.EDM.config_kd_cifar10 import create_config as kd_cifar10_create_config
from fastgen.utils import LazyCall as L

""" Single-step KD on CIFAR-10 with SWIFT target extrapolation (same data, schedule and init as config_kd_cifar10).

weak = student snapshot at 1/4 of the budget; compare `recon_loss_plain` against the baseline run.
"""


def create_config():
    config = kd_cifar10_create_config()

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

    config.log_config.name = "kd_swift_a025"
    return config
