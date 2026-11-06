# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.config import (
    BaseConfig,
    BaseModelConfig,
)
from fastgen.methods.knowledge_distillation.KD import KDModel
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
)

""" Configs for the KD model, on CIFAR-10 dataset. """


@attrs.define(slots=False)
class KDModelConfig(BaseModelConfig):
    """Model config for KD, with optional SWIFT target extrapolation.

    SWIFT (Self-guided Weak-to-strong dIFfusion models via Target extrapolation, arXiv:2610.25844) replaces the
    regression target `x_T` (the pre-computed teacher output) by `x_T + alpha * (x_T - G_weak(x_t, t))`, where
    `G_weak` is a frozen early snapshot of the student itself. Requires an EMA callback of type "snapshot" that
    fills the network named `swift_weak_name` at `swift_snapshot_iter`; the extrapolation is only active after that.
    """

    # extrapolation strength; 0 disables SWIFT and recovers plain KD
    swift_alpha: float = 0.0
    # (total) iteration at which the weak snapshot is taken; must match the snapshot EMA callback
    swift_snapshot_iter: int = 0
    # name of the frozen weak network; must start with "ema" (handled like an EMA network: not sharded, checkpointed)
    swift_weak_name: str = "ema_swift"


@attrs.define(slots=False)
class Config(BaseConfig):
    model: KDModelConfig = attrs.field(factory=KDModelConfig)
    model_class: DictConfig = L(KDModel)(
        config=None,
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    config.dataloader_train.batch_size = 256
    config.model.net_scheduler.warm_up_steps = [0]

    return config
