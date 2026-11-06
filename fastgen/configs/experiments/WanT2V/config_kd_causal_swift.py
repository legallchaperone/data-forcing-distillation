# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os

from omegaconf import open_dict

import fastgen.configs.methods.config_kd_causal as config_kd_causal_default
from fastgen.callbacks.ema import EMACallback
from fastgen.configs.data import DATA_ROOT_DIR, PathLoaderConfig
from fastgen.configs.net import CausalWan_1_3B_Config
from fastgen.utils import LazyCall as L

PATH_DATA_DIR = f"{DATA_ROOT_DIR}/wan1.3b-kd-paths"  # written by scripts/data_gen/generate_wan_kd_paths.py

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

    # KD with 4 steps (needs to be aligned with the steps in the data). The path generator writes the actual
    # recorded teacher timesteps to index.json; use them when available so t_list matches the data exactly.
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
    _index = os.path.join(PATH_DATA_DIR, "index.json")
    if os.path.isfile(_index):
        with open(_index) as f:
            # recorded as float32 (e.g. 0.99900007): round and clamp into the scheduler's valid range
            config.model.sample_t_cfg.t_list = [min(round(float(t), 6), 0.999) for t in json.load(f)["path_timesteps"]]

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

    config.dataloader_train = copy.deepcopy(PathLoaderConfig)
    config.dataloader_train.datatags = [f"WDS:{PATH_DATA_DIR}"]
    config.dataloader_train.batch_size = 2
    # Path samples are ~50 MB each: bound host memory (the loader default is 16 workers x 1000-sample shuffle buffer).
    with open_dict(config.dataloader_train):
        config.dataloader_train.num_workers = 2
        config.dataloader_train.shuffle_size = 32

    config.log_config.group = "wan_kd_causal_swift"
    return config
