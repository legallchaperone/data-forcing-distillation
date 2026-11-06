# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import torch
import pytest
from fastgen.methods import KDModel
from fastgen.configs.config import BaseModelConfig as ModelConfig
from fastgen.configs.config_utils import override_config_with_opts


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]

    model = KDModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "noise": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
    }

    return model, data


def test_single_train_step_student_update(get_model_data):
    model, data = get_model_data
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "recon_loss" in loss_map
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], torch.Tensor)


def test_optimizers(get_model_data):
    model, data = get_model_data
    # Test for both student and fake_score optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


@pytest.fixture
def get_swift_model_data():
    gc.collect()
    from fastgen.configs.methods.config_kd import KDModelConfig
    from fastgen.utils import instantiate
    from fastgen.utils import LazyCall as L
    from fastgen.callbacks.ema import EMACallback

    instance = KDModelConfig()
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.input_shape = [3, 8, 8]
    instance.swift_alpha = 0.5
    instance.swift_snapshot_iter = 2

    model = KDModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    snapshot_callback = instantiate(
        L(EMACallback)(type="snapshot", snapshot_iter=instance.swift_snapshot_iter, ema_name="ema_swift")
    )
    snapshot_callback.on_model_init_end(model)

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "noise": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
    }
    return model, data, snapshot_callback


def test_swift_weak_network_setup(get_swift_model_data):
    model, _, _ = get_swift_model_data
    # The weak snapshot is registered as an (extra) EMA network, frozen, and never used for inference
    assert "ema_swift" in model.use_ema
    assert model.ema_swift is not None and not model.ema_swift.training
    assert not any(p.requires_grad for p in model.ema_swift.parameters())
    assert "ema_swift" in model.model_dict
    assert model.net_inference is model.net


def test_swift_target_inactive_before_snapshot(get_swift_model_data):
    model, data, _ = get_swift_model_data
    iteration = 0  # before the snapshot: plain KD target
    assert not model.swift_active(iteration)
    loss_map, outputs = model.single_train_step(data, iteration)
    assert "recon_loss" in loss_map and "recon_loss_plain" in loss_map
    assert loss_map["swift_active"].item() == 0.0
    assert torch.allclose(loss_map["recon_loss"].float(), loss_map["recon_loss_plain"].float())
    assert "gen_rand" in outputs


def test_swift_snapshot_and_extrapolated_target(get_swift_model_data):
    model, data, snapshot_callback = get_swift_model_data
    init_state = {k: v.detach().clone().float() for k, v in model.ema_swift.state_dict().items()}
    # Train a few steps so the student drifts away from its initialization
    for iteration in range(3):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)
        snapshot_callback.on_training_step_end(model, None, None, None, iteration=iteration)
        if iteration < model.swift_snapshot_iter:
            # the snapshot callback must leave the weak network untouched on every other iteration
            assert all(torch.allclose(init_state[k], v.float()) for k, v in model.ema_swift.state_dict().items())

    # After `swift_snapshot_iter` the weak network equals the student at that iteration (and differs from init)
    assert not all(torch.allclose(init_state[k], v.float()) for k, v in model.ema_swift.state_dict().items())
    net_state = {k: v.detach().float() for k, v in model.net.state_dict().items()}
    assert all(torch.allclose(net_state[k], v.float()) for k, v in model.ema_swift.state_dict().items())

    # SWIFT is active from the next iteration on and the target is extrapolated away from the weak prediction
    iteration = model.swift_snapshot_iter + 1
    assert model.swift_active(iteration)
    t_student = torch.full((1,), model.net.noise_scheduler.max_t, device=model.device)
    with torch.no_grad():
        weak_pred = model.ema_swift(data["noise"], t_student, condition=data["condition"], fwd_pred_type="x0")
        target = model.swift_target(data["real"], data["noise"], t_student, data["condition"], iteration)
    expected = data["real"] + model.swift_alpha * (data["real"] - weak_pred.to(data["real"].dtype))
    assert torch.allclose(target.float(), expected.float(), atol=1e-5)

    loss_map, _ = model.single_train_step(data, iteration)
    assert loss_map["swift_active"].item() == 1.0
