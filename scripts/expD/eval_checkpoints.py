# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate causal-KD checkpoints on held-out teacher paths and decode a few sample videos.

For every checkpoint in the run's checkpoint dir, computes the plain KD regression loss (MSE of the student's
x0 prediction against the teacher ODE endpoint, with the same random timestep draws for every checkpoint) on a
held-out path dataset generated with a different seed, and optionally decodes videos from the last checkpoint.

Example:
    PYTHONPATH=$(pwd) python scripts/expD/eval_checkpoints.py \
        --config fastgen/configs/experiments/WanT2V/config_kd_causal_swift.py \
        --heldout-dir $DATA_ROOT_DIR/wan1.3b-kd-paths-heldout --videos 2 --out /tmp/fg/out/eval_swift.json \
        - log_config.name=kd_causal_swift_a025 model.swift_alpha=0.25 log_config.wandb_mode=disabled
"""

import argparse
import glob
import io
import json
import os
import re
import tarfile

import numpy as np
import torch

import fastgen.utils.logging_utils as logger
from fastgen.methods import CausVidModel
from fastgen.utils import instantiate
from fastgen.utils.checkpointer import Checkpointer
from fastgen.utils.scripts import parse_args, setup


def load_shards(shard_dir, max_samples=None):
    samples = []
    for shard in sorted(glob.glob(os.path.join(shard_dir, "*.tar"))):
        items = {}
        with tarfile.open(shard) as tf:
            for m in tf.getmembers():
                key, _, ext = m.name.partition(".")
                items.setdefault(key, {})[ext] = tf.extractfile(m).read()
        for key in sorted(items):
            it = items[key]
            if not all(k in it for k in ("latent.pth", "path.pth", "txt_emb.pth")):
                continue
            samples.append(
                dict(
                    key=key,
                    real=torch.load(io.BytesIO(it["latent.pth"]), weights_only=True),
                    path=torch.load(io.BytesIO(it["path.pth"]), weights_only=True),
                    condition=torch.load(io.BytesIO(it["txt_emb.pth"]), weights_only=True),
                    prompt=json.loads(it["json"]).get("prompt", "") if "json" in it else "",
                )
            )
            if max_samples and len(samples) >= max_samples:
                return samples
    return samples


@torch.no_grad()
def heldout_loss(model, samples, num_t_draws, seed):
    losses = []
    for draw in range(num_t_draws):
        for i, s in enumerate(samples):
            torch.manual_seed(seed * 1000 + draw * 100 + i)  # identical timestep draws for every checkpoint
            data = {
                "real": s["real"][None].to(model.device, model.precision),
                "path": s["path"][None].to(model.device, model.precision),
                "condition": s["condition"][None].to(model.device, model.precision),
            }
            with model.autocast():
                loss_map, _ = model.single_train_step(data, 0)  # iteration 0: plain target, also for SWIFT models
            losses.append(loss_map["recon_loss"].float().item())
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(len(losses)))


@torch.no_grad()
def decode_videos(model, samples, out_dir, tag, seed=0):
    import imageio

    os.makedirs(out_dir, exist_ok=True)
    net = model.net
    if not hasattr(net, "vae"):
        net.init_vae()
    net.vae.to(device=model.device, dtype=model.precision)
    net.eval()
    ctx = dict(device=model.device, dtype=model.precision)
    for i, s in enumerate(samples):
        g = torch.Generator(device=model.device).manual_seed(seed + i)
        noise = torch.randn(s["real"][None].shape, generator=g, **ctx)
        latents = CausVidModel.generator_fn(
            net,
            noise,
            condition=s["condition"][None].to(**ctx),
            student_sample_steps=model.config.student_sample_steps,
            t_list=model.config.sample_t_cfg.t_list,
            context_noise=getattr(model.config, "context_noise", 0),
            precision_amp=model.precision_amp_infer,
        )
        video = net.vae.decode(latents)[0]  # VAE wrapper handles latent de-normalization; [C, T, H, W] in [-1, 1]
        frames = ((video.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{tag}_{i:02d}.mp4")
        imageio.mimwrite(path, list(frames), fps=16, quality=7)
        with open(os.path.join(out_dir, f"{tag}_{i:02d}.txt"), "w") as f:
            f.write(s["prompt"])
        logger.info(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate causal KD checkpoints on held-out paths")
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--num-t-draws", type=int, default=4)
    parser.add_argument("--videos", type=int, default=0, help="decode this many videos from the last checkpoint")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--out", required=True, help="json output")
    args = parse_args(parser)
    config = setup(args)

    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None
    checkpointer = Checkpointer(config.trainer.checkpointer)
    samples = load_shards(args.heldout_dir, args.max_samples)
    logger.info(f"{len(samples)} held-out samples from {args.heldout_dir}")

    ckpts = sorted(glob.glob(os.path.join(config.trainer.checkpointer.save_dir, "*.pth")),
                   key=lambda p: int(re.search(r"(\d+).pth", p).group(1)))
    ckpts = [p for p in ckpts if re.search(r"/\d+\.pth$", p)]
    results = {"run": config.log_config.name, "ckpt": [], "heldout_mse": [], "heldout_sem": []}
    for ckpt in ckpts:
        checkpointer.load(model.model_dict, path=ckpt)
        model.on_train_begin()
        model.net.eval()
        mean, sem = heldout_loss(model, samples, args.num_t_draws, config.trainer.seed)
        it = int(re.search(r"(\d+).pth", ckpt).group(1))
        results["ckpt"].append(it); results["heldout_mse"].append(mean); results["heldout_sem"].append(sem)
        logger.info(f"{config.log_config.name} ckpt {it}: held-out plain MSE {mean:.5f} ± {sem:.5f}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    if args.videos and ckpts:
        decode_videos(model, samples[: args.videos], args.video_dir or os.path.join(config.log_config.save_path, "videos"),
                      tag=f"{config.log_config.name}_iter{results['ckpt'][-1]}")
    logger.success(f"wrote {args.out}")


if __name__ == "__main__":
    main()
