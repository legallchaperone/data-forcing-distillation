# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate multi-step KD path data (teacher ODE trajectories) with a bidirectional Wan T2V teacher.

Writes WebDataset shards compatible with `PathLoaderConfig` / `CausalKDModel`:
    latent.pth    final teacher sample x_0                      [C, T, H, W]
    path.pth      teacher latents at the student timesteps      [len(t_list), C, T, H, W]
    txt_emb.pth   umT5 prompt embedding                         [512, 4096]
plus `index.json` with `path_timesteps`: the *actual* teacher timesteps at which the path was recorded
(the closest UniPC steps to `--t-list`). Set `model.sample_t_cfg.t_list` of the KD config to these values.
`neg_prompt_emb.npy` (negative prompt embedding) is saved next to the shards for later use.

Example (Wan2.1-T2V-1.3B, 480p, 81 frames -> 21 latent frames):
    PYTHONPATH=$(pwd) python scripts/data_gen/generate_wan_kd_paths.py \
        --prompts web_dfd_eval_prompts.txt --num-samples 200 --out-dir $DATA_ROOT_DIR/wan1.3b-kd-paths
"""

import argparse
import json
import os

import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

import fastgen.utils.logging_utils as logger
from fastgen.configs.data import DATA_ROOT_DIR
from fastgen.configs.net import Wan_1_3B_Config
from fastgen.utils import instantiate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Wan teacher KD paths")
    parser.add_argument("--prompts", default="web_dfd_eval_prompts.txt", help="text file, one prompt per line")
    parser.add_argument("--neg-prompt-file", default="scripts/inference/prompts/negative_prompt.txt")
    parser.add_argument("--out-dir", default=f"{DATA_ROOT_DIR}/wan1.3b-kd-paths")
    parser.add_argument("--num-samples", type=int, default=200, help="prompts are cycled with new seeds if needed")
    parser.add_argument("--latent-shape", default="16,21,60,104", help="C,T,H,W of the video latent")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--t-list", default="0.999,0.937,0.833,0.624", help="student timesteps to record (desc.)")
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2, help="prompts sampled together")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0, help="resume: first sample index to write")
    return parser.parse_args()


@torch.no_grad()
def sample_with_path(net, noise, condition, neg_condition, t_targets, num_steps, guidance_scale, shift):
    """Same as `Wan.sample`, but also returns the latents at the UniPC steps closest to `t_targets`."""
    sched = net.unipc_scheduler
    sched.config.flow_shift = shift
    sched.set_timesteps(num_inference_steps=num_steps, device=noise.device)
    timesteps = sched.timesteps
    t_norm = (timesteps / sched.config.num_train_timesteps).float()
    record_at = {int(torch.argmin((t_norm - t).abs()).item()): t for t in t_targets}
    recorded, recorded_t = {}, {}

    latents = net.noise_scheduler.latents(noise=noise, t_init=timesteps[0] / sched.config.num_train_timesteps)
    for idx, timestep in enumerate(timesteps):
        if idx in record_at:
            recorded[idx] = latents.clone()
            recorded_t[idx] = float(t_norm[idx])
        t = (timestep / sched.config.num_train_timesteps).expand(latents.shape[0])
        t = net.noise_scheduler.safe_clamp(t, min=net.noise_scheduler.min_t, max=net.noise_scheduler.max_t).to(
            latents.dtype
        )
        flow_pred = net(latents, t, condition=condition, r=None, return_features_early=False, feature_indices={}, return_logvar=False)
        if guidance_scale is not None:
            flow_uncond = net(latents, t, condition=neg_condition, r=None, return_features_early=False, feature_indices={}, return_logvar=False)
            flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)
        latents = sched.step(flow_pred, timestep, latents, return_dict=False)[0]

    order = sorted(record_at.keys())
    path = torch.stack([recorded[i] for i in order], dim=1)  # [B, len(t_list), C, T, H, W]
    path_t = [recorded_t[i] for i in order]
    return latents, path, path_t


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    with open(args.neg_prompt_file) as f:
        neg_prompt = [line.strip() for line in f if line.strip()][0]
    logger.info(f"{len(prompts)} prompts, negative prompt: {neg_prompt[:60]}...")

    net = instantiate(Wan_1_3B_Config)
    net.init_text_encoder()
    net.to(device=device, dtype=dtype).eval().requires_grad_(False)

    neg_emb = net.text_encoder.encode([neg_prompt], precision=dtype).to(device)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "neg_prompt_emb.npy"), neg_emb[0].float().cpu().numpy())

    shape = tuple(int(v) for v in args.latent_shape.split(","))
    t_targets = [float(v) for v in args.t_list.split(",")]
    path_timesteps = None

    assert args.start_idx % args.shard_size == 0, "--start-idx must be a multiple of --shard-size (resume at a shard boundary)"
    shard_pattern = os.path.join(args.out_dir, "%05d.tar")
    B = args.batch_size
    with wds.ShardWriter(shard_pattern, maxcount=args.shard_size, start_shard=args.start_idx // args.shard_size) as sink:
        for start in tqdm(range(args.start_idx, args.num_samples, B), desc="teacher ODE (batches)"):
            idxs = list(range(start, min(start + B, args.num_samples)))
            batch_prompts = [prompts[i % len(prompts)] for i in idxs]
            noise = torch.cat(
                [torch.randn((1, *shape), device=device, dtype=dtype,
                             generator=torch.Generator(device=device).manual_seed(args.seed * 100003 + i)) for i in idxs]
            )
            cond = net.text_encoder.encode(batch_prompts, precision=dtype).to(device)
            x0, path, path_t = sample_with_path(
                net, noise, cond, neg_emb.expand(len(idxs), -1, -1), t_targets, args.num_steps, args.guidance_scale, args.shift
            )
            if path_timesteps is None:
                path_timesteps = path_t + [0.0]
                with open(os.path.join(args.out_dir, "index.json"), "w") as f:
                    json.dump(dict(path_timesteps=path_timesteps, requested_t_list=t_targets + [0.0],
                                   num_steps=args.num_steps, guidance_scale=args.guidance_scale, shift=args.shift,
                                   latent_shape=shape), f, indent=2)
                logger.info(f"Recording path at teacher timesteps {path_t} (requested {t_targets})")
            for b, idx in enumerate(idxs):
                sink.write(
                    {
                        "__key__": f"{idx:08d}",
                        "latent.pth": x0[b].to(torch.float16).cpu(),
                        "path.pth": path[b].to(torch.float16).cpu(),
                        "txt_emb.pth": cond[b].to(torch.float16).cpu(),
                        "json": json.dumps(dict(prompt=batch_prompts[b], seed=args.seed * 100003 + idx)).encode(),
                    }
                )
    logger.success(f"Wrote samples {args.start_idx}..{args.num_samples - 1} to {args.out_dir}")


if __name__ == "__main__":
    main()
