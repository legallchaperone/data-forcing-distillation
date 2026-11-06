# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate single-step KD pairs (noise, teacher sample, condition) with a class-conditional EDM teacher.

Writes WebDataset shards compatible with `PairLoaderConfig` (`fastgen/configs/data.py`):
    latent.pth   teacher sample x_0 (deterministic EDM Euler sampler, no CFG)
    noise.pth    the latent fed to the one-step student at t = max_t, i.e. eps * sigma(max_t)
    txt_emb.pth  one-hot class label

Example (CIFAR-10, EDM cond-vp teacher from `scripts/download_data.py --dataset cifar10 --only-models`):
    PYTHONPATH=$(pwd) python scripts/data_gen/generate_edm_kd_pairs.py \
        --ckpt $CKPT_ROOT_DIR/cifar10/edm-cifar10-32x32-cond-vp.pth \
        --out-dir $DATA_ROOT_DIR/cifar10-edm-kd-pairs --num-samples 50000
"""

import argparse
import os

import torch
import webdataset as wds
from tqdm import tqdm

import fastgen.utils.logging_utils as logger
from fastgen.configs.data import DATA_ROOT_DIR
from fastgen.configs.net import CKPT_ROOT_DIR, EDM_CIFAR10_Config
from fastgen.utils import instantiate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EDM teacher KD pairs")
    parser.add_argument("--ckpt", default=f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth")
    parser.add_argument("--out-dir", default=f"{DATA_ROOT_DIR}/cifar10-edm-kd-pairs")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=40, help="EDM Euler sampler steps")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--label-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    net = instantiate(EDM_CIFAR10_Config).to(device).eval()
    load_info = net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True), strict=False)
    logger.info(f"Loaded teacher from {args.ckpt}: {load_info}")

    sigmas = net.noise_scheduler.get_t_list(args.num_steps, device=device)
    os.makedirs(args.out_dir, exist_ok=True)
    shape = (args.batch_size, net.img_channels, net.img_resolution, net.img_resolution)

    idx = 0
    with wds.ShardWriter(os.path.join(args.out_dir, "%05d.tar"), maxcount=args.shard_size) as sink:
        for _ in tqdm(range((args.num_samples + args.batch_size - 1) // args.batch_size), desc="sampling"):
            eps = torch.randn(shape, device=device)
            labels = torch.randint(0, args.label_dim, (args.batch_size,), device=device)
            cond = torch.nn.functional.one_hot(labels, args.label_dim).float()
            x_T = net.noise_scheduler.latents(noise=eps, t_init=sigmas[0])
            x_0 = net.sample(noise=eps, condition=cond, guidance_scale=None, num_steps=args.num_steps)
            for b in range(args.batch_size):
                if idx >= args.num_samples:
                    break
                sink.write(
                    {
                        "__key__": f"{idx:08d}",
                        "latent.pth": x_0[b].float().cpu(),
                        "noise.pth": x_T[b].float().cpu(),
                        "txt_emb.pth": cond[b].cpu(),
                    }
                )
                idx += 1
    logger.success(f"Wrote {idx} pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
