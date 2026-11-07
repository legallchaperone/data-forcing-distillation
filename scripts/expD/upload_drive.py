# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upload Exp D artifacts to Google Drive from a standalone process (serial, resumable, skip-if-same-size).

Uses gdrive_fsspec with the credentials cached by an earlier interactive authorization on the same machine
(pydata-google-auth cache), so it must run after the notebook's auth cell has succeeded once.

    python scripts/expD/upload_drive.py --folder fastgen_expD --final-only
"""

import argparse
import glob
import os
import time

from gdrive_fsspec import GoogleDriveFileSystem

OUT = "/tmp/fg/out/fastgen/wan_kd_causal_swift"
RUNS = {"kd_causal_baseline": "baseline", "kd_causal_swift_a025": "swift_a025"}


def jobs(final_only: bool, with_videos: bool):
    j = []
    for run, tag in RUNS.items():
        ck = sorted(glob.glob(f"{OUT}/{run}/checkpoints/*.pth"))
        for p in (ck[-1:] if final_only else ck):
            j.append((p, f"{tag}/checkpoints/{os.path.basename(p)}"))
    for r in ("baseline", "swift"):
        for p in (f"/tmp/fg/out/evalD_{r}.json", f"/tmp/fg/trainD_{r}.log", f"/tmp/fg/evalD_{r}.log"):
            if os.path.exists(p):
                j.append((p, f"{'results' if p.endswith('.json') else 'logs'}/{os.path.basename(p)}"))
    if with_videos:
        for p in sorted(glob.glob(f"{OUT}/*/videos/*")):
            j.append((p, f"videos/{os.path.basename(p)}"))
    # small files first, checkpoints last
    return sorted(j, key=lambda x: os.path.getsize(x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="fastgen_expD")
    ap.add_argument("--final-only", action="store_true", help="only the last checkpoint of each run")
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()
    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    for local, rel in jobs(args.final_only, not args.no_videos):
        remote = f"{args.folder}/{rel}"
        size = os.path.getsize(local)
        try:
            if fs.exists(remote) and fs.size(remote) == size:
                print(f"skip   {rel} ({size/1e6:.0f} MB, already there)", flush=True)
                continue
        except Exception:
            pass
        try:
            fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
        except Exception:
            pass
        t0 = time.time()
        fs.put_file(local, remote)
        print(f"ok     {rel} ({size/1e6:.0f} MB in {time.time()-t0:.0f}s)", flush=True)
    print("UPLOAD DONE", flush=True)


if __name__ == "__main__":
    main()
