"""Idempotent compatibility patches for a Self-Forcing checkout running on a GPU without a flash-attn wheel
(e.g. RTX PRO 6000 / Blackwell) and without torchvision.

    python patch_selfforcing.py /path/to/Self-Forcing

Patches:
  1. wan/modules/attention.py : flash-attn import guard tolerates a broken stub package; flash_attention()
     falls back to torch SDPA when no flash-attn is available (the bidirectional WanModel calls it directly).
  2. wan/modules/clip.py       : torchvision import becomes optional (CLIP is only used for image-to-video).
  3. wan/__init__.py           : image2video / text2video imports become optional (they import torchvision).
"""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."


def patch(path, old, new, must=True):
    s = open(path).read()
    if new in s:
        print("already patched:", path)
        return
    if old not in s:
        if must:
            raise SystemExit(f"anchor not found in {path}")
        print("anchor not found, skipped:", path)
        return
    open(path, "w").write(s.replace(old, new, 1))
    print("patched:", path)


att = f"{root}/wan/modules/attention.py"
patch(att,
      "try:\n    import flash_attn\n    FLASH_ATTN_2_AVAILABLE = True\nexcept ModuleNotFoundError:\n    FLASH_ATTN_2_AVAILABLE = False",
      "try:\n    import flash_attn\n    from flash_attn import flash_attn_varlen_func  # noqa: F401\n    FLASH_ATTN_2_AVAILABLE = True\nexcept Exception:\n    FLASH_ATTN_2_AVAILABLE = False",
      must=False)
patch(att,
      "    half_dtypes = (torch.float16, torch.bfloat16)\n    assert dtype in half_dtypes",
      "    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):\n"
      "        # SDPA fallback: padding masks (q_lens / k_lens) are ignored, same as attention() below;\n"
      "        # padded text tokens are zeroed upstream so this is harmless for Wan.\n"
      "        out_dtype = q.dtype\n"
      "        qh, kh, vh = (x.transpose(1, 2).to(dtype) for x in (q, k, v))\n"
      "        if q_scale is not None:\n"
      "            qh = qh * q_scale\n"
      "        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)\n"
      "        return o.transpose(1, 2).contiguous().to(out_dtype)\n"
      "    half_dtypes = (torch.float16, torch.bfloat16)\n    assert dtype in half_dtypes")

patch(f"{root}/wan/modules/clip.py",
      "import torchvision.transforms as T\n",
      "try:\n    import torchvision.transforms as T\nexcept Exception:\n    T = None\n")

init = f"{root}/wan/__init__.py"
patch(init, "from .image2video import WanI2V\n", "try:\n    from .image2video import WanI2V\nexcept Exception:\n    pass\n")
patch(init, "from .text2video import WanT2V\n", "try:\n    from .text2video import WanT2V\nexcept Exception:\n    pass\n")
print("done")
