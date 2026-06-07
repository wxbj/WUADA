import numpy as np
from PIL import Image
from pathlib import Path

def to_uint8(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = a.min()
    vmax = a.max()
    if vmax == vmin:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - vmin) / (vmax - vmin)
    return (a * 255).clip(0, 255).astype(np.uint8)

def middle_channel(x):
    a = np.asarray(x)
    if a.ndim == 2:
        return a
    a = np.squeeze(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[-1] == 3:
            return a[..., 1]
        if a.shape[0] == 3:
            return a[1]
        for ax, size in enumerate(a.shape):
            if size == 3:
                slc = [slice(None)] * a.ndim
                slc[ax] = 1
                return a[tuple(slc)]
    raise ValueError(f"shape {a.shape} not supported")

def convert_npy_to_png(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for p in sorted(input_dir.rglob("*.npy")):
        arr = np.load(p)
        gray = middle_channel(arr)
        u8 = to_uint8(gray)
        out_path = output_dir.joinpath(p.relative_to(input_dir)).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(u8).save(out_path)


# 用法示例：
convert_npy_to_png("MM-WHS", "MM-WHS-png")
convert_npy_to_png("MS-CMRSeg", "MS-CMRSeg-png")

