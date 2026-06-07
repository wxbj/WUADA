import numpy as np
from PIL import Image
from pathlib import Path

PALETTE = {
    0: (0,   0,   0),
    1: (255, 0,   0),
    2: (0,   255, 0),
    3: (0,   0,   255),
    4: (255, 255, 0),
}
BG_VAL = 255

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

def is_label_0to4_or_bg(arr2d):
    vals = np.unique(arr2d)
    if np.issubdtype(vals.dtype, np.floating):
        ints = np.rint(vals).astype(np.int64)
        if not np.all(np.isclose(vals, ints)):
            return False
        vals = ints
    return np.all(np.isin(vals, [0,1,2,3,4,BG_VAL]))

def colorize_to_rgba(arr2d):
    a = arr2d.astype(np.int64, copy=False)
    h, w = a.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    fg = (a != BG_VAL)
    rgba[..., 3][fg] = 255
    for v, (r,g,b) in PALETTE.items():
        m = (a == v)
        rgba[..., 0][m] = r
        rgba[..., 1][m] = g
        rgba[..., 2][m] = b
    return rgba

def convert_npy_to_png(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for p in sorted(input_dir.rglob("*.npy")):
        arr = np.load(p, allow_pickle=False)
        gray = middle_channel(arr)
        out_path = output_dir.joinpath(p.relative_to(input_dir)).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if is_label_0to4_or_bg(gray):
            rgba = colorize_to_rgba(gray)
            Image.fromarray(rgba, mode="RGBA").save(out_path)
        else:
            u8 = to_uint8(gray)
            Image.fromarray(u8, mode="L").save(out_path)

# 用法示例：
# convert_npy_to_png("PA_mr2ct_u3plus_r101_RUSH-ADA", "PA_mr2ct_u3plus_r101_RUSH-ADA_png")
# convert_npy_to_png("PA_ct2mr_u3plus_r101_RUSH-ADA", "PA_ct2mr_u3plus_r101_RUSH-ADA_png")
convert_npy_to_png("RA_ct2mr_u3plus_r101_RUSH-ADA", "RA_ct2mr_u3plus_r101_RUSH-ADA_png")
# convert_npy_to_png("RA_mr2ct_u3plus_r101_RUSH-ADA", "RA_mr2ct_u3plus_r101_RUSH-ADA_png")
# convert_npy_to_png("PA_bssfp2lge_u3plus_r101_RUSH-ADA", "PA_bssfp2lge_u3plus_r101_RUSH-ADA_png")
