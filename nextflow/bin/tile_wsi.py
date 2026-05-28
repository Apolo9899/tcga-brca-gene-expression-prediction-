#!/usr/bin/env python3
"""
Tile a whole slide image (WSI) into 256×256 px fragments at 20× magnification.

Tissue filter: tiles whose mean luminance (ITU-R BT.601 coefficients) exceeds
--lum-thr (default 210/255) are discarded as background or adipose tissue.

At most --max-tiles valid tiles are saved per WSI, selected randomly.
"""
import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


LUMINANCE_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def luminance(tile_rgb: np.ndarray) -> float:
    return float((tile_rgb.astype(np.float32) * LUMINANCE_WEIGHTS).sum(axis=-1).mean())


def tile_wsi_openslide(wsi_path: Path, outdir: Path,
                       tile_size: int, mag: int,
                       lum_thr: float, max_tiles: int,
                       seed: int = 33) -> int:
    slide = openslide.OpenSlide(str(wsi_path))

    # Find level closest to requested magnification
    native_mag = float(slide.properties.get(
        openslide.PROPERTY_NAME_OBJECTIVE_POWER, mag))
    downsample = native_mag / mag
    level = slide.get_best_level_for_downsample(downsample)
    level_ds = slide.level_downsamples[level]

    w, h = slide.level_dimensions[level]
    step = tile_size

    # Collect all candidate positions
    positions = [
        (col, row)
        for row in range(0, h - step + 1, step)
        for col in range(0, w - step + 1, step)
    ]
    random.seed(seed)
    random.shuffle(positions)

    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for col, row in tqdm(positions, desc=wsi_path.stem, leave=False):
        if saved >= max_tiles:
            break
        # Read at native resolution then resize to tile_size
        x = int(col * level_ds)
        y = int(row * level_ds)
        region = slide.read_region((x, y), level, (step, step)).convert("RGB")
        arr = np.array(region)
        if luminance(arr) > lum_thr:
            continue   # background / fat
        fname = outdir / f"{wsi_path.stem}_r{row:05d}_c{col:05d}.png"
        region.save(str(fname))
        saved += 1

    slide.close()
    return saved


def tile_wsi_pillow(wsi_path: Path, outdir: Path,
                    tile_size: int, lum_thr: float,
                    max_tiles: int, seed: int = 33) -> int:
    """Fallback for non-SVS formats using Pillow."""
    img = Image.open(str(wsi_path)).convert("RGB")
    w, h = img.size
    positions = [
        (col, row)
        for row in range(0, h - tile_size + 1, tile_size)
        for col in range(0, w - tile_size + 1, tile_size)
    ]
    random.seed(seed)
    random.shuffle(positions)

    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for col, row in tqdm(positions, desc=wsi_path.stem, leave=False):
        if saved >= max_tiles:
            break
        tile = img.crop((col, row, col + tile_size, row + tile_size))
        arr = np.array(tile)
        if luminance(arr) > lum_thr:
            continue
        fname = outdir / f"{wsi_path.stem}_r{row:05d}_c{col:05d}.png"
        tile.save(str(fname))
        saved += 1

    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wsi",       required=True, type=Path)
    ap.add_argument("--outdir",    required=True, type=Path)
    ap.add_argument("--tile-size", type=int,   default=256)
    ap.add_argument("--mag",       type=int,   default=20)
    ap.add_argument("--lum-thr",   type=float, default=210.0)
    ap.add_argument("--max-tiles", type=int,   default=500)
    ap.add_argument("--seed",      type=int,   default=33)
    args = ap.parse_args()

    if HAS_OPENSLIDE and args.wsi.suffix.lower() in {".svs", ".ndpi", ".scn", ".mrxs"}:
        n = tile_wsi_openslide(args.wsi, args.outdir,
                               args.tile_size, args.mag,
                               args.lum_thr, args.max_tiles, args.seed)
    else:
        n = tile_wsi_pillow(args.wsi, args.outdir,
                            args.lum_thr, args.max_tiles, args.seed)

    print(f"{args.wsi.name}: {n} valid tiles saved → {args.outdir}")


if __name__ == "__main__":
    main()
