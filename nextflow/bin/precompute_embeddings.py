#!/usr/bin/env python3
"""
Precompute tile embeddings for a given foundation model backbone.

For each WSI (identified by slide_id = tile filename prefix), loads all PNG tiles,
runs them through the frozen backbone, and saves a (N_tiles, D) tensor as
<outdir>/<slide_id>.pt

Supported backbones: dinov2, dinov3, phikon, uni, virchow2, gigapath
"""
import argparse, os
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


SEED = 33
torch.manual_seed(SEED)

BACKBONE_EMBED_DIM = {
    "dinov2":   384,
    "dinov3":   384,
    "phikon":   768,
    "uni":     1024,
    "virchow2": 1280,
    "gigapath": 1536,
}

TILE_MEAN = [0.485, 0.456, 0.406]
TILE_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=TILE_MEAN, std=TILE_STD),
])


def load_backbone(backbone: str, device: torch.device) -> torch.nn.Module:
    backbone = backbone.lower()
    if backbone == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                               pretrained=True)
    elif backbone == "dinov3":
        model = torch.hub.load("facebookresearch/dinov2",
                               "dinov2_vits14_reg", pretrained=True)
    elif backbone == "phikon":
        from transformers import ViTModel
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        model._forward = model.forward
        def _phikon_forward(pixel_values):
            out = model._forward(pixel_values=pixel_values)
            return out.last_hidden_state[:, 0]   # CLS token
        model.forward = _phikon_forward
    elif backbone == "uni":
        import timm
        model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True,
                                  init_values=1e-5, dynamic_img_size=True)
        model.forward = lambda x: model.forward_features(x)[:, 0]
    elif backbone == "virchow2":
        import timm
        model = timm.create_model("hf-hub:paige-ai/Virchow2",
                                  pretrained=True, mlp_layer=timm.layers.SwiGLUPacked,
                                  act_layer=torch.nn.SiLU)
        def _virchow_forward(x):
            out = model.forward_features(x)
            cls, patch = out[:, 0], out[:, 5:]
            return torch.cat([cls, patch.mean(1)], dim=-1)
        model.forward = _virchow_forward
    elif backbone == "gigapath":
        import timm
        model = timm.create_model("hf-hub:prov-gigapath/prov-gigapath",
                                  pretrained=True)
        model.forward = lambda x: model.forward_features(x)[:, 0]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def group_tiles_by_slide(tiles_dir: Path) -> dict[str, list[Path]]:
    """Group PNG files by slide_id (filename prefix before first _r)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(tiles_dir.rglob("*.png")):
        slide_id = f.stem.rsplit("_r", 1)[0]
        groups[slide_id].append(f)
    return groups


@torch.no_grad()
def embed_slide(tile_paths: list[Path], model: torch.nn.Module,
                device: torch.device, batch_size: int = 32) -> torch.Tensor:
    embeddings = []
    for i in range(0, len(tile_paths), batch_size):
        batch_files = tile_paths[i: i + batch_size]
        imgs = torch.stack([
            TRANSFORM(Image.open(str(f)).convert("RGB"))
            for f in batch_files
        ]).to(device)
        emb = model(imgs).cpu()
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone",   required=True)
    ap.add_argument("--tiles-dir",  required=True, type=Path)
    ap.add_argument("--outdir",     required=True, type=Path)
    ap.add_argument("--max-tiles",  type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Backbone: {args.backbone}  |  Device: {device}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading backbone…")
    model = load_backbone(args.backbone, device)

    print("Grouping tiles by slide…")
    slides = group_tiles_by_slide(args.tiles_dir)
    print(f"  {len(slides)} slides found.")

    for slide_id, tile_paths in tqdm(slides.items(), desc="Embedding slides"):
        out_path = args.outdir / f"{slide_id}.pt"
        if out_path.exists():
            continue   # already computed
        # Limit to max_tiles (already shuffled in tile_wsi.py)
        tile_paths = tile_paths[: args.max_tiles]
        emb = embed_slide(tile_paths, model, device, args.batch_size)
        torch.save(emb, str(out_path))

    print(f"Done. Embeddings saved to {args.outdir}")


if __name__ == "__main__":
    main()
