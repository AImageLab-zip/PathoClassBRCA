#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Set

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from UNI.uni import get_encoder


@torch.inference_mode()
def uni_patch_embeddings(model, images: torch.Tensor) -> torch.Tensor:
    embs = model(images)  # (N, 1024)
    if isinstance(embs, np.ndarray):
        embs = torch.from_numpy(embs)
    return embs.float().cpu()


def safe_open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def list_patch_jpgs(slide_dir: Path) -> List[Path]:
    return sorted(slide_dir.glob("*.jpg"))


def parse_xy(name: str):
    m = re.search(r"_x_(\d+)_y_(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_slides_from_csv(csv_path: str) -> Dict[str, Set[str]]:
    slides: Dict[str, Set[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("status") or "").strip() != "FOUND":
                continue
            cancer = (r.get("cancer") or "").strip()
            slide = (r.get("slide_base") or "").strip()
            if not cancer or not slide:
                continue
            slides.setdefault(cancer, set()).add(slide)
    return slides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True,
                    help="Root: /work/.../TCGA (contains <CANCER>/images/...)")
    ap.add_argument("--csv", required=True, help="CSV file with slide list")
    ap.add_argument("--out_root", required=True, help="Output root for .pt tensors")
    ap.add_argument("--uni_path", required=True, help="Assets dir for the UNI encoder")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--cancers", nargs="+", default=["BRCA"],
                    help="Cancer types to process")
    args = ap.parse_args()

    print("Loading CSV...")
    csv_slides = load_slides_from_csv(args.csv)
    print({k: len(v) for k, v in csv_slides.items()})

    device = args.device if torch.cuda.is_available() else "cpu"

    print("Loading UNI encoder...")
    model, transform = get_encoder(enc_name="uni", device=device, assets_dir=args.uni_path)

    images_root = Path(args.images_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for cancer in args.cancers:
        if cancer not in csv_slides:
            print(f"Skipping cancer {cancer} (not in CSV)")
            continue

        cancer_dir = images_root / cancer / "images"
        if not cancer_dir.is_dir():
            print(f"Skipping {cancer}: missing folder {cancer_dir}")
            continue

        print(f"\n=== Processing cancer: {cancer} ===")

        out_dir = out_root / cancer
        out_dir.mkdir(parents=True, exist_ok=True)

        allowed_slides = csv_slides[cancer]

        slide_dirs = sorted([
            cancer_dir / slide_id
            for slide_id in allowed_slides
            if (cancer_dir / slide_id).is_dir()
        ])

        print(f"{cancer} slides in CSV: {len(allowed_slides)}")
        print(f"{cancer} slide folders matched: {len(slide_dirs)}")

        for slide_dir in tqdm(slide_dirs, desc=f"{cancer} slides"):
            slide_id = slide_dir.name

            out_pt = out_dir / f"{slide_id}.pt"
            if args.skip_existing and out_pt.exists():
                continue

            patch_files = list_patch_jpgs(slide_dir)
            if not patch_files:
                continue

            # raster scan (y, x) with validation
            bad = [p.name for p in patch_files if parse_xy(p.name) is None][:5]
            if bad:
                raise ValueError(f"{slide_id}: filename senza _x_/_y_ (esempi): {bad}")

            patch_files.sort(key=lambda p: (parse_xy(p.name)[1], parse_xy(p.name)[0]))

            all_embs = []
            for i in range(0, len(patch_files), args.batch_size):
                batch_paths = patch_files[i:i + args.batch_size]
                batch_imgs = [safe_open_rgb(str(p)) for p in batch_paths]

                batch_tensor = torch.stack([transform(img) for img in batch_imgs]).to(device)
                all_embs.append(uni_patch_embeddings(model, batch_tensor))

            patch_tensor = torch.cat(all_embs, dim=0)  # (N, 1024)
            torch.save(patch_tensor, out_pt)

    print("\nAll cancers done.")


if __name__ == "__main__":
    main()