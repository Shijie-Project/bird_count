"""
Convert per-image YOLO-style annotation .txt files into a single JSON per split.

Input  (YOLO, per image):
    annotations/<split>/<stem>.txt: lines of "class x_norm y_norm"
    images/<split>/<stem>.jpg

Output (custom point-annotation schema, one file per split):
    annotations/<split>.json
    {
      "images": [{"id": <int>, "file_name": "<stem>.jpg",
                       "width": W, "height": H}, ...],
      "annotations": [{"image_id": "<stem>", "points": [[x_px, y_px], ...]}, ...]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_yolo_txt(path: Path, width: int, height: int) -> list[list[float]]:
    """Parse a YOLO .txt into a list of [x_px, y_px] points.

    Tolerates blank lines and ignores any extra columns past x_norm/y_norm.
    """
    points: list[list[float]] = []
    if not path.exists():
        return points

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # YOLO row: class x_norm y_norm [w_norm h_norm ...]
            x = float(parts[1]) * width
            y = float(parts[2]) * height
            points.append([x, y])
    return points


def convert_split(data_root: Path, split: str, out_path: Path) -> None:
    img_dir = data_root / "images" / split
    ann_dir = data_root / "annotations" / split

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    images: list[dict] = []
    annotations: list[dict] = []

    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g. "001"
        try:
            image_id = int(stem)
        except ValueError:
            # Fallback for non-numeric stems: use sequential id.
            image_id = len(images) + 1

        with Image.open(img_path) as im:
            w, h = im.size

        images.append({"id": image_id, "file_name": img_path.name, "width": w, "height": h})

        points = parse_yolo_txt(ann_dir / f"{stem}.txt", w, h)
        annotations.append({"image_id": stem, "points": points})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations}, f, indent=2)

    print(f"[{split}] {len(images)} images, {sum(len(a['points']) for a in annotations)} points -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO per-file txt -> single JSON per split")
    p.add_argument(
        "--data-root", type=Path, default=Path("../data"), help="dataset root containing images/ and annotations/"
    )
    p.add_argument("--splits", nargs="+", default=["all"], help="splits to convert")
    args = p.parse_args()

    for split in args.splits:
        out_path = args.data_root / "annotations" / f"{split}.json"
        convert_split(args.data_root, split, out_path)


if __name__ == "__main__":
    main()
