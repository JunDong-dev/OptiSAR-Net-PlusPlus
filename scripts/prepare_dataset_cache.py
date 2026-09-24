#!/usr/bin/env python3
"""Create the NumPy grounding cache consumed by the training dataloader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def xywhn(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = bbox
    return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]


def build_cache(annotation_path: Path, image_dir: Path, output: Path) -> None:
    with annotation_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    images = {item["id"]: item for item in data["images"]}
    grouped = {image_id: [] for image_id in images}
    for annotation in data["annotations"]:
        grouped[annotation["image_id"]].append(annotation)

    labels = []
    instances = 0
    for image_id, image_annotations in tqdm(grouped.items(), desc=annotation_path.stem):
        image = images[image_id]
        image_path = (image_dir / image["file_name"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        texts: list[list[str]] = []
        categories: list[list[str]] = []
        text_to_id: dict[str, int] = {}
        category_seen: set[str] = set()
        boxes: list[list[float]] = []
        classes: list[list[float]] = []
        seen: set[tuple] = set()

        for annotation in image_annotations:
            if annotation.get("iscrowd", 0):
                continue
            caption = annotation["caption"].strip()
            category = annotation["category_name"].strip()
            bbox = annotation["bbox"]
            if not caption or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            if caption not in text_to_id:
                text_to_id[caption] = len(texts)
                texts.append([caption])
            if category not in category_seen:
                category_seen.add(category)
                categories.append([category])
            normalized = xywhn(bbox, int(image["width"]), int(image["height"]))
            key = (text_to_id[caption], *normalized)
            if key in seen:
                continue
            seen.add(key)
            classes.append([float(text_to_id[caption])])
            boxes.append(normalized)

        instances += len(boxes)
        labels.append(
            {
                "im_file": image_path,
                "shape": (int(image["height"]), int(image["width"])),
                "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
                "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                "segments": [],
                "normalized": True,
                "bbox_format": "xywh",
                "texts": texts,
                "cat_names": categories,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".npy")
    np.save(temporary, np.asarray(labels, dtype=object), allow_pickle=True)
    temporary.replace(output)
    print(f"wrote {output} ({len(labels)} images, {instances} instances)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output or root / f"{args.split}.cache"
    build_cache(root / f"{args.split}.json", root / "images" / args.split, output)


if __name__ == "__main__":
    main()
