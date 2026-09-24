#!/usr/bin/env python3
"""Evaluate visual grounding metrics on a standard OptSAR-RSVG split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from metrics import intersection_union, summarize
from tqdm import tqdm

from ultralytics import OptiSARNetPlusPlus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--domain", choices=("all", "optical", "sar"), default="all")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=1e-4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    with (root / f"{args.split}.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    images = {item["id"]: item for item in data["images"]}
    annotations = data["annotations"]
    if args.domain != "all":
        annotations = [item for item in annotations if item["category_name"].lower().startswith(args.domain)]
    if args.limit:
        annotations = annotations[: args.limit]

    model = OptiSARNetPlusPlus(str(args.checkpoint))
    model.eval()
    embedding_cache: dict[str, torch.Tensor] = {}
    records: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for annotation in tqdm(annotations, desc=args.split):
        caption = annotation["caption"].strip()
        if caption not in embedding_cache:
            with torch.inference_mode():
                embedding_cache[caption] = model.model.get_text_pe([caption], cache_clip_model=True)
        model.model.set_classes([caption], embedding_cache[caption])
        image = images[annotation["image_id"]]
        image_path = root / "images" / args.split / image["file_name"]
        with torch.inference_mode():
            result = model.predict(
                str(image_path), device=args.device, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False
            )[0]
        prediction = None
        if result.boxes is not None and len(result.boxes):
            index = int(result.boxes.conf.argmax())
            prediction = [float(value) for value in result.boxes.xyxy[index].cpu()]
        record = intersection_union(prediction, annotation["bbox"], image["width"], image["height"])
        domain = "sar" if annotation["category_name"].lower().startswith("sar") else "optical"
        records["all"].append(record)
        records[domain].append(record)

    metrics = {name: summarize(values) for name, values in records.items()}
    rendered = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
