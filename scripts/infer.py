#!/usr/bin/env python3
"""Run query-conditioned grounding on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ultralytics import OptiSARNetPlusPlus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=1e-4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    model = OptiSARNetPlusPlus(str(args.checkpoint))
    model.eval()
    with torch.inference_mode():
        embeddings = model.model.get_text_pe([args.query], cache_clip_model=True)
        model.model.set_classes([args.query], embeddings)
        result = model.predict(
            str(args.image), device=args.device, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False
        )[0]

    boxes = result.boxes
    prediction = None
    if boxes is not None and len(boxes):
        index = int(boxes.conf.argmax())
        prediction = {
            "query": args.query,
            "confidence": float(boxes.conf[index].cpu()),
            "bbox_xyxy": [float(value) for value in boxes.xyxy[index].cpu()],
        }
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        result.save(filename=str(args.save))


if __name__ == "__main__":
    main()
