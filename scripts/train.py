#!/usr/bin/env python3
"""Train OptiSAR-Net++ with the paper configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from ultralytics import OptiSARNetPlusPlus
from ultralytics.models.yolo.optisar_net_plusplus.train_from_scratch import OptiSARNetPlusPlusTrainerFromScratch

NAMES = {
    0: "optical vehicle",
    1: "optical bridge",
    2: "optical crossroad",
    3: "optical T junction",
    4: "optical ground track field",
    5: "optical baseball diamond",
    6: "optical swimming pool",
    7: "optical tennis court",
    8: "optical parking lot",
    9: "optical basketball court",
    10: "optical storage tank",
    11: "optical ship",
    12: "optical airplane",
    13: "optical harbor",
    14: "sar transmission tower",
    15: "sar ship",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("configs/optisar-net-plusplus-66m.yaml"))
    parser.add_argument("--text-cache-dir", type=Path, default=Path("artifacts/text_embeddings"))
    parser.add_argument("--text-model", default="mobileclip2:b")
    parser.add_argument("--dataset-name", default="OptSAR_RSVG")
    parser.add_argument("--neg-text-count", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--balanced-modal-batch",
        action="store_true",
        help="Sample equal numbers of Optical and SAR images in every per-GPU training batch.",
    )
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size used for gradient accumulation.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-interval", type=int, default=20)
    parser.add_argument(
        "--val-event-patience",
        type=int,
        default=0,
        help="Stop after this many validation events fail to improve meanIoU; 0 disables event-based stopping.",
    )
    parser.add_argument("--lr0", type=float, default=2e-3)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms when possible.")
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="optisar_net_pp_m")
    parser.add_argument("--seed", type=int, default=0)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--weights",
        type=Path,
        help="Load model weights only and start a new optimizer/scheduler run.",
    )
    checkpoint_group.add_argument("--resume", type=Path, help="Resume the full training state from a checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Build the pipeline for one epoch on 1%% of the data.")
    return parser.parse_args()


def runtime_dataset_yaml(root: Path, project: Path) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    path = project / "optsar_rsvg.runtime.yaml"
    payload = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": NAMES,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    root = args.dataset_root.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    cache_path = root / "train.cache"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"{cache_path} is missing. Run: python scripts/prepare_dataset_cache.py '{root}' --split train"
        )

    text_cache_dir = args.text_cache_dir.expanduser().resolve()
    prefix = f"adv_{args.dataset_name}"
    required = (
        text_cache_dir / args.text_model / f"{prefix}_train_label_embeddings.pt",
        text_cache_dir / args.text_model / f"{prefix}_global_grounding_neg_embeddings_text{args.neg_text_count}.pt",
        text_cache_dir / f"{prefix}_global_grounding_neg_text{args.neg_text_count}.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing text artifacts:\n" + "\n".join(missing))

    dataset_yaml = runtime_dataset_yaml(root, args.project)
    data = {
        "train": {
            "grounding_data": [
                {
                    "img_path": str(root / "images" / "train"),
                    "json_file": str(root / "train.json"),
                }
            ]
        },
        "val": {"yolo_data": [str(dataset_yaml)]},
    }

    checkpoint_path = args.resume or args.weights
    model_source = checkpoint_path.expanduser().resolve() if checkpoint_path else model_path
    if checkpoint_path and not model_source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_source}")

    model = OptiSARNetPlusPlus(str(model_source))
    model.train(
        data=data,
        trainer=OptiSARNetPlusPlusTrainerFromScratch,
        pretrained=False,
        epochs=1 if args.dry_run else args.epochs,
        fraction=0.01 if args.dry_run else 1.0,
        batch=args.batch,
        balanced_modal_batch=args.balanced_modal_batch,
        nbs=args.nbs,
        imgsz=640,
        workers=args.workers,
        device=args.device,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        weight_decay=0.025,
        momentum=0.9,
        warmup_bias_lr=0.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mosaic=args.mosaic,
        close_mosaic=10,
        patience=args.patience,
        val_interval=args.val_interval,
        val_event_patience=args.val_event_patience,
        seed=args.seed,
        deterministic=args.deterministic,
        amp=not args.no_amp,
        val=not args.dry_run,
        save=True,
        plots=not args.dry_run,
        text_model=args.text_model,
        dataset_name=args.dataset_name,
        text_cache_dir=str(text_cache_dir),
        neg_text_count=args.neg_text_count,
        max_text_samples=20,
        adv_sample_ratio=0.5,
        grounding_images_dir=str(root / "images" / "val"),
        grounding_annotations=str(root / "val.json"),
        moe_loss_weight=1.5,
        region_loss_gain=1.0,
        project=str(args.project),
        name=args.name,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
