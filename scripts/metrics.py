"""Visual grounding metrics used by evaluation and tests."""

from __future__ import annotations

import numpy as np

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def intersection_union(
    prediction: list[float] | None, target: list[float], width: int, height: int
) -> tuple[float, float]:
    tx1, ty1, tw, th = target
    tx2, ty2 = tx1 + tw, ty1 + th
    if prediction is None:
        target_area = max(0.0, min(tx2, width) - max(tx1, 0.0)) * max(0.0, min(ty2, height) - max(ty1, 0.0))
        return 0.0, target_area
    px1, py1, px2, py2 = prediction
    px1, px2 = np.clip((px1, px2), 0, width)
    py1, py2 = np.clip((py1, py2), 0, height)
    tx1, tx2 = np.clip((tx1, tx2), 0, width)
    ty1, ty2 = np.clip((ty1, ty2), 0, height)
    intersection = max(0.0, min(px2, tx2) - max(px1, tx1)) * max(0.0, min(py2, ty2) - max(py1, ty1))
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    return intersection, pred_area + target_area - intersection


def summarize(records: list[tuple[float, float]]) -> dict[str, float | int]:
    ious = [intersection / union if union else 0.0 for intersection, union in records]
    return {
        "samples": len(records),
        "meanIoU": float(np.mean(ious)) if ious else 0.0,
        "cumIoU": sum(item[0] for item in records) / sum(item[1] for item in records) if records else 0.0,
        **{
            f"Pr@{threshold}": float(np.mean([iou >= threshold for iou in ious])) if ious else 0.0
            for threshold in THRESHOLDS
        },
    }
