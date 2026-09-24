#!/usr/bin/env python3
"""Precompute caption and adversarial-negative embeddings for training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics.data.augment import adv_RandomLoadText
from ultralytics.nn.text_model import build_text_model


def encode(model, texts: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="text embeddings"):
        tokens = model.tokenize(texts[start : start + batch_size])
        chunks.append(model.encode_text(tokens).cpu())
    return torch.cat(chunks)


def variants(text: str) -> set[str]:
    lowered = text.lower()
    return {
        lowered.replace(source, target, 1)
        for source, target in adv_RandomLoadText._ADV_REPLACEMENTS
        if source in lowered
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--dataset-name", default="OptSAR_RSVG")
    parser.add_argument("--text-model", default="mobileclip2:b")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/text_embeddings"))
    parser.add_argument("--global-negatives", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    annotation_path = args.dataset_root.expanduser().resolve() / "train.json"
    with annotation_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]

    counts = Counter(
        annotation["caption"].strip() for annotation in annotations if annotation.get("caption", "").strip()
    )
    categories = {
        annotation["category_name"].strip() for annotation in annotations if annotation.get("category_name", "").strip()
    }
    training_texts = set(counts) | categories | {"background"}
    for text in tuple(training_texts):
        training_texts.update(variants(text))
    ordered_training = sorted(training_texts)
    global_texts = [text for text, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    global_texts = global_texts[: args.global_negatives]

    model = build_text_model(args.text_model, device=torch.device(args.device))
    training_features = encode(model, ordered_training, args.batch_size)
    global_features = encode(model, global_texts, args.batch_size)

    model_dir = args.output_dir / args.text_model
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"adv_{args.dataset_name}"
    text_path = args.output_dir / f"{prefix}_global_grounding_neg_text{args.global_negatives}.json"
    negative_path = model_dir / f"{prefix}_global_grounding_neg_embeddings_text{args.global_negatives}.pt"
    training_path = model_dir / f"{prefix}_train_label_embeddings.pt"

    text_path.write_text(json.dumps(global_texts, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(global_features, negative_path)
    torch.save(dict(zip(ordered_training, training_features)), training_path)
    print(f"wrote {training_path}")
    print(f"wrote {negative_path}")
    print(f"wrote {text_path}")


if __name__ == "__main__":
    main()
