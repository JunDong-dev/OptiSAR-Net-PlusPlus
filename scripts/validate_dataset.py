#!/usr/bin/env python3
"""Validate the released OptSAR-RSVG directory without loading image pixels."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

EXPECTED_IMAGES = {"train": 37_957, "val": 4_434, "test": 4_434}
EXPECTED_ANNOTATIONS = {"train": 74_049, "val": 7_996, "test": 8_103}
ALLOWED_TOP_LEVEL = {"images", "labels", "train.json", "val.json", "test.json"}
RELEASE_FILENAME = re.compile(r"^(opt|sar)_(\d{6})\.[A-Za-z0-9]+$")


def validate_split(root: Path, split: str, strict_release: bool) -> tuple[dict, list[str]]:
    errors: list[str] = []
    annotation_path = root / f"{split}.json"
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    if not annotation_path.is_file():
        return {}, [f"missing annotation file: {annotation_path}"]
    if not image_dir.is_dir():
        return {}, [f"missing image directory: {image_dir}"]
    if not label_dir.is_dir():
        return {}, [f"missing label directory: {label_dir}"]

    with annotation_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    image_by_id = {item["id"]: item for item in images}
    image_ids = set(image_by_id)
    category_ids = {item["id"] for item in categories}
    category_names = {item["id"]: item["name"].strip().lower() for item in categories}
    filenames = [item["file_name"] for item in images]
    disk_images = {path.name for path in image_dir.iterdir() if path.is_file() and not path.name.startswith(".")}
    disk_labels = {path.name for path in label_dir.glob("*.txt")}

    if len(image_ids) != len(images):
        errors.append(f"{split}: duplicate image IDs")
    if len(set(filenames)) != len(filenames):
        errors.append(f"{split}: duplicate image filenames")
    if len({item.get("id") for item in annotations}) != len(annotations):
        errors.append(f"{split}: duplicate annotation IDs")
    if len(category_ids) != len(categories):
        errors.append(f"{split}: duplicate category IDs")

    referenced = set(filenames)
    missing_images = referenced - disk_images
    extra_images = disk_images - referenced
    expected_labels = {f"{Path(name).stem}.txt" for name in referenced}
    missing_labels = expected_labels - disk_labels
    extra_labels = disk_labels - expected_labels
    dangling_annotations = sum(ann.get("image_id") not in image_ids for ann in annotations)
    dangling_categories = sum(ann.get("category_id") not in category_ids for ann in annotations)
    invalid_boxes = 0
    domains_by_image: dict[int, set[str]] = defaultdict(set)
    for ann in annotations:
        category_name = category_names.get(ann.get("category_id"), "")
        if category_name.startswith("optical "):
            domains_by_image[ann.get("image_id")].add("opt")
        elif category_name.startswith("sar "):
            domains_by_image[ann.get("image_id")].add("sar")
        bbox = ann.get("bbox", [])
        image = image_by_id.get(ann.get("image_id"))
        if len(bbox) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
            invalid_boxes += 1
            continue
        x, y, width, height = bbox
        if (
            image is None
            or x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image["width"] + 1e-6
            or y + height > image["height"] + 1e-6
        ):
            invalid_boxes += 1

    invalid_label_rows = 0
    label_boxes = 0
    for label_name in disk_labels:
        for line in (label_dir / label_name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label_boxes += 1
            fields = line.split()
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except (IndexError, ValueError):
                invalid_label_rows += 1
                continue
            if len(fields) != 5 or len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
                invalid_label_rows += 1
                continue
            cx, cy, width, height = coordinates
            if (
                class_id not in category_ids
                or width <= 0
                or height <= 0
                or cx - width / 2 < -1e-6
                or cy - height / 2 < -1e-6
                or cx + width / 2 > 1 + 1e-6
                or cy + height / 2 > 1 + 1e-6
            ):
                invalid_label_rows += 1

    if missing_images:
        errors.append(f"{split}: {len(missing_images)} referenced images are missing")
    if extra_images:
        errors.append(f"{split}: {len(extra_images)} unreferenced images are present")
    if missing_labels:
        errors.append(f"{split}: {len(missing_labels)} YOLO label files are missing")
    if extra_labels:
        errors.append(f"{split}: {len(extra_labels)} unreferenced YOLO label files are present")
    if dangling_annotations:
        errors.append(f"{split}: {dangling_annotations} annotations reference unknown images")
    if dangling_categories:
        errors.append(f"{split}: {dangling_categories} annotations reference unknown categories")
    if invalid_boxes:
        errors.append(f"{split}: {invalid_boxes} annotations have invalid boxes")
    if invalid_label_rows:
        errors.append(f"{split}: {invalid_label_rows} YOLO label rows are invalid")
    if len(categories) != 16:
        errors.append(f"{split}: expected 16 categories, found {len(categories)}")

    if strict_release:
        malformed_names = 0
        domain_mismatches = 0
        for image in images:
            match = RELEASE_FILENAME.fullmatch(image["file_name"])
            if match is None:
                malformed_names += 1
                continue
            annotation_domains = domains_by_image.get(image["id"], set())
            if annotation_domains != {match.group(1)}:
                domain_mismatches += 1
        if malformed_names:
            errors.append(f"{split}: {malformed_names} filenames do not match opt/sar_XXXXXX")
        if domain_mismatches:
            errors.append(f"{split}: {domain_mismatches} filename prefixes disagree with annotations")

    if strict_release:
        if len(images) != EXPECTED_IMAGES[split]:
            errors.append(f"{split}: expected {EXPECTED_IMAGES[split]} images, found {len(images)}")
        if len(annotations) != EXPECTED_ANNOTATIONS[split]:
            errors.append(f"{split}: expected {EXPECTED_ANNOTATIONS[split]} annotations, found {len(annotations)}")

    summary = {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "image_files": len(disk_images),
        "label_files": len(disk_labels),
        "label_boxes": label_boxes,
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--strict-release", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()

    errors: list[str] = []
    summaries = {}
    if args.strict_release:
        present = {path.name for path in root.iterdir() if not path.name.startswith(".")}
        unexpected = present - ALLOWED_TOP_LEVEL
        if unexpected:
            errors.append(f"unexpected top-level entries: {sorted(unexpected)}")

    split_filenames: dict[str, set[str]] = {}
    domain_numbers: dict[str, list[int]] = defaultdict(list)
    for split in ("train", "val", "test"):
        summary, split_errors = validate_split(root, split, args.strict_release)
        summaries[split] = summary
        errors.extend(split_errors)
        json_path = root / f"{split}.json"
        if json_path.is_file():
            with json_path.open(encoding="utf-8") as handle:
                split_filenames[split] = {item["file_name"] for item in json.load(handle)["images"]}
            if args.strict_release:
                for name in split_filenames[split]:
                    match = RELEASE_FILENAME.fullmatch(name)
                    if match:
                        domain_numbers[match.group(1)].append(int(match.group(2)))

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_filenames.get(left, set()) & split_filenames.get(right, set())
        if overlap:
            errors.append(f"{left}/{right}: {len(overlap)} filenames overlap")

    if args.strict_release:
        for domain in ("opt", "sar"):
            numbers = domain_numbers[domain]
            expected = list(range(1, len(numbers) + 1))
            if sorted(numbers) != expected:
                errors.append(f"{domain}: numbering is not unique and continuous from 000001")

    report = {"dataset_root": str(root), "splits": summaries, "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
