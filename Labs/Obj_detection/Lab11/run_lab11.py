"""Execute Lab 11 and save reproducible results inside the Lab11 folder.

Practice labels in this automated run are derived from Penn-Fudan instance masks,
NOT manually drawn. To complete the manual labeling exercise, replace the three
files in work/manual_labels/ with your own annotations using labeler.html.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import detection_dataset as dd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def main():
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    source = dd.download_pennfudan(WORK / "source")
    archive = dd.make_labeling_practice_bundle(source, WORK / "labeling_practice", image_count=3)
    images = sorted((WORK / "labeling_practice" / "images").glob("*.png"))
    manual_dir = WORK / "manual_labels"
    manual_dir.mkdir(exist_ok=True)

    reports = []
    for image in images:
        ids, boxes = dd.reference_labels_for_image(source, image.name)
        label = manual_dir / f"{image.stem}.txt"
        # Source-derived labels, useful for automated verification but not manual annotation.
        dd.write_yolo_labels(label, ids, boxes)
        quality = dd.annotation_quality_report(source, image.name, label)
        reports.append({
            "image": image.name,
            "label": label.name,
            "annotation_source": "reference instance mask (automatically generated)",
            "matched": len(quality["matches"]),
            "duplicate": len(quality["duplicate_indices"]),
            "extra": len(quality["extra_indices"]),
            "missed": len(quality["missed_indices"]),
        })

    base = dd.prepare_base_dataset(source, WORK / "pennfudan_yolo", seed=2026)
    base_summary = dd.validate_yolo_dataset(base)
    augmented = dd.make_augmented_dataset(
        base, WORK / "pennfudan_yolo_augmented", copies_per_image=1, seed=2026
    )
    augmented_summary = dd.validate_yolo_dataset(augmented)
    if base_summary["val"] != augmented_summary["val"] or base_summary["test"] != augmented_summary["test"]:
        raise AssertionError("Validation/test counts changed during augmentation")

    # Extra byte-for-byte checks: validation/test images and annotations must remain untouched.
    for split in ("val", "test"):
        for folder in ("images", "labels"):
            for file in (base / split / folder).iterdir():
                matching = augmented / split / folder / file.name
                if not matching.is_file() or file.read_bytes() != matching.read_bytes():
                    raise AssertionError(f"Changed {split} file: {file}")

    for split in ("train", "val", "test"):
        if augmented_summary[split]["images"] <= 0:
            raise AssertionError(f"Empty split: {split}")
    if augmented_summary["train"]["images"] != 2 * base_summary["train"]["images"]:
        raise AssertionError("Expected one augmented copy per original training image")

    for split, root in (("base", base), ("augmented", augmented)):
        img, ids, boxes, path = dd.load_yolo_sample(root, "train", index=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        dd.plot_yolo_sample(ax, img, ids, boxes)
        ax.set_title(f"{split}: {path.name}")
        fig.savefig(RESULTS / f"{split}_example.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    report = {
        "lab": "Lab 11 - Labeling and Augmenting a Detection Dataset",
        "source": "Penn-Fudan Pedestrian",
        "annotation_note": "Three demonstration labels were GENERATED FROM REFERENCE MASKS; manual labeling not performed.",
        "practice_archive": str(archive.relative_to(ROOT)),
        "practice_annotations": reports,
        "base": base_summary,
        "augmented": augmented_summary,
        "seed": 2026,
        "val_test_unchanged": True,
        "copies_per_training_image": 1
    }
    (RESULTS / "lab11_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
