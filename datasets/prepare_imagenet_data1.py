#!/usr/bin/env python3
"""Stage ImageNet validation data for examples/models/imagenet_example.ipynb.

The notebook expects:

    datasets/imagenet_data1/
      val/<class-folder>/*.JPEG
      train_tiny/<class-folder>/*.JPEG

By default this script uses ImageNet devkit IDs as class folder names
("1".."1000"), which matches the notebook's canonical label-fix cells.
It can also write the standard WNID folder layout with --folder-names wnid.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import shutil
import sys
import tarfile
from pathlib import Path


IMAGE_RE = re.compile(r"^ILSVRC2012_val_(\d{8})\.JPEG$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Extract ILSVRC2012_img_val.tar into an ImageFolder layout named "
            "imagenet_data1, and build train_tiny as a small subset of val."
        )
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=script_dir,
        help="Directory containing ILSVRC2012_img_val.tar and the extracted devkit.",
    )
    parser.add_argument(
        "--val-tar",
        type=Path,
        default=None,
        help="Path to ILSVRC2012_img_val.tar. Defaults to DATASETS_DIR/ILSVRC2012_img_val.tar.",
    )
    parser.add_argument(
        "--devkit-dir",
        type=Path,
        default=None,
        help="Path to ILSVRC2012_devkit_t12. Defaults to DATASETS_DIR/ILSVRC2012_devkit_t12.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to DATASETS_DIR/imagenet_data1.",
    )
    parser.add_argument(
        "--tiny-per-class",
        type=int,
        default=5,
        help="Number of validation images to hard-link/copy into train_tiny per class.",
    )
    parser.add_argument(
        "--folder-names",
        choices=("ilsvrc-id", "wnid"),
        default="ilsvrc-id",
        help=(
            "Class folder naming scheme. Use ilsvrc-id for the existing notebook "
            "label-fix cells, or wnid for the standard ImageNet layout."
        ),
    )
    parser.add_argument(
        "--copy-tiny",
        action="store_true",
        help="Copy train_tiny files instead of hard-linking them from val.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete OUTPUT_DIR first if it already contains staged data.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    datasets_dir = args.datasets_dir.resolve()
    val_tar = (args.val_tar or datasets_dir / "ILSVRC2012_img_val.tar").resolve()
    devkit_dir = (args.devkit_dir or datasets_dir / "ILSVRC2012_devkit_t12").resolve()
    output_dir = (args.output_dir or datasets_dir / "imagenet_data1").resolve()
    return datasets_dir, val_tar, devkit_dir, output_dir


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_ilsvrc_id_to_wnid(meta_path: Path) -> dict[int, str]:
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required to read the ImageNet devkit meta.mat. "
            "Install scipy or run inside the project Docker environment."
        ) from exc

    meta = sio.loadmat(meta_path, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    leaves = [meta[i] for i, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids = list(zip(*leaves))[:2]
    id_to_wnid = {int(idx): str(wnid) for idx, wnid in zip(idcs, wnids)}

    if len(id_to_wnid) != 1000:
        raise RuntimeError(f"Expected 1000 leaf classes in {meta_path}, found {len(id_to_wnid)}")
    return id_to_wnid


def load_ground_truth(path: Path) -> list[int]:
    labels = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 50000:
        raise RuntimeError(f"Expected 50000 validation labels in {path}, found {len(labels)}")
    return labels


def has_payload(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def prepare_output(output_dir: Path, datasets_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    datasets_dir = datasets_dir.resolve()

    protected = {datasets_dir, datasets_dir.parent, Path(output_dir.anchor).resolve()}
    if overwrite and output_dir in protected:
        raise RuntimeError(f"Refusing to overwrite protected directory: {output_dir}")

    if has_payload(output_dir):
        if not overwrite:
            raise RuntimeError(
                f"{output_dir} already exists and is not empty. "
                "Remove it yourself or rerun with --overwrite."
            )
        shutil.rmtree(output_dir)

    val_dir = output_dir / "val"
    tiny_dir = output_dir / "train_tiny"
    val_dir.mkdir(parents=True, exist_ok=True)
    tiny_dir.mkdir(parents=True, exist_ok=True)
    return val_dir, tiny_dir


def link_or_copy(src: Path, dst: Path, copy_tiny: bool) -> None:
    if dst.exists():
        return
    if copy_tiny:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def class_folder_name(ilsvrc_id: int, id_to_wnid: dict[int, str], scheme: str) -> str:
    if scheme == "ilsvrc-id":
        return str(ilsvrc_id)
    if scheme == "wnid":
        return id_to_wnid[ilsvrc_id]
    raise ValueError(f"Unsupported folder scheme: {scheme}")


def stage_validation(
    val_tar: Path,
    val_dir: Path,
    tiny_dir: Path,
    labels: list[int],
    id_to_wnid: dict[int, str],
    folder_names: str,
    tiny_per_class: int,
    copy_tiny: bool,
) -> None:
    if tiny_per_class < 0:
        raise ValueError("--tiny-per-class must be >= 0")

    folder_by_id = {
        ilsvrc_id: class_folder_name(ilsvrc_id, id_to_wnid, folder_names)
        for ilsvrc_id in sorted(id_to_wnid)
    }
    for folder in folder_by_id.values():
        (val_dir / folder).mkdir(parents=True, exist_ok=True)
        (tiny_dir / folder).mkdir(parents=True, exist_ok=True)

    val_counts = {folder: 0 for folder in folder_by_id.values()}
    tiny_counts = {folder: 0 for folder in folder_by_id.values()}
    extracted = 0

    with tarfile.open(val_tar, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue

            filename = posixpath.basename(member.name)
            match = IMAGE_RE.match(filename)
            if not match:
                print(f"Skipping non-validation-image member: {member.name}", file=sys.stderr)
                continue

            image_idx = int(match.group(1))
            if image_idx < 1 or image_idx > len(labels):
                raise RuntimeError(f"Image index out of range in tar member: {member.name}")

            ilsvrc_id = labels[image_idx - 1]
            folder = folder_by_id[ilsvrc_id]
            dst = val_dir / folder / filename

            src = tar.extractfile(member)
            if src is None:
                raise RuntimeError(f"Could not read tar member: {member.name}")
            with src, dst.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)

            val_counts[folder] += 1
            extracted += 1

            if tiny_counts[folder] < tiny_per_class:
                tiny_dst = tiny_dir / folder / filename
                link_or_copy(dst, tiny_dst, copy_tiny)
                tiny_counts[folder] += 1

            if extracted % 5000 == 0:
                print(f"Extracted {extracted}/50000 validation images...")

    missing_val = [folder for folder, count in val_counts.items() if count == 0]
    missing_tiny = [
        folder for folder, count in tiny_counts.items()
        if tiny_per_class > 0 and count < tiny_per_class
    ]

    if extracted != len(labels):
        raise RuntimeError(f"Expected to extract {len(labels)} images, extracted {extracted}")
    if missing_val:
        raise RuntimeError(f"Validation set has empty class folders: {missing_val[:10]}")
    if missing_tiny:
        raise RuntimeError(f"train_tiny has under-filled class folders: {missing_tiny[:10]}")

    print(f"Extracted validation images: {extracted}")
    print(f"train_tiny images: {sum(tiny_counts.values())} ({tiny_per_class} per class)")


def main() -> int:
    args = parse_args()
    datasets_dir, val_tar, devkit_dir, output_dir = resolve_paths(args)
    gt_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    meta_path = devkit_dir / "data" / "meta.mat"

    require_dir(datasets_dir, "Datasets directory")
    require_file(val_tar, "ImageNet validation tar")
    require_dir(devkit_dir, "ImageNet devkit directory")
    require_file(gt_path, "ImageNet validation ground-truth file")
    require_file(meta_path, "ImageNet devkit meta.mat")

    print(f"Validation tar: {val_tar}")
    print(f"Devkit: {devkit_dir}")
    print(f"Output: {output_dir}")
    print(f"Folder names: {args.folder_names}")
    print(f"train_tiny per class: {args.tiny_per_class}")

    id_to_wnid = load_ilsvrc_id_to_wnid(meta_path)
    labels = load_ground_truth(gt_path)
    val_dir, tiny_dir = prepare_output(output_dir, datasets_dir, args.overwrite)
    stage_validation(
        val_tar=val_tar,
        val_dir=val_dir,
        tiny_dir=tiny_dir,
        labels=labels,
        id_to_wnid=id_to_wnid,
        folder_names=args.folder_names,
        tiny_per_class=args.tiny_per_class,
        copy_tiny=args.copy_tiny,
    )

    print("Done.")
    print(f"Validation directory: {val_dir}")
    print(f"Calibration/retraining subset: {tiny_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
