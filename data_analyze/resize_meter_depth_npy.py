#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch process meter-depth NPY files:
1) find *_meter.npy under input root
2) center-crop to square
3) resize to target resolution (default 256x256)
4) save to output root while preserving relative paths
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def center_crop_to_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return arr[y0 : y0 + side, x0 : x0 + side]


def to_2d_depth(arr: np.ndarray, src_path: Path) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported depth shape {arr.shape} in {src_path}")


def process_one(src: Path, dst: Path, target_size: int, interpolation: int) -> None:
    arr = np.load(src)
    arr = to_2d_depth(arr, src)
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    cropped = center_crop_to_square(arr)
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=interpolation)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, resized.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch crop+resize meter depth npy files to 256x256."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("test_modules/test_results"),
        help="Root dir to search for *_meter.npy",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("test_modules/test_results_meter_256"),
        help="Output root directory",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Output depth resolution (square)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_meter.npy",
        help="Glob pattern under input_root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="nearest",
        choices=["nearest", "linear", "area", "cubic"],
        help="Resize interpolation",
    )
    return parser.parse_args()


def get_interpolation(name: str) -> int:
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    files = sorted(input_root.rglob(args.pattern))
    if not files:
        print(f"[WARN] No files matched pattern '{args.pattern}' under: {input_root}")
        return

    interpolation = get_interpolation(args.interpolation)

    processed = 0
    skipped = 0
    failed = 0

    for src in files:
        rel = src.relative_to(input_root)
        dst = output_root / rel

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            process_one(src, dst, args.target_size, interpolation)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[ERR] {src}: {exc}")

    print("[DONE] meter depth batch processing complete")
    print(f"[INFO] input_root:  {input_root}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] matched:     {len(files)}")
    print(f"[INFO] processed:   {processed}")
    print(f"[INFO] skipped:     {skipped}")
    print(f"[INFO] failed:      {failed}")


if __name__ == "__main__":
    main()
