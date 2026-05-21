#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate t-SNE visualization for simulation and real depth observations.

Input:
    A YAML file containing paths to sim and real depth images.

Supported formats:
    .npy, .npz, .csv, .tsv, .png, .jpg, .jpeg, .tiff

Expected depth shape:
    256 x 256
    or 256 x 256 x 1

Example:
    python depth_tsne.py \
        --config depth_dataset.yaml \
        --output depth_tsne.png
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def expand_paths(path_list: List[str]) -> List[str]:
    """Expand glob patterns and return sorted unique paths."""
    all_paths = []

    for p in path_list:
        expanded = glob.glob(p)
        if len(expanded) == 0:
            print(f"[WARN] No file matched: {p}")
        all_paths.extend(expanded)

    all_paths = sorted(list(set(all_paths)))
    return all_paths


def load_csv_with_fallback_encoding(
    path: str,
    delimiter: str,
    skiprows: int,
) -> np.ndarray:
    """Load CSV/TSV with encoding fallback for Windows locale differences."""
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return np.genfromtxt(
                    f,
                    delimiter=delimiter,
                    skip_header=max(0, int(skiprows)),
                    dtype=np.float32,
                )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to decode text depth file: {path}. "
        f"Tried encodings: {encodings}. Last error: {last_error}"
    )


def load_depth_file(path: str, csv_skiprows: int = 0) -> np.ndarray:
    """Load a depth file and return a 2D float32 array."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)

    elif ext == ".npz":
        data = np.load(path)
        if "depth" in data:
            arr = data["depth"]
        elif "arr_0" in data:
            arr = data["arr_0"]
        else:
            raise KeyError(
                f"{path} is .npz but does not contain 'depth' or 'arr_0'."
            )

    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Failed to read image: {path}")

    elif ext in [".csv", ".tsv"]:
        delimiter = "," if ext == ".csv" else "\t"
        arr = load_csv_with_fallback_encoding(
            path,
            delimiter=delimiter,
            skiprows=csv_skiprows,
        )

    else:
        raise ValueError(f"Unsupported file extension: {ext}, file: {path}")

    arr = np.asarray(arr)

    # If shape is H x W x 1, squeeze channel.
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # If RGB/RGBA image is provided, convert to grayscale.
    if arr.ndim == 3 and arr.shape[-1] in [3, 4]:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    if arr.ndim != 2:
        raise ValueError(f"Depth must be 2D or HxWx1, got shape {arr.shape}: {path}")

    arr = arr.astype(np.float32)

    # Replace invalid values.
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    return arr


def preprocess_depth(
    arr: np.ndarray,
    target_size: int = 256,
    normalize_mode: str = "none",
    max_depth: float = 10.0,
) -> np.ndarray:
    """
    Preprocess depth to target_size x target_size.

    normalize_mode:
        none:
            Use the values as they are.
            Recommended if your data is already normalized to [0, 1].

        max_depth:
            Clip to [0, max_depth] and divide by max_depth.
            Recommended if your data is metric depth in meters.

        image:
            Normalize each image independently to [0, 1].
            Not recommended for sim-vs-real distribution comparison,
            because it removes global depth scale differences.
    """
    if arr.shape != (target_size, target_size):
        arr = cv2.resize(
            arr,
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST,
        )

    arr = arr.astype(np.float32)

    if normalize_mode == "none":
        pass

    elif normalize_mode == "max_depth":
        arr = np.clip(arr, 0.0, max_depth)
        arr = arr / max_depth

    elif normalize_mode == "image":
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    else:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    return arr


def load_dataset(
    config_path: str,
    target_size: int,
    normalize_mode: str,
    max_depth: float,
    sim_csv_skiprows: int,
    real_csv_skiprows: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load sim and real depth data from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "sim" not in cfg or "real" not in cfg:
        raise KeyError("YAML must contain both 'sim' and 'real' fields.")

    sim_paths = expand_paths(cfg["sim"])
    real_paths = expand_paths(cfg["real"])

    if len(sim_paths) == 0:
        raise RuntimeError("No simulation depth files found.")
    if len(real_paths) == 0:
        raise RuntimeError("No real depth files found.")

    print(f"[INFO] Found {len(sim_paths)} sim depth files.")
    print(f"[INFO] Found {len(real_paths)} real depth files.")

    all_depths = []
    labels = []
    paths = []

    for p in sim_paths:
        arr = load_depth_file(
            p,
            csv_skiprows=sim_csv_skiprows,
        )
        arr = preprocess_depth(
            arr,
            target_size=target_size,
            normalize_mode=normalize_mode,
            max_depth=max_depth,
        )
        all_depths.append(arr)
        labels.append(0)
        paths.append(p)

    for p in real_paths:
        arr = load_depth_file(
            p,
            csv_skiprows=real_csv_skiprows,
        )
        arr = preprocess_depth(
            arr,
            target_size=target_size,
            normalize_mode=normalize_mode,
            max_depth=max_depth,
        )
        all_depths.append(arr)
        labels.append(1)
        paths.append(p)

    X = np.stack(all_depths, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    return X, y, paths


def run_tsne(
    X: np.ndarray,
    pca_dim: int = 50,
    perplexity: float = 30.0,
    random_state: int = 0,
    standardize: bool = False,
) -> np.ndarray:
    """Flatten depth images, optionally standardize, apply PCA and t-SNE."""
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)

    print(f"[INFO] Flattened depth shape: {X_flat.shape}")

    if standardize:
        print("[INFO] Applying StandardScaler.")
        X_flat = StandardScaler().fit_transform(X_flat)

    # PCA dimension should be smaller than number of samples.
    effective_pca_dim = min(pca_dim, num_samples - 1, X_flat.shape[1])
    if effective_pca_dim < 2:
        raise RuntimeError(
            "Too few samples for PCA/t-SNE. Please provide more depth images."
        )

    print(f"[INFO] Applying PCA: {X_flat.shape[1]} -> {effective_pca_dim}")
    X_pca = PCA(
        n_components=effective_pca_dim,
        random_state=random_state,
    ).fit_transform(X_flat)

    # Perplexity must be smaller than number of samples.
    effective_perplexity = min(perplexity, max(2, (num_samples - 1) / 3))
    print(f"[INFO] Applying t-SNE with perplexity={effective_perplexity:.2f}")

    X_tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    ).fit_transform(X_pca)

    return X_tsne


def plot_tsne(
    X_tsne: np.ndarray,
    y: np.ndarray,
    output_path: str,
    title: str,
):
    """Plot and save t-SNE figure."""
    plt.figure(figsize=(7, 6))

    sim_mask = y == 0
    real_mask = y == 1

    plt.scatter(
        X_tsne[sim_mask, 0],
        X_tsne[sim_mask, 1],
        s=12,
        alpha=0.7,
        label="Simulation",
    )
    plt.scatter(
        X_tsne[real_mask, 0],
        X_tsne[real_mask, 1],
        s=12,
        alpha=0.7,
        label="Real",
    )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved t-SNE figure to: {output_path}")


def save_embedding(
    X_tsne: np.ndarray,
    y: np.ndarray,
    paths: List[str],
    output_csv: str,
):
    """Save t-SNE coordinates and metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("path,label,tsne_x,tsne_y\n")
        for i in range(len(paths)):
            label = "sim" if y[i] == 0 else "real"
            f.write(
                f"{paths[i]},{label},{X_tsne[i, 0]:.8f},{X_tsne[i, 1]:.8f}\n"
            )

    print(f"[INFO] Saved embedding csv to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="data_analyze/depth_dataset.yaml",
        help="Path to YAML file containing sim and real depth paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="depth_tsne.png",
        help="Output t-SNE figure path.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="depth_tsne_embedding.csv",
        help="Output CSV path for t-SNE coordinates.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target depth resolution. Default: 256.",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="none",
        choices=["none", "max_depth", "image"],
        help=(
            "Depth normalization mode. "
            "'none' for already normalized [0,1] depth; "
            "'max_depth' for metric depth in meters; "
            "'image' for per-image normalization."
        ),
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=10.0,
        help="Max depth in meters, used only when normalize_mode=max_depth.",
    )
    parser.add_argument(
        "--sim_csv_skiprows",
        type=int,
        default=1,
        help="Rows to skip for simulation CSV/TSV files (e.g., header row).",
    )
    parser.add_argument(
        "--real_csv_skiprows",
        type=int,
        default=0,
        help="Rows to skip for real CSV/TSV files.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=50,
        help="PCA dimension before t-SNE.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply StandardScaler before PCA. Usually not needed for normalized depth.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="t-SNE of Processed Depth Observations",
        help="Figure title.",
    )

    args = parser.parse_args()

    X, y, paths = load_dataset(
        config_path=args.config,
        target_size=args.target_size,
        normalize_mode=args.normalize_mode,
        max_depth=args.max_depth,
        sim_csv_skiprows=args.sim_csv_skiprows,
        real_csv_skiprows=args.real_csv_skiprows,
    )

    print(f"[INFO] Dataset shape: {X.shape}")
    print(f"[INFO] Depth range: min={X.min():.4f}, max={X.max():.4f}")

    X_tsne = run_tsne(
        X,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        random_state=args.random_state,
        standardize=args.standardize,
    )

    plot_tsne(
        X_tsne=X_tsne,
        y=y,
        output_path=args.output,
        title=args.title,
    )

    save_embedding(
        X_tsne=X_tsne,
        y=y,
        paths=paths,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
