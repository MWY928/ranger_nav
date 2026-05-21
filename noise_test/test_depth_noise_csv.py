import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_noise import add_realistic_depth_noise


def _load_depth_csv(
    path: Path, delimiter: str = ",", encoding: str = "auto"
) -> np.ndarray:
    encodings = (
        [encoding]
        if encoding != "auto"
        else ["utf-8-sig", "utf-8", "gbk", "latin-1"]
    )
    last_error = None
    for enc in encodings:
        for skiprows in (0, 1):
            try:
                arr = np.loadtxt(
                    path,
                    delimiter=delimiter,
                    dtype=np.float32,
                    skiprows=skiprows,
                    encoding=enc,
                )
                if arr.ndim != 2:
                    raise ValueError(f"Expected 2D depth CSV, got shape={arr.shape}")
                return arr
            except (ValueError, UnicodeDecodeError) as e:
                last_error = e
                continue
    raise RuntimeError(
        f"Failed to load depth CSV: {path}. "
        f"Tried encodings={encodings}, skiprows=(0,1). Last error: {last_error}"
    )


def _stats(name: str, depth: np.ndarray, max_depth: float) -> str:
    finite = np.isfinite(depth)
    if not finite.any():
        return f"{name}: all values are non-finite"

    d = depth.astype(np.float32, copy=False)
    valid = finite & (d > 0.0) & (d < max_depth)

    msg = [
        f"[{name}] shape={d.shape}, dtype={d.dtype}",
        f"  min={np.nanmin(d):.6f}, max={np.nanmax(d):.6f}, mean={np.nanmean(d):.6f}",
        f"  finite_ratio={finite.mean():.4f}, valid_ratio={valid.mean():.4f}, zero_ratio={(d == 0).mean():.4f}",
    ]
    if valid.any():
        vals = d[valid]
        msg.append(
            f"  valid_p50={np.percentile(vals, 50):.6f}, valid_p95={np.percentile(vals, 95):.6f}"
        )
    return "\n".join(msg)


def _to_color(depth: np.ndarray, max_depth: float) -> np.ndarray:
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=max_depth, neginf=0.0)
    d = np.clip(d, 0.0, max_depth)
    d_u8 = (d / max_depth * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply realistic depth noise to a depth CSV")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to depth CSV (meters)")
    parser.add_argument("--output_dir", type=str, default="noise_test/output", help="Output directory")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    parser.add_argument(
        "--encoding",
        type=str,
        default="auto",
        help="CSV encoding, e.g. auto/utf-8-sig/utf-8/gbk/latin-1",
    )

    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--dropout_base", type=float, default=0.03)
    parser.add_argument("--dropout_far", type=float, default=0.1)
    parser.add_argument("--corr_scale", type=int, default=8)
    parser.add_argument("--edge_dropout", type=float, default=0.2)
    parser.add_argument("--edge_threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    np.random.seed(args.seed)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = _load_depth_csv(
        input_csv, delimiter=args.delimiter, encoding=args.encoding
    )
    depth = np.nan_to_num(depth, nan=0.0, posinf=args.max_depth, neginf=0.0).astype(np.float32)

    depth_noisy = add_realistic_depth_noise(
        depth,
        max_depth=args.max_depth,
        alpha=args.alpha,
        dropout_base=args.dropout_base,
        dropout_far=args.dropout_far,
        corr_scale=args.corr_scale,
        edge_dropout=args.edge_dropout,
        edge_threshold=args.edge_threshold,
    )

    stem = input_csv.stem
    noisy_csv = output_dir / f"{stem}_noisy.csv"
    noisy_npy = output_dir / f"{stem}_noisy.npy"
    raw_npy = output_dir / f"{stem}_raw.npy"
    vis_raw = output_dir / f"{stem}_raw_color.png"
    vis_noisy = output_dir / f"{stem}_noisy_color.png"
    vis_compare = output_dir / f"{stem}_compare.png"

    np.savetxt(noisy_csv, depth_noisy, delimiter=",", fmt="%.6f")
    np.save(noisy_npy, depth_noisy)
    np.save(raw_npy, depth)

    raw_color = _to_color(depth, args.max_depth)
    noisy_color = _to_color(depth_noisy, args.max_depth)
    cv2.imwrite(str(vis_raw), raw_color)
    cv2.imwrite(str(vis_noisy), noisy_color)

    compare = cv2.hconcat([raw_color, noisy_color])
    cv2.imwrite(str(vis_compare), compare)

    print(_stats("RAW", depth, args.max_depth))
    print(_stats("NOISY", depth_noisy, args.max_depth))
    print("\nSaved files:")
    print(f"  {noisy_csv}")
    print(f"  {noisy_npy}")
    print(f"  {raw_npy}")
    print(f"  {vis_raw}")
    print(f"  {vis_noisy}")
    print(f"  {vis_compare}")


if __name__ == "__main__":
    main()
