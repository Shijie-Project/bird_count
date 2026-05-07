"""Evaluate the trained density model on a dataset split.

Reports both audience-friendly summaries (for exhibition) and standard
counting metrics (for performance evaluation). Optionally writes per-image
density-overlay PNGs and a metrics JSON report.
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Optional

import cv2
import dotenv
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import collate
from datasets.bird import Bird
from metrics import (
    CountingMetrics,
    PileupClassificationMetrics,
    compute_metrics,
    compute_pileup_classification,
    compute_stratified,
    fraction_within,
)
from models.shufflenet import get_shufflenet_density_model
from utils import set_seed


dotenv.load_dotenv()
warnings.simplefilter("ignore", UserWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate density model on a dataset split")

    g = p.add_argument_group("data")
    g.add_argument("--data-path", default="../data", help="dataset root")
    g.add_argument("--split", default="val", choices=["val", "train", "val_legacy", "train_legacy"])
    g.add_argument(
        "--crop-size", type=int, default=512, help="train crop size (forwarded to Bird; only matters for asserts)"
    )
    g.add_argument("--num-workers", type=int, default=2)

    g = p.add_argument_group("model")
    g.add_argument("--ckpt", default=None, help="checkpoint path; defaults to MODEL_PATH from .env")
    g.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")
    g.add_argument(
        "--no-fuse", action="store_true", help="skip Conv+BN fusion (debug only; fusion is mathematically equivalent)"
    )

    g = p.add_argument_group("metrics")
    g.add_argument(
        "--pileup-threshold",
        type=float,
        default=100.0,
        help="count threshold above which an image is considered a pile-up event",
    )
    g.add_argument("--abs-tol", type=float, default=5.0, help="absolute tolerance (chickens) for the '+/- N' headline")
    g.add_argument(
        "--rel-tol",
        type=float,
        default=0.10,
        help="relative tolerance (e.g. 0.10 = 10%) for the 'within tolerance' headline",
    )

    g = p.add_argument_group("output")
    g.add_argument("--no-density-map", action="store_true", help="skip per-image overlay PNGs")
    g.add_argument("--metrics-out", default=None, help="optional JSON file to write the full metrics report into")
    g.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _resolve_checkpoint(args) -> str:
    path = args.ckpt or os.getenv("MODEL_PATH")
    if not path:
        raise SystemExit("No checkpoint specified. Pass --ckpt or set MODEL_PATH in .env.")
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    return path


def _save_overlay(img_path: str, pred_map: np.ndarray, gt_count: float, pred_count: float, out_path: Path) -> None:
    """Save the source image with the predicted heatmap blended on top + count label."""
    original = cv2.imread(img_path)
    if original is None:
        return  # source image not reachable from this machine; skip silently
    h, w = original.shape[:2]

    vmin, vmax = float(pred_map.min()), float(pred_map.max())
    normed = (pred_map - vmin) / (vmax - vmin + 1e-5)
    normed = cv2.resize(normed, (w, h))
    heatmap = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = original.copy()
    mask = normed > 0.01
    if mask.any():
        overlay[mask] = cv2.addWeighted(original[mask], 0.5, heatmap[mask], 0.5, 0)

    err = pred_count - gt_count
    cv2.putText(
        overlay,
        f"GT: {gt_count:.1f}  Pred: {pred_count:.1f}  Err: {err:+.1f}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), overlay)


@torch.inference_mode()
def run_eval(model, device, loader, out_dir: Optional[Path]):
    preds, gts = [], []
    for sample in loader:
        inputs = sample["image"].to(device, non_blocking=True).float()
        gt_count = sample["density"].sum().item()
        outputs = model(inputs)
        pred_count = outputs.sum().item()
        err = pred_count - gt_count

        name = sample["name"][0]
        path = sample["path"][0]
        preds.append(pred_count)
        gts.append(gt_count)
        print(f"  {name}: GT {gt_count:7.1f}  Pred {pred_count:7.1f}  Err {err:+7.1f}")

        if out_dir is not None:
            _save_overlay(path, outputs[0, 0].cpu().numpy(), gt_count, pred_count, out_dir / f"{name}_density.png")

    return np.asarray(preds), np.asarray(gts)


def _print_exhibition_summary(
    metrics: CountingMetrics,
    pileup: PileupClassificationMetrics,
    frac_abs: float,
    frac_rel: float,
    abs_tol: float,
    rel_tol: float,
) -> None:
    bias_word = "over-counts" if metrics.bias > 0 else "under-counts"
    bias_amount = abs(metrics.bias)

    print()
    print("=" * 64)
    print("                   EXHIBITION SUMMARY")
    print("=" * 64)
    print(f"  Images analyzed         : {metrics.n_images:,}")
    print(f"  Total chickens (GT)     : {metrics.total_gt:,.0f}")
    print(f"  Average miscount        : {metrics.mae:.1f} chickens per image")
    print(f"  Within +/-{abs_tol:g} chickens     : {frac_abs * 100:5.1f}% of images")
    print(f"  Within +/-{rel_tol * 100:g}% of true count : {frac_rel * 100:5.1f}% of images")
    print(f"  Best image              : off by {metrics.best_abs_error:.2f}")
    print(f"  Worst image             : off by {metrics.worst_abs_error:.2f}")
    print(f"  System bias             : {bias_word} by {bias_amount:.2f} on average")

    print()
    actual = pileup.tp + pileup.fn
    print(f"  Pile-ups in dataset     : {actual} (events with > {pileup.threshold:g} chickens)")
    if actual > 0:
        caught_pct = pileup.tp / actual * 100
        print(f"  Pile-ups detected       : {pileup.tp} of {actual} ({caught_pct:.0f}%)")
    print(f"  False alarms            : {pileup.fp}")
    print("=" * 64)


def _print_technical_metrics(metrics: CountingMetrics, stratified, pileup: PileupClassificationMetrics) -> None:
    print()
    print("=" * 64)
    print("                   TECHNICAL METRICS")
    print("=" * 64)
    print(f"  N images       : {metrics.n_images}")
    print(f"  Total GT count : {metrics.total_gt:.1f}")
    print(f"  MAE            : {metrics.mae:.4f}")
    print(f"  RMSE           : {metrics.rmse:.4f}")
    print(f"  NAE (MAE/mean) : {metrics.nae:.4f}")
    print(f"  MAPE           : {metrics.mape:.2f} %")
    print(f"  Bias (signed)  : {metrics.bias:+.4f}")
    print(f"  R^2            : {metrics.r2:.4f}")
    print(f"  Pearson r      : {metrics.pearson:.4f}")
    print(f"  |Error| best   : {metrics.best_abs_error:.2f}")
    print(f"  |Error| worst  : {metrics.worst_abs_error:.2f}")

    print()
    print("  Stratified MAE by GT count band:")
    print(f"    {'Band':<22} {'N':>5} {'MAE':>10} {'MAPE':>10}")
    for s in stratified:
        mape_str = f"{s.mape:.1f} %" if s.mape is not None else "  n/a"
        mae_str = f"{s.mae:.3f}" if not math.isnan(s.mae) else "n/a"
        print(f"    {s.band:<22} {s.n_images:>5} {mae_str:>10} {mape_str:>10}")

    print()
    print(f"  Pile-up detection (threshold = {pileup.threshold:g}):")
    print(f"    TP {pileup.tp:>4}   FP {pileup.fp:>4}   FN {pileup.fn:>4}   TN {pileup.tn:>4}")
    print(f"    Precision : {pileup.precision:.4f}")
    print(f"    Recall    : {pileup.recall:.4f}")
    print(f"    F1        : {pileup.f1:.4f}")
    print(f"    Accuracy  : {pileup.accuracy:.4f}")
    print("=" * 64)


def _scrub_nans(obj):
    """Recursively replace NaN/inf with None so the result is strict JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _scrub_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nans(v) for v in obj]
    return obj


def _write_metrics_json(
    path: Path,
    metrics: CountingMetrics,
    stratified,
    pileup: PileupClassificationMetrics,
    frac_abs: float,
    frac_rel: float,
    abs_tol: float,
    rel_tol: float,
) -> None:
    payload = {
        "overall": metrics.to_dict(),
        "fraction_within_abs": {"tol_chickens": abs_tol, "fraction": frac_abs},
        "fraction_within_rel": {"tol_fraction": rel_tol, "fraction": frac_rel},
        "stratified": [s.to_dict() for s in stratified],
        "pileup_detection": pileup.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_scrub_nans(payload), indent=2))
    print(f"\nMetrics report written to: {path}")


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.strip())
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint(args)
    model = get_shufflenet_density_model(model_path=ckpt_path, device=device, fuse=not args.no_fuse)

    out_dir: Optional[Path] = None
    if not args.no_density_map:
        out_dir = Path(ckpt_path).parent / "density_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing density overlays to: {out_dir}")

    dataset = Bird(args.data_path, args.crop_size, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    print(f"\nEvaluating on '{args.split}' split ({len(dataset)} images)")
    print("-" * 64)
    preds, gts = run_eval(model, device, loader, out_dir)
    print("-" * 64)

    metrics = compute_metrics(preds, gts)
    stratified = compute_stratified(preds, gts)
    pileup = compute_pileup_classification(preds, gts, args.pileup_threshold)
    frac_abs = fraction_within(preds, gts, abs_tol=args.abs_tol)
    frac_rel = fraction_within(preds, gts, rel_tol=args.rel_tol)

    _print_exhibition_summary(metrics, pileup, frac_abs, frac_rel, args.abs_tol, args.rel_tol)
    _print_technical_metrics(metrics, stratified, pileup)

    if args.metrics_out:
        _write_metrics_json(
            Path(args.metrics_out),
            metrics,
            stratified,
            pileup,
            frac_abs,
            frac_rel,
            args.abs_tol,
            args.rel_tol,
        )


if __name__ == "__main__":
    main()
