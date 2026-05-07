"""Summarize a sweep directory: scan for best_ep*_mae*_mse*.pth files and
print a table sorted by MAE.

The trainer encodes the best epoch + MAE + MSE in the saved filename
(`best_ep0042_mae5.32_mse8.71.pth`), so we can build a leaderboard without
reading any checkpoints. The parent directory carries the hyperparameters
(`input-{cs}_wot-{x}_wtv-{y}_waux-{z}_reg-{r}_nIter-{n}_normCood-{c}/{ts}`).

Usage:
    python -m tools.summarize_sweep ../ckpts/sweep [--top 10]
"""

import argparse
import re
from pathlib import Path
from typing import Optional


_BEST_PATTERN = re.compile(r"best_ep(?P<epoch>\d+)_mae(?P<mae>[\d.]+)_mse(?P<mse>[\d.]+)\.pth$")


def _parse_best(path: Path) -> Optional[dict]:
    m = _BEST_PATTERN.match(path.name)
    if not m:
        return None
    return {
        "epoch": int(m.group("epoch")),
        "mae": float(m.group("mae")),
        "mse": float(m.group("mse")),
        "path": path,
    }


def _run_label(path: Path, sweep_root: Path) -> str:
    """Best-effort short label for a run directory."""
    rel = path.relative_to(sweep_root).parent  # strip the .pth filename
    parts = rel.parts
    # Trainer writes <sweep-tag>/<config-stem>/<timestamp>/best_*.pth.
    # Drop the timestamp and join the rest. If the structure is shallower,
    # fall back to the raw relpath.
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return str(rel)


def collect(sweep_root: Path) -> list:
    runs = []
    for ckpt in sweep_root.rglob("best_*.pth"):
        info = _parse_best(ckpt)
        if info is None:
            continue
        info["label"] = _run_label(ckpt, sweep_root)
        runs.append(info)
    runs.sort(key=lambda r: (r["mae"], r["mse"]))
    return runs


def print_table(runs: list, top: Optional[int] = None) -> None:
    if not runs:
        print("No best_*.pth files found.")
        return
    if top is not None:
        runs = runs[:top]

    label_w = max(len(r["label"]) for r in runs)
    label_w = max(label_w, len("run"))
    print()
    print(f"{'rank':>4}  {'label':<{label_w}}  {'epoch':>5}  {'MAE':>8}  {'MSE':>8}")
    print(f"{'-' * 4}  {'-' * label_w}  {'-' * 5}  {'-' * 8}  {'-' * 8}")
    for i, r in enumerate(runs, start=1):
        print(f"{i:>4}  {r['label']:<{label_w}}  {r['epoch']:>5}  {r['mae']:>8.3f}  {r['mse']:>8.3f}")
    print()
    print(f"Total runs: {len(runs)}")


def main():
    p = argparse.ArgumentParser(description="Summarize a sweep directory")
    p.add_argument("sweep_root", type=Path, help="root directory of the sweep")
    p.add_argument("--top", type=int, default=None, help="show only the top-K runs by MAE")
    args = p.parse_args()

    if not args.sweep_root.exists():
        raise SystemExit(f"Sweep root does not exist: {args.sweep_root}")

    runs = collect(args.sweep_root)
    print_table(runs, top=args.top)


if __name__ == "__main__":
    main()
