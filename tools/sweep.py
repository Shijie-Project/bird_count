"""Cross-platform hyperparameter sweep driver for train.py.

Edit the GRID and COMMON_ARGS dicts below, then run:

    python tools/sweep.py
    python tools/sweep.py --dry-run     # print commands without running

Each combination lands in its own checkpoint subdirectory under SWEEP_ROOT,
plus a tee'd log under SWEEP_ROOT/_logs. Re-running skips combinations that
already produced a best_*.pth, so you can resume an interrupted sweep.

Roll up the results with:
    python tools/summarize_sweep.py ../ckpts/sweep
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


# --- Sweep grid (edit me) ---------------------------------------------------
# Keys are train.py CLI flags WITHOUT the leading "--"; values are lists of
# strings to sweep over.
GRID = {
    "lr": ["1e-5", "5e-5", "1e-4"],
    "waux": ["0.5", "1.0", "2.0"],
    "aux-sigma": ["1.5", "2.0", "3.0"],
}

# --- Fixed args applied to every run (edit me) ------------------------------
SWEEP_ROOT = Path("../ckpts/sweep")
COMMON_ARGS = [
    "--max-epoch",
    "200",
    "--val-epoch",
    "5",
    "--val-start",
    "30",
    "--batch-size",
    "8",
    "--num-workers",
    "4",
    "--crop-size",
    "512",
]


def _tag(combo: dict) -> str:
    """Filesystem-safe tag from a parameter dict, e.g. 'lr1e-5_waux1.0_auxsigma2.0'."""
    return "_".join(f"{k.replace('-', '')}{v}" for k, v in combo.items())


def _has_best(run_dir: Path) -> bool:
    return any(run_dir.rglob("best_*.pth"))


def _stream(cmd: list, log_path: Path) -> int:
    """Run `cmd`, stream output to console and `log_path`. Return the exit code."""
    with open(log_path, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log_f.write(chunk)
        return proc.wait()


def main():
    ap = argparse.ArgumentParser(description="Run a hyperparameter sweep over train.py")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    ap.add_argument(
        "--stop-on-fail", action="store_true", help="abort the sweep on the first failed run (default: keep going)"
    )
    args = ap.parse_args()

    keys = list(GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = SWEEP_ROOT / "_logs"
    log_dir.mkdir(exist_ok=True)

    failures = []
    for i, combo in enumerate(combos, start=1):
        tag = _tag(combo)
        run_dir = SWEEP_ROOT / tag
        log_path = log_dir / f"{tag}.log"

        if _has_best(run_dir):
            print(f"[{i}/{len(combos)}] SKIP  {tag}  (best_*.pth already exists)")
            continue

        cmd = [sys.executable, "train.py", *COMMON_ARGS]
        for k, v in combo.items():
            cmd += [f"--{k}", v]
        cmd += ["--checkpoint-dir", str(run_dir)]

        print(f"[{i}/{len(combos)}] RUN   {tag}")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue

        rc = _stream(cmd, log_path)
        if rc != 0:
            failures.append((tag, rc, log_path))
            print(f"  -> FAILED (exit {rc}); log at {log_path}")
            if args.stop_on_fail:
                sys.exit(rc)

    print()
    if failures:
        print(f"Sweep finished with {len(failures)} failure(s):")
        for tag, rc, log_path in failures:
            print(f"  - {tag} (exit {rc}, see {log_path})")
    else:
        print("Sweep complete (all runs succeeded or were skipped).")
    print()
    print("Summarize results with:")
    print(f"  python tools/summarize_sweep.py {SWEEP_ROOT}")


if __name__ == "__main__":
    main()
