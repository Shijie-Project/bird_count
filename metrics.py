"""Evaluation metrics for crowd-counting density models.

Two kinds of consumers in mind:
- **Standard ML metrics** (MAE, RMSE, NAE, MAPE, R², Pearson, Bias) for paper /
  technical reporting.
- **Audience-friendly summaries** for exhibition: how often we're within a
  reasonable tolerance, how well pile-ups are caught at a chosen threshold.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class CountingMetrics:
    """Aggregate count-regression metrics over a dataset."""

    n_images: int
    total_gt: float
    mae: float  # mean absolute error
    rmse: float  # root mean squared error
    nae: float  # MAE normalized by mean GT
    bias: float  # mean signed error (pred - gt); + = over-counts
    mape: float  # mean absolute percentage error (skips GT=0); %
    r2: float  # coefficient of determination
    pearson: float  # Pearson correlation between pred and gt
    worst_abs_error: float
    best_abs_error: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StratifiedMetrics:
    """MAE / MAPE within a single count band, e.g. 'Low (1-10)'."""

    band: str
    count_range: tuple[float, float]
    n_images: int
    mae: float
    mape: Optional[float]  # None if no images in this band have GT > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["count_range"] = list(d["count_range"])
        return d


@dataclass
class PileupClassificationMetrics:
    """Threshold-based classification: pile-up = (count > threshold)."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else float("nan")

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


DEFAULT_BANDS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("Empty", (0, 0)),
    ("Low (1-10)", (1, 10)),
    ("Medium (11-50)", (11, 50)),
    ("High (51-100)", (51, 100)),
    ("Pile-up (>100)", (101, float("inf"))),
)


def compute_metrics(preds: Sequence[float], gts: Sequence[float]) -> CountingMetrics:
    """Compute standard counting metrics from per-image predictions and GTs."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    if len(p) != len(g) or len(p) == 0:
        raise ValueError("preds and gts must be non-empty and equal-length")

    err = p - g
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err**2).mean()))
    mean_gt = float(g.mean())
    nae = mae / mean_gt if mean_gt > 0 else float("nan")
    bias = float(err.mean())

    nonzero = g > 0
    if nonzero.any():
        mape = float((abs_err[nonzero] / g[nonzero]).mean() * 100)
    else:
        mape = float("nan")

    ss_res = float(((p - g) ** 2).sum())
    ss_tot = float(((g - mean_gt) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if g.std() > 0 and p.std() > 0:
        pearson = float(np.corrcoef(p, g)[0, 1])
    else:
        pearson = float("nan")

    return CountingMetrics(
        n_images=len(p),
        total_gt=float(g.sum()),
        mae=mae,
        rmse=rmse,
        nae=nae,
        bias=bias,
        mape=mape,
        r2=r2,
        pearson=pearson,
        worst_abs_error=float(abs_err.max()),
        best_abs_error=float(abs_err.min()),
    )


def compute_stratified(
    preds: Sequence[float],
    gts: Sequence[float],
    bands: Sequence[tuple[str, tuple[float, float]]] = DEFAULT_BANDS,
) -> list:
    """Per-band MAE / MAPE so we can see where errors concentrate."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    out = []
    for label, (lo, hi) in bands:
        mask = (g >= lo) & (g <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append(StratifiedMetrics(label, (lo, hi), 0, float("nan"), None))
            continue
        ae = np.abs(p[mask] - g[mask])
        mae = float(ae.mean())
        nz = g[mask] > 0
        mape = float((ae[nz] / g[mask][nz]).mean() * 100) if nz.any() else None
        out.append(StratifiedMetrics(label, (lo, hi), n, mae, mape))
    return out


def compute_pileup_classification(
    preds: Sequence[float], gts: Sequence[float], threshold: float
) -> PileupClassificationMetrics:
    """Treat each image as positive iff count > threshold; compute confusion stats."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    pred_pos = p > threshold
    gt_pos = g > threshold
    return PileupClassificationMetrics(
        threshold=float(threshold),
        tp=int((pred_pos & gt_pos).sum()),
        fp=int((pred_pos & ~gt_pos).sum()),
        tn=int((~pred_pos & ~gt_pos).sum()),
        fn=int((~pred_pos & gt_pos).sum()),
    )


def fraction_within(
    preds: Sequence[float],
    gts: Sequence[float],
    *,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> float:
    """Fraction of images where |pred - gt| <= max(abs_tol, rel_tol * gt).

    Either tolerance may be None to disable that mode. If both are provided,
    the looser tolerance per-image wins.
    """
    if abs_tol is None and rel_tol is None:
        raise ValueError("Pass at least one of abs_tol or rel_tol")
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    err = np.abs(p - g)
    tol = np.zeros_like(err)
    if abs_tol is not None:
        tol = np.maximum(tol, float(abs_tol))
    if rel_tol is not None:
        tol = np.maximum(tol, float(rel_tol) * g)
    return float((err <= tol).mean())
