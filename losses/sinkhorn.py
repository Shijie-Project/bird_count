"""Sinkhorn-Knopp solver for entropic-regularized optimal transport.

Pared-down PyTorch port of `ot.bregman.sinkhorn` from POT
(https://pythonot.github.io). Only the basic Sinkhorn-Knopp variant is kept
since DM-Count uses small problems where log-stabilization is unnecessary.

Reference: M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal
Transport", NeurIPS 2013.
"""

import logging

import torch


_LOG = logging.getLogger(__name__)
M_EPS = 1e-16


def sinkhorn(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9, log=True, **_):
    """Solve min_gamma <gamma, C> + reg * H(gamma) s.t. row/col marginals = a, b.

    Args:
        a: (na,) target marginal (sums to 1).
        b: (nb,) source marginal (sums to 1).
        C: (na, nb) cost matrix.
        reg: entropic regularization strength.
        maxIter: max Sinkhorn iterations.
        stopThr: stop when (b - K^T u v) MSE < stopThr.
        log: if True, return (P, dict) with `u`, `v`, `alpha`, `beta`, `err`.

    Returns:
        P: (na, nb) transport plan, or (P, log_dict) if `log=True`.
    """
    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, "C must be 2-D"
    assert na == a.shape[0] and nb == b.shape[0], "shape of a/b must match C"
    assert reg > 0, "reg must be > 0"
    assert a.min() >= 0.0 and b.min() >= 0.0, "a/b must be non-negative"

    log_dict = {"err": []} if log else None

    u = torch.full((na,), 1.0 / na, dtype=a.dtype, device=device)
    v = torch.full((nb,), 1.0 / nb, dtype=b.dtype, device=device)

    K = torch.exp(C / -reg)

    err = float("inf")
    it = 0
    while err > stopThr and it < maxIter:
        upre, vpre = u, v
        v = b / (torch.matmul(u, K) + M_EPS)
        u = a / (torch.matmul(K, v) + M_EPS)

        if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
            _LOG.warning("sinkhorn: numerical instability at iter %d, rolling back", it)
            u, v = upre, vpre
            break

        if log and (it % 10 == 0):
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            log_dict["err"].append(err)

        it += 1

    if log:
        log_dict["u"] = u
        log_dict["v"] = v
        log_dict["alpha"] = reg * torch.log(u + M_EPS)
        log_dict["beta"] = reg * torch.log(v + M_EPS)

    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return (P, log_dict) if log else P
