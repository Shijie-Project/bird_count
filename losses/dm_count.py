"""DM-Count loss orchestrator with optional Gaussian-density auxiliary term.

Combines the three DM-Count terms (OT + global count + normalized-L1 "TV") and,
optionally, a per-pixel L1 against a Gaussian-smoothed GT to provide dense
supervision in empty regions.

Returns (total_loss, parts_dict) where parts_dict holds the (already-weighted)
scalar contribution of each term plus the diagnostic Wasserstein distance and
OT objective value.
"""

import torch
import torch.nn as nn

from .density import DensityAuxLoss
from .ot import OT_Loss


class DMCountLoss(nn.Module):
    def __init__(
        self,
        c_size,
        stride,
        norm_cood,
        device,
        wot=0.1,
        wtv=0.01,
        wcount=1.0,
        waux=1.0,
        aux_sigma=2.0,
        dense_weight_alpha=0.0,
        num_of_iter_in_ot=100,
        reg=10.0,
    ):
        super().__init__()
        self.ot = OT_Loss(c_size, stride, norm_cood, device, num_of_iter_in_ot, reg)
        self._tv_l1 = nn.L1Loss(reduction="none")
        self._mae = nn.L1Loss()
        self.aux = DensityAuxLoss(aux_sigma, dense_weight_alpha).to(device) if aux_sigma > 0 else None
        self.wot = wot
        self.wtv = wtv
        self.wcount = wcount
        self.waux = waux

    def forward(self, outputs, gt_discrete, points):
        """
        outputs:     (B, 1, H', W') raw model output (non-negative).
        gt_discrete: (B, 1, H', W') sum-pooled discrete count target.
        points:      list of (N_i, 2) keypoint tensors per sample.
        """
        N = outputs.size(0)
        flat_sum = outputs.view(N, -1).sum(dim=1)
        outputs_normed = outputs / (flat_sum.view(N, 1, 1, 1) + 1e-6)

        # OT term — DM-Count distribution match via Sinkhorn. OT_Loss accumulates
        # per-sample contributions, so divide by N to make it batch-size invariant
        # (so `wot` doesn't need rescaling when batch size changes). The other three
        # terms below are already mean-per-batch.
        ot_loss, wd, ot_obj = self.ot(outputs_normed, outputs, points)
        ot_loss = ot_loss / N
        ot_obj = ot_obj / N
        wd = wd / N

        # Global count L1.
        gd_count = torch.tensor([len(p) for p in points], device=outputs.device, dtype=torch.float32)
        count_loss = self._mae(flat_sum, gd_count)

        # Normalized-L1 distribution match (the DM-Count "TV" term).
        gd_count_t = gd_count.view(N, 1, 1, 1)
        gt_normed = gt_discrete / (gd_count_t + 1e-6)
        tv_per_sample = self._tv_l1(outputs_normed, gt_normed).sum(dim=(1, 2, 3))
        tv_loss = (tv_per_sample * gd_count).mean()

        # Optional per-pixel auxiliary against Gaussian-smoothed GT.
        if self.aux is not None:
            aux_loss = self.aux(outputs, gt_discrete)
        else:
            aux_loss = outputs.new_zeros(())

        total = self.wot * ot_loss + self.wcount * count_loss + self.wtv * tv_loss + self.waux * aux_loss

        parts = {
            "ot": float(ot_loss.detach()) * self.wot,
            "count": float(count_loss.detach()) * self.wcount,
            "tv": float(tv_loss.detach()) * self.wtv,
            "aux": float(aux_loss.detach()) * self.waux,
            "wd": wd,
            "ot_obj": float(ot_obj.detach()) * self.wot,
        }
        return total, parts
