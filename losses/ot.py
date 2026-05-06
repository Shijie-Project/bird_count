"""Optimal-transport loss for crowd counting (DM-Count, Wang et al., NeurIPS 2020).

Computes per-image OT between the predicted (normalized) density map and the
empirical distribution over annotated keypoints. Sinkhorn solves for the dual
potential `beta`, then DM-Count's analytical gradient
    d(OT)/d(unnormed_density) = beta / sum(unnormed) - <beta, unnormed> / sum(unnormed)^2
is applied via a surrogate `loss = <unnormed_density, im_grad.detach()>`.
"""

import torch
from torch.nn import Module

from .sinkhorn import sinkhorn


class OT_Loss(Module):
    def __init__(self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super().__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # Pixel centers of each density-map cell, in input-image coordinates.
        cood = torch.arange(0, c_size, step=stride, dtype=torch.float32, device=device) + stride / 2
        if norm_cood:
            cood = cood / c_size * 2 - 1  # map to [-1, 1]
        self.cood = cood.unsqueeze(0)  # (1, #cood)
        self.output_size = self.cood.size(1)

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)

        loss = torch.zeros((), device=self.device)
        ot_obj_values = torch.zeros((), device=self.device)
        wd = 0.0

        for idx, im_points in enumerate(points):
            if len(im_points) == 0:
                continue

            if self.norm_cood:
                im_points = im_points / self.c_size * 2 - 1
            x = im_points[:, 0:1]
            y = im_points[:, 1:2]
            x_dis = (x - self.cood).pow(2)  # (#gt, #cood)
            y_dis = (y - self.cood).pow(2)  # (#gt, #cood)
            dis = y_dis.unsqueeze(2) + x_dis.unsqueeze(1)
            dis = dis.view(dis.size(0), -1)  # (#gt, #cood^2)

            source_prob = normed_density[idx][0].view(-1).detach()
            target_prob = torch.full((len(im_points),), 1.0 / len(im_points), device=self.device)

            P, log = sinkhorn(
                target_prob,
                source_prob,
                dis,
                self.reg,
                maxIter=self.num_of_iter_in_ot,
                log=True,
            )
            beta = log["beta"]  # (#cood^2,)
            ot_obj_values = (
                ot_obj_values + (normed_density[idx] * beta.view(1, self.output_size, self.output_size)).sum()
            )

            # Analytical gradient (DM-Count eq.) wrapped in a surrogate.
            source_density = unnormed_density[idx][0].view(-1).detach()
            source_count = source_density.sum()
            denom = source_count * source_count + 1e-8
            im_grad = beta / source_count.clamp_min(1e-8) - (source_density * beta).sum() / denom
            im_grad = im_grad.detach().view(1, self.output_size, self.output_size)
            loss = loss + (unnormed_density[idx] * im_grad).sum()
            wd += float((dis * P).sum().item())

        return loss, wd, ot_obj_values
