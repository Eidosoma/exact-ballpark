# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""GetCenterDist and GetSurfaceDist batched (the helpers of CMB-701): the Python
`PyGetCenterDist` and `PyGetSurfaceDist` (Thunkers.cpp:1774, 1798, Carbon's `destiny` at 114e89fa, MIT, (c) CCP Games)
for many pairs of rows at once, on any batched park's tensors.

Destiny returns `(ball1->mNewPos - ball2->mNewPos).Length()` and that length `- ball1->mRadius - ball2->mRadius`, or
None when either ball is missing. Here: the position difference component by component, its components squared and
summed left to right, the square root (Vector3d::Length); then the centre distance less a's radius, less b's, in that
order (the radii are floats, held in float64 by the parks, so the promotion to double is exact). Only correctly
rounded IEEE operations in the reference's order (`exact_ballpark.full.FullBallpark.center_dist`, itself held to
Destiny's compiled code by tests/test_movement_full_oracle.py and tests/test_movement_vec_modes_leftovers.py), so the
result is the reference's bit for bit on the CPU and on a CUDA device (tests/test_movement_vec_modes_leftovers.py). No
host synchronisation: a CUDA graph can capture it.

A pair with a missing ball (a row of -1, or past the park's rows) is not found: its distances are NaN and `found` is
False, where Destiny returns None. The caller maps Destiny's ids to rows (VecCombatPark.lookup does it by id).
"""

from __future__ import annotations

import torch

F64, I64 = torch.float64, torch.int64


def center_surface_dist(pos: torch.Tensor, radius: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple:
    """(centre [M], surface [M], found [M]) for the M pairs of rows (a[k], b[k]) of a park whose positions are pos
    [N, 3] (float64) and radii radius [N] (floats held in float64)."""
    n = pos.shape[0]
    a, b = a.to(device=pos.device, dtype=I64), b.to(device=pos.device, dtype=I64)
    found = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    sa, sb = torch.where(found, a, 0), torch.where(found, b, 0)
    d = pos[sa] - pos[sb]
    center = torch.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
    surface = center - radius[sa] - radius[sb]
    nan = torch.full_like(center, float("nan"))
    return torch.where(found, center, nan), torch.where(found, surface, nan), found


def park_dist(park, a: torch.Tensor, b: torch.Tensor) -> tuple:
    """center_surface_dist on a park's own `pos` and `radius` (VecOrientPark and its subclasses: VecModesPark,
    VecCombatPark, VecComposedPark): the rows a and b. A park that keeps them under other names passes its tensors to
    center_surface_dist."""
    return center_surface_dist(park.pos, park.radius, a, b)
