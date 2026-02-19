from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import torch

import config

# Grid channels:
#   0: occupancy (0 empty, 1 wall, 2 red, 3 blue)
#   1: hp        (0..MAX_HP)
#   2: agent_id  (-1 if empty)

@dataclass
class Zones:
    """
    Immutable masks for special tiles (kept off-grid to avoid renderer/engine churn).
    Shapes: [H, W] torch.bool on config.TORCH_DEVICE.
    """
    heal_mask: torch.Tensor                  # True where tiles heal HP
    cp_masks: List[torch.Tensor]             # list of boolean masks for capture patches

    @property
    def cp_count(self) -> int:
        return len(self.cp_masks)


# --------------------------------------------------------------
# Random thin gray walls (1-cell thick, meandering segments)
# --------------------------------------------------------------
@torch.no_grad()
def add_random_walls(
    grid: torch.Tensor,
    n_segments: int = config.RANDOM_WALLS,
    seg_min: int = config.WALL_SEG_MIN,
    seg_max: int = config.WALL_SEG_MAX,
    avoid_margin: int = config.WALL_AVOID_MARGIN,
    allow_over_agents: bool = False,
) -> None:
    """
    Carve n_segments 1-cell-thick wall traces into grid[0] by writing 1.0 to occupancy.
    Designed to be called BEFORE spawning agents. If called after spawn and
    allow_over_agents=False, we skip cells that are occupied by agents (2/3).
    """
    assert grid.ndim == 3 and grid.size(0) >= 3, "grid must be (3,H,W)"
    occ = grid[0]
    H, W = int(occ.size(0)), int(occ.size(1))

    # 8-connected step vectors (dx, dy)
    dirs8 = torch.tensor(
        [[ 0, -1],[ 1, -1],[ 1,  0],[ 1,  1],[ 0,  1],[-1,  1],[-1,  0],[-1, -1]],
        dtype=torch.long, device=occ.device
    )

    def _place_wall_cell(x: int, y: int) -> None:
        if 0 <= x < W and 0 <= y < H:
            if not allow_over_agents:
                v = float(occ[y, x].item())
                if v in (2.0, 3.0):  # skip unit cells
                    return
            occ[y, x] = 1.0
            grid[1, y, x] = 0.0
            grid[2, y, x] = -1.0

    # Bounds for starts (respect avoid_margin; outer border already walled by grid maker)
    x0_min, x0_max = max(1, avoid_margin), W - max(1, avoid_margin) - 1
    y0_min, y0_max = max(1, avoid_margin), H - max(1, avoid_margin) - 1

    if x0_min >= x0_max or y0_min >= y0_max or n_segments <= 0:
        return

    for _ in range(max(0, int(n_segments))):
        x = random.randint(x0_min, x0_max)
        y = random.randint(y0_min, y0_max)
        L = random.randint(max(1, int(seg_min)), max(1, int(seg_max)))
        # start drawing
        _place_wall_cell(x, y)
        last_dir = random.randrange(8)

        for _step in range(L):
            # small turn bias to keep lines meandering, not jittering
            if random.random() < 0.70:
                d = last_dir
            else:
                d = (last_dir + random.choice([-2, -1, 1, 2])) % 8
            last_dir = d
            dx, dy = int(dirs8[d, 0].item()), int(dirs8[d, 1].item())

            # step and clamp inside interior
            x = max(1, min(W - 2, x + dx))
            y = max(1, min(H - 2, y + dy))

            # guarantee 1-cell thickness (avoid 2x2 solid blocks)
            _place_wall_cell(x, y)
            # optionally punch a gap occasionally to reduce partition risk
            if random.random() < 0.05:
                # leave a deliberate gap (do nothing)
                pass



# --------------------------------------------------------------
# Heal & Capture zones (rectangular patches, scaled to grid)
# --------------------------------------------------------------
@torch.no_grad()
def make_zones(
    H: int,
    W: int,
    *,
    heal_count: int = config.HEAL_ZONE_COUNT,
    heal_ratio: float = config.HEAL_ZONE_SIZE_RATIO,
    cp_count: int = config.CP_COUNT,
    cp_ratio: float = config.CP_SIZE_RATIO,
    device: torch.device | None = None,
) -> Zones:
    """
    Returns boolean masks for heal and capture zones.
    - Sizes are computed as round(ratio * grid_dimension) per side (rectangular).
    - Masks are non-overlapping where possible; if overlap occurs, that's OK (semantics are additive).
    """
    device = device or config.TORCH_DEVICE
    heal_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    cp_masks: List[torch.Tensor] = []

    # Helper to sample a rectangle that sits within the interior
    def _sample_rect(h_side: int, w_side: int) -> Tuple[int, int, int, int]:
        # keep 1-cell border clear (outer walls)
        x0 = random.randint(1, max(1, W - w_side - 2))
        y0 = random.randint(1, max(1, H - h_side - 2))
        return y0, y0 + h_side, x0, x0 + w_side  # (y0, y1, x0, x1)

    # Heal zones
    if heal_count > 0 and heal_ratio > 0.0:
        h_side = max(1, int(round(heal_ratio * H)))
        w_side = max(1, int(round(heal_ratio * W)))
        for _ in range(int(heal_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)
            heal_mask[y0:y1, x0:x1] = True

    # Capture zones
    if cp_count > 0 and cp_ratio > 0.0:
        h_side = max(1, int(round(cp_ratio * H)))
        w_side = max(1, int(round(cp_ratio * W)))
        for _ in range(int(cp_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)
            m = torch.zeros((H, W), dtype=torch.bool, device=device)
            m[y0:y1, x0:x1] = True
            cp_masks.append(m)

    return Zones(heal_mask=heal_mask, cp_masks=cp_masks)
