from __future__ import annotations
import torch
import config

# 8 directions (dx, dy): N, NE, E, SE, S, SW, W, NW
DIRS8 = torch.tensor([
    [ 0, -1],
    [ 1, -1],
    [ 1,  0],
    [ 1,  1],
    [ 0,  1],
    [-1,  1],
    [-1,  0],
    [-1, -1],
], dtype=torch.long)

@torch.no_grad()
def build_mask(
    pos_xy: torch.Tensor,            # (N,2) long/float (x,y)
    teams: torch.Tensor,             # (N,)  float: 2.0=red, 3.0=blue
    grid: torch.Tensor,              # (3,H,W) float; ch0=occ(0,1,2,3)
    unit: torch.Tensor | None = None # (N,) long/float: 1=soldier, 2=archer
) -> torch.Tensor:
    """
    Returns action mask [N, A] bool.
      A=17: idle(1) + 8 moves + 8 melee (r=1)
      A=41: idle(1) + 8 moves + 8×(r=1..4) ranged
        - soldier: r=1 only
        - archer:  r=1..config.ARCHER_RANGE (clipped to 4)
    """
    device = grid.device
    N = int(pos_xy.size(0))
    H, W = int(grid.size(1)), int(grid.size(2))
    A = int(getattr(config, "NUM_ACTIONS", 17))
    mask = torch.zeros((N, A), dtype=torch.bool, device=device)

    # idle
    if A >= 1:
        mask[:, 0] = True

    if N == 0 or A <= 1:
        return mask

    # positions
    x0 = pos_xy[:, 0].to(torch.long, non_blocking=True)
    y0 = pos_xy[:, 1].to(torch.long, non_blocking=True)
    dirs = DIRS8.to(device)  # (8,2)

    # -------------------- MOVE (cols 1..8) --------------------
    nx = x0.view(N, 1) + dirs[:, 0].view(1, 8)
    ny = y0.view(N, 1) + dirs[:, 1].view(1, 8)
    inb = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
    nx_cl = nx.clamp(0, W - 1)
    ny_cl = ny.clamp(0, H - 1)
    occ = grid[0][ny_cl, nx_cl]          # (N,8)
    free = (occ == 0.0) & inb
    move_cols = min(8, max(0, A - 1))
    if move_cols > 0:
        mask[:, 1:1 + move_cols] = free[:, :move_cols]

    # -------------------- ATTACK --------------------
    if A <= 9:
        return mask  # no attack columns at all

    teamv = teams.to(torch.long, non_blocking=True)  # (N,)

    # Legacy 17-action: only r=1 melee per dir
    if A <= 17:
        tgt_team = occ  # r=1 neighbor occupancy
        enemy = (tgt_team != 0.0) & (tgt_team != 1.0) & (tgt_team != teamv.view(N, 1))
        k = min(8, max(0, A - 9))
        if k > 0:
            mask[:, 9:9 + k] = enemy[:, :k]
        return mask

    # 41-action layout: 8 dirs × 4 ranges
    RMAX = 4
    dx = dirs[:, 0].view(1, 8, 1)  # (1,8,1)
    dy = dirs[:, 1].view(1, 8, 1)
    rvec = torch.arange(1, RMAX + 1, device=device, dtype=torch.long).view(1, 1, RMAX)

    tx = x0.view(N, 1, 1) + dx * rvec  # (N,8,4)
    ty = y0.view(N, 1, 1) + dy * rvec
    inb_r = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
    txc = tx.clamp(0, W - 1)
    tyc = ty.clamp(0, H - 1)

    tgt_occ = grid[0][tyc, txc]  # (N,8,4)
    enemy_r = (tgt_occ != 0.0) & (tgt_occ != 1.0) & (tgt_occ.to(torch.long) != teamv.view(N, 1, 1))
    enemy_r &= inb_r  # enforce bounds

    # Unit gating: soldiers r=1; archers r<=ARCHER_RANGE
    if unit is None:
        units = torch.full((N,), 2, device=device, dtype=torch.long)  # default permissive: archer
    else:
        units = unit.to(torch.long, non_blocking=True)

    ar_range = int(getattr(config, "ARCHER_RANGE", 4))
    ar_range = max(1, min(RMAX, ar_range))

    allow_r = torch.zeros((N, RMAX), dtype=torch.bool, device=device)  # (N,4)
    # soldiers
    allow_r[units == 1, 0] = True
    # archers
    if (units == 2).any():
        allow_r[units == 2, :ar_range] = True

    atk_ok = enemy_r & allow_r.view(N, 1, RMAX)  # (N,8,4)

    # Write contiguous 4-column blocks per direction
    base = 9
    for d in range(8):
        c0 = base + d * RMAX
        c1 = c0 + RMAX
        if c0 >= A:
            break
        cols = slice(c0, min(c1, A))
        rlim = cols.stop - cols.start  # number of range columns we’ll write (<=4)
        if rlim > 0:
            mask[:, cols] = atk_ok[:, d, :rlim]

    return mask
