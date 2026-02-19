from __future__ import annotations
import torch
import config

import torch  
from torch import Tensor  
def make_grid(device: torch.device) -> torch.Tensor:
    """
    Channels:
      0: occupancy (0 empty, 1 wall, 2 red, 3 blue)
      1: hp        (0..MAX_HP)
      2: agent_id  (-1 for empty)
    """
    H, W = config.GRID_HEIGHT, config.GRID_WIDTH
    g = torch.zeros((3, H, W), dtype=config.TORCH_DTYPE, device=device)
    # walls
    g[0, 0, :] = 1.0; g[0, H-1, :] = 1.0; g[0, :, 0] = 1.0; g[0, :, W-1] = 1.0
    g[2].fill_(-1.0)
    return g

def assert_on_same_device(*tensors: torch.Tensor) -> None:
    """
    Hard guard: all tensors must be on same device & dtype matches config.torch_dtype for floats.
    """
    if not tensors:
        return
    dev = tensors[0].device
    for t in tensors:
        if t.device != dev:
            raise RuntimeError(f"Device mismatch: {dev} vs {t.device}")
        if t.is_floating_point() and t.dtype != config.TORCH_DTYPE:
            raise RuntimeError(f"Dtype mismatch: expected {config.TORCH_DTYPE}, got {t.dtype}")
