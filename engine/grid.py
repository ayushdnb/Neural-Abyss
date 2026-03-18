"""Grid allocation and device consistency checks."""

from __future__ import annotations

import torch
import config
from torch import Tensor

def make_grid(device: torch.device) -> torch.Tensor:
    """
    Construct and return the simulation "grid" tensor.

    OVERVIEW
    --------
    The grid is represented as a rank-3 tensor with shape (C, H, W), where:
      - C = number of channels (here, 3)
      - H = GRID_HEIGHT (from config)
      - W = GRID_WIDTH  (from config)

    This is a common pattern in ML systems and simulation engines:
      - (channels, height, width) is analogous to an "image" with multiple feature planes.
      - Each channel encodes a different semantic aspect of the environment state.

    CHANNEL SEMANTICS
    -----------------
    Channel 0: occupancy / cell type encoding (float-valued but used as categorical codes)
      0.0 => empty
      1.0 => wall
      2.0 => red team unit (or red occupant)
      3.0 => blue team unit (or blue occupant)

    Channel 1: hp (hit points), typically in range [0, MAX_HP]
      - This channel stores a scalar "health" per cell.
      - In many grid simulations, hp is meaningful only where an agent exists; elsewhere it is 0.

    Channel 2: agent_id (identifier of the agent occupying the cell)
      - By convention in this code: -1.0 indicates "no agent".
      - Otherwise stores an ID value (often an integer in meaning, even if stored as float).
      - The use of -1 is a standard sentinel value indicating "empty" or "invalid index".

    IMPORTANT NOTE ON DTYPE AND REPRESENTATION
    ------------------------------------------
    The grid is created with dtype=config.TORCH_DTYPE, which is likely a floating type.
    Even though occupancy and agent_id are conceptually discrete/integer, using a float dtype:
      - keeps all channels in a single tensor with a single dtype,
      - avoids mixed-dtype operations,
      - is convenient for neural network inputs and GPU operations.
    This is a design choice: semantics are discrete, but representation is float.

    PARAMETERS
    ----------
    device: torch.device
      - The device on which the grid tensor will be allocated.
      - Examples: torch.device("cpu"), torch.device("cuda"), torch.device("cuda:0")

    RETURNS
    -------
    torch.Tensor
      - Tensor of shape (3, H, W), with dtype=config.TORCH_DTYPE on the specified device.
      - Initialized as empty interior with boundary walls and agent_id set to -1 everywhere.

    PERFORMANCE / ENGINEERING RATIONALE
    ----------------------------------
    - Using a single contiguous tensor for environment state is GPU-friendly.
    - Channel-first layout (C, H, W) is standard in many PyTorch workflows.
    - The operations used here are vectorized assignments (fast, minimal Python loops).
    """

    # Read grid dimensions from configuration. These should be integers.
    # H = number of rows (vertical axis), W = number of columns (horizontal axis).
    H, W = config.GRID_HEIGHT, config.GRID_WIDTH

    # Allocate a tensor of zeros with shape (3, H, W).
    # torch.zeros(...) creates a tensor filled with 0.0 (or 0 for integer dtypes).
    # dtype=config.TORCH_DTYPE ensures consistency across the system:
    #   - computations remain in a known precision (e.g., float32),
    #   - reduces accidental dtype promotion/demotion,
    #   - improves reproducibility and performance predictability.
    # device=device places the tensor directly on the target device (CPU/GPU).
    g = torch.zeros((3, H, W), dtype=config.TORCH_DTYPE, device=device)

    # WALL INITIALIZATION (BOUNDARY CONDITIONS)
    # We create a "box" of walls around the perimeter of the grid.
    # This is a common technique in grid simulations:
    # - It avoids needing extra bounds checks during movement or ray-casting.
    # - Agents cannot leave the map because boundary cells are non-traversable.
    # Here, we set occupancy channel (channel 0) to 1.0 (wall code) on:
    #   - top row:    y = 0
    #   - bottom row: y = H-1
    #   - left col:   x = 0
    #   - right col:  x = W-1
    # Indexing details:
    #   g[0, 0, :]     => channel 0, row 0, all columns
    #   g[0, H-1, :]   => channel 0, row H-1, all columns
    #   g[0, :, 0]     => channel 0, all rows, column 0
    #   g[0, :, W-1]   => channel 0, all rows, column W-1
    # Note: Semicolons are used to place multiple statements on one line.
    # Kept exactly as provided.
    g[0, 0, :] = 1.0; g[0, H-1, :] = 1.0; g[0, :, 0] = 1.0; g[0, :, W-1] = 1.0

    # AGENT ID INITIALIZATION
    # Channel 2 stores agent_id. We set it to -1.0 everywhere.
    # Why -1?
    # - In indexing contexts, valid IDs are usually non-negative (0..N-1).
    # - -1 is a conventional sentinel meaning "no agent present".
    # Why fill_?
    # - g[2] selects channel 2 with shape (H, W) as a view.
    # - .fill_(value) performs an in-place fill on that view efficiently.
    g[2].fill_(-1.0)

    # Return the fully initialized grid.
    return g


def assert_on_same_device(*tensors: torch.Tensor) -> None:
    """
    Runtime invariant check: ensure tensors are colocated on the same device
    and (for floating tensors) use the configured dtype.

    PURPOSE
    -------
    In GPU-accelerated systems, device mismatches are a frequent source of:
      - runtime errors (e.g., attempting to add CPU tensor to CUDA tensor),
      - silent performance regressions (unexpected CPU fallback),
      - correctness bugs (implicit copies, hidden casts, or unexpected precision).

    This function acts as a "hard guard" to fail fast and loudly when:
      1) tensors are spread across different devices, OR
      2) a floating tensor's dtype does not match config.TORCH_DTYPE.

    DESIGN DETAILS
    --------------
    - Accepts a variable number of tensors: *tensors
      This enables calls like:
        assert_on_same_device(t1, t2, t3)

    - If no tensors are provided, it returns immediately.
      This makes it safe to call even when a list might be empty.

    DEVICE CHECK
    ------------
    - The first tensor defines the reference device (dev).
    - Every subsequent tensor must match dev exactly.

    DTYPE CHECK (FLOATING ONLY)
    ---------------------------
    - Only applies to floating point tensors (t.is_floating_point()).
      This is important because:
        - integer tensors may legitimately use int64/int32 for indexing,
        - boolean tensors may be used for masks,
        - enforcing float dtype on those would be incorrect.

    - For float tensors, enforce:
        t.dtype == config.TORCH_DTYPE

    ERROR HANDLING
    --------------
    - Raises RuntimeError with a clear diagnostic message.
      RuntimeError is appropriate here because this is a runtime invariant violation,
      not a recoverable condition.

    RETURN VALUE
    ------------
    - Returns None on success; raises an exception on failure.
    """

    # If the caller passes no tensors, there is nothing to validate.
    if not tensors:
        return

    # Use the device of the first tensor as the reference device.
    # In PyTorch, t.device is a torch.device object (e.g., cpu, cuda:0).
    dev = tensors[0].device

    # Iterate through each provided tensor and validate constraints.
    for t in tensors:
        # DEVICE CONSISTENCY CHECK
        # If any tensor is on a different device than the reference, abort.
        # Example mismatch: dev=cpu vs t.device=cuda:0
        if t.device != dev:
            raise RuntimeError(f"Device mismatch: {dev} vs {t.device}")

        # FLOAT DTYPE CONSISTENCY CHECK
        # Only enforce dtype for floating point tensors.
        # This avoids incorrectly rejecting integer/boolean tensors that are
        # legitimately used as indices/masks.
        # If floating, the dtype must match the project's configured float dtype.
        # This is commonly float32 for speed, or float16/bfloat16 for AMP/mixed precision.
        if t.is_floating_point() and t.dtype != config.TORCH_DTYPE:
            raise RuntimeError(f"Dtype mismatch: expected {config.TORCH_DTYPE}, got {t.dtype}")
