from __future__ import annotations
# ──────────────────────────────────────────────────────────────────────────────
# Forward-annotation semantics
# ──────────────────────────────────────────────────────────────────────────────
# This directive causes all type annotations to be stored as strings rather than
# evaluated immediately. In a multi-module simulation codebase (often featuring
# circular imports between engine, registry, and environment utilities), this is
# a pragmatic choice to reduce import-order fragility.
#
# It has no impact on tensor operations or runtime behavior beyond annotation
# evaluation semantics.
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Standard library and typing imports
# ──────────────────────────────────────────────────────────────────────────────
# dataclass provides a clean, declarative way to define data containers.
from dataclasses import dataclass
#
# List / Tuple cover rectangular masks, while Dict / Any support explicit
# checkpoint-bridge helpers on the zone container.
from typing import Any, Dict, List, Tuple
#
# random is Python's standard pseudo-random generator. It is used here for
# sampling starting points, directions, and rectangle placements.
import random

# ──────────────────────────────────────────────────────────────────────────────
# PyTorch import
# ──────────────────────────────────────────────────────────────────────────────
# torch is used for:
#   • grid manipulation
#   • boolean masks representing zones
#   • device placement (CPU/GPU)
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Project configuration
# ──────────────────────────────────────────────────────────────────────────────
# This module is expected to define knobs such as:
#   • TORCH_DEVICE
#   • RANDOM_WALLS, WALL_SEG_MIN, WALL_SEG_MAX, WALL_AVOID_MARGIN
#   • HEAL_ZONE_COUNT, HEAL_ZONE_SIZE_RATIO
#   • CP_COUNT, CP_SIZE_RATIO
#
# Those knobs determine the number, size, and placement of procedural features.
import config

# ──────────────────────────────────────────────────────────────────────────────
# Grid channels documentation
# ──────────────────────────────────────────────────────────────────────────────
# The code operates on a grid tensor of shape (3, H, W) with these semantics:
#
#   grid[0] occupancy (tile kind / team marker)
#       0 empty
#       1 wall
#       2 red occupancy marker
#       3 blue occupancy marker
#
#   grid[1] hp
#       typically 0..MAX_HP; for walls/empty tiles, it is usually 0 or unused.
#
#   grid[2] agent_id
#       -1 indicates no agent is present
#       >=0 indicates an agent exists in that cell (registry slot/id)
#
# The zone masks below are deliberately kept as separate boolean tensors rather
# than being embedded into occupancy channels. This is a design decision that
# reduces churn in renderer/engine logic: special zones can be added/removed
# without changing core grid encoding or touching other subsystems.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Zones:
    """
    Canonical semantic-zone container kept off-grid to avoid renderer/engine churn.

    Patch-1 / Patch-3 contract:
    ---------------------------
    The simulation is moving away from a heal-only zone model toward a more
    general signed base-zone representation. This class is the foundational seam
    for that refactor:

      • `base_zone_value_map` is the canonical persistent zone field
      • positive values represent beneficial/heal-like base zones
      • negative values represent harmful/poison-like base zones
      • zero means dormant / no base-zone effect
      • capture-point logic remains separate in `cp_masks`

    Patch 3 adds a minimal operator-facing editing seam:
    - viewer edits target the canonical base layer only
    - edits clamp into [-1, +1]
    - optional future lock masks can block edits without redefining the base map

    Why keep zones off-grid?
    ------------------------
    The occupancy / hp / agent-id grid is already heavily coupled to movement,
    rendering, and observation code. A separate zone container lets the repo grow
    zone semantics without rewriting those core channels.

    Compatibility policy in this patch:
    -----------------------------------
    - `base_zone_value_map` is the new canonical field.
    - `heal_mask` remains available as a derived compatibility helper so legacy
      read-only consumers (viewer, old summaries, older checkpoint readers) can
      still treat positive base zones as classic heal tiles.
    - Capture-point masks stay boolean and behaviorally unchanged.
    - Catastrophe override logic is intentionally NOT implemented yet.

    Data invariants:
    ---------------
    • `base_zone_value_map` has shape (H, W), floating dtype, values clamped to
      [-1, +1].
    • every CP mask has shape (H, W) and dtype torch.bool.
    • tensors should live on one device so GPU-side consumers do not need extra
      host/device transfers.
    """
    base_zone_value_map: torch.Tensor        # signed float grid in [-1, +1]
    cp_masks: List[torch.Tensor]             # list of boolean masks for capture patches

    def __post_init__(self) -> None:
        """
        Normalize zone tensors into a stable canonical form.

        This is intentionally conservative:
        - preserve shape
        - clamp signed values into the advertised range
        - keep CP masks boolean
        - fail fast on mismatched shapes
        """
        base = self.base_zone_value_map
        if not torch.is_tensor(base):
            raise TypeError("base_zone_value_map must be a torch.Tensor")
        if int(base.ndim) != 2:
            raise ValueError(
                f"base_zone_value_map must be rank-2 (H,W), got shape={tuple(base.shape)}"
            )

        self.base_zone_value_map = base.to(dtype=torch.float32).clamp(-1.0, 1.0)

        H, W = int(self.base_zone_value_map.shape[0]), int(self.base_zone_value_map.shape[1])
        norm_cp_masks: List[torch.Tensor] = []
        for i, m in enumerate(list(self.cp_masks or [])):
            if not torch.is_tensor(m):
                raise TypeError(f"cp_masks[{i}] must be a torch.Tensor")
            if tuple(m.shape) != (H, W):
                raise ValueError(
                    f"cp_masks[{i}] shape {tuple(m.shape)} does not match base zone shape {(H, W)}"
                )
            norm_cp_masks.append(m.bool())
        self.cp_masks = norm_cp_masks

    @classmethod
    def from_legacy_heal_mask(
        cls,
        *,
        heal_mask: torch.Tensor,
        cp_masks: List[torch.Tensor],
    ) -> "Zones":
        """
        Construct canonical signed zones from the legacy heal-mask payload.

        Legacy meaning:
        - heal_mask == True  -> base-zone value +1.0
        - heal_mask == False -> base-zone value  0.0

        This is the checkpoint bridge for old worlds saved before the signed-zone
        foundation existed.
        """
        if not torch.is_tensor(heal_mask):
            raise TypeError("heal_mask must be a torch.Tensor")

        heal_mask_bool = heal_mask.bool()
        base_zone_value_map = torch.zeros(
            heal_mask_bool.shape,
            dtype=torch.float32,
            device=heal_mask_bool.device,
        )
        base_zone_value_map[heal_mask_bool] = 1.0
        return cls(base_zone_value_map=base_zone_value_map, cp_masks=list(cp_masks or []))

    @property
    def heal_mask(self) -> torch.Tensor:
        """
        Derived legacy compatibility view of positive base zones.

        Important:
        - this is NOT the canonical storage field anymore
        - callers that only need old heal semantics can still consume it safely
        - negative signed zones are intentionally excluded from this view
        """
        return self.base_positive_mask

    @property
    def base_positive_mask(self) -> torch.Tensor:
        """Return boolean mask of cells with positive base-zone value."""
        return self.base_zone_value_map > 0.0

    @property
    def base_negative_mask(self) -> torch.Tensor:
        """Return boolean mask of cells with negative base-zone value."""
        return self.base_zone_value_map < 0.0

    @property
    def cp_count(self) -> int:
        """Return the number of distinct capture-point masks."""
        return len(self.cp_masks)

    @property
    def base_zone_edit_locked_mask(self) -> torch.Tensor:
        """
        Return a boolean mask of cells currently locked against manual base-zone edits.

        Patch-3 default behavior:
        - catastrophe override logic does not exist yet
        - therefore the default lock mask is all-False

        Forward-compatibility hook:
        - a later subsystem may inject `_base_zone_edit_locked_mask`
        - the viewer can consult this property without knowing who set it
        """
        mask = getattr(self, "_base_zone_edit_locked_mask", None)
        if mask is None:
            return torch.zeros_like(self.base_zone_value_map, dtype=torch.bool)
        if not torch.is_tensor(mask):
            raise TypeError("_base_zone_edit_locked_mask must be a torch.Tensor when provided")
        if tuple(mask.shape) != tuple(self.base_zone_value_map.shape):
            raise ValueError(
                f"_base_zone_edit_locked_mask shape {tuple(mask.shape)} does not match base zone shape {tuple(self.base_zone_value_map.shape)}"
            )
        return mask.to(device=self.base_zone_value_map.device, dtype=torch.bool)

    def _validate_xy(self, x: int, y: int) -> tuple[int, int]:
        """Normalize and bounds-check one grid coordinate pair."""
        xi = int(x)
        yi = int(y)
        H, W = int(self.base_zone_value_map.shape[0]), int(self.base_zone_value_map.shape[1])
        if not (0 <= xi < W and 0 <= yi < H):
            raise IndexError(f"base-zone coordinate out of bounds: ({xi}, {yi}) for grid {(W, H)}")
        return xi, yi

    def is_base_zone_edit_locked_at(self, x: int, y: int) -> bool:
        """Return True when manual editing should be rejected at one cell."""
        xi, yi = self._validate_xy(x, y)
        return bool(self.base_zone_edit_locked_mask[yi, xi].item())

    def get_base_zone_value_at(self, x: int, y: int) -> float:
        """Return the canonical signed base-zone value at one cell."""
        xi, yi = self._validate_xy(x, y)
        return float(self.base_zone_value_map[yi, xi].item())

    def set_base_zone_value_at(
        self,
        x: int,
        y: int,
        value: float,
        *,
        respect_edit_lock: bool = True,
    ) -> float:
        """
        Set one cell in the canonical base-zone layer and return the stored value.

        Guarantees:
        - clamps to [-1, +1]
        - edits the canonical persistent base layer only
        - optionally respects a future edit-lock mask
        """
        xi, yi = self._validate_xy(x, y)
        if respect_edit_lock and self.is_base_zone_edit_locked_at(xi, yi):
            raise RuntimeError(f"base-zone cell ({xi}, {yi}) is locked for manual editing")

        v = float(max(-1.0, min(1.0, float(value))))
        self.base_zone_value_map[yi, xi] = v
        return float(self.base_zone_value_map[yi, xi].item())

    def add_base_zone_delta_at(
        self,
        x: int,
        y: int,
        delta: float,
        *,
        respect_edit_lock: bool = True,
    ) -> float:
        """Add a delta to one cell in the canonical base-zone layer."""
        current = self.get_base_zone_value_at(x, y)
        return self.set_base_zone_value_at(
            x,
            y,
            current + float(delta),
            respect_edit_lock=respect_edit_lock,
        )

    def set_runtime_base_zone_edit_locked_mask(self, mask: torch.Tensor | None) -> None:
        """
        Install or clear a runtime-only base-zone edit-lock mask.
        """
        if mask is None:
            if hasattr(self, "_base_zone_edit_locked_mask"):
                delattr(self, "_base_zone_edit_locked_mask")
            return
        if not torch.is_tensor(mask):
            raise TypeError("runtime base-zone edit lock mask must be a torch.Tensor or None")
        if tuple(mask.shape) != tuple(self.base_zone_value_map.shape):
            raise ValueError(
                f"runtime base-zone edit lock mask shape {tuple(mask.shape)} does not match base zone shape {tuple(self.base_zone_value_map.shape)}"
            )
        self._base_zone_edit_locked_mask = mask.to(device=self.base_zone_value_map.device, dtype=torch.bool)

    def clear_runtime_base_zone_edit_locked_mask(self) -> None:
        """Remove any runtime-injected base-zone edit lock mask."""
        self.set_runtime_base_zone_edit_locked_mask(None)

    def checkpoint_base_zones_payload(self) -> Dict[str, Any]:
        """
        Return the canonical base-zone payload fragment used for checkpoints.

        The payload is intentionally small and explicit so future layers
        (catastrophe overrides, viewer edits, derived effective fields) can add
        their own sections without redefining the base-zone contract.
        """
        return {"value_map": self.base_zone_value_map}


# ------------------------------------------------------------------------------
# Random thin gray walls (1-cell thick, meandering segments)
# ------------------------------------------------------------------------------
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
    Procedurally carve “thin” (1-cell thick) wall traces into a grid.

    Functional objective:
    ---------------------
    This function modifies grid *in-place* by writing walls into the occupancy
    channel (grid[0]) using the wall code 1.0. It constructs walls as a set of
    random meandering polyline-like segments on the grid.

    Intended call timing:
    ---------------------
    It is designed to be called *before* spawning agents. If called after agent
    spawn, and allow_over_agents is False, it avoids overwriting agent cells.

    Inputs:
    -------
    grid:
        Tensor of shape (3, H, W). The function asserts this minimal structure.

    n_segments:
        Number of independent wall traces to draw. Each trace is a random walk
        of length L (see seg_min/seg_max).

    seg_min, seg_max:
        Minimum/maximum segment length, in steps. Each segment draws L steps.

    avoid_margin:
        Prevents starting positions too close to boundaries. This helps avoid
        interfering with existing “outer border walls” often pre-generated by
        the grid maker. Additionally, the function clamps motion within the
        interior [1, W-2] × [1, H-2].

    allow_over_agents:
        If False, the function will not place walls on cells whose occupancy is
        currently 2.0 or 3.0 (agent/team markers). If True, it can overwrite.

    Side effects (in-place mutations):
    ---------------------------------
    For each wall cell placed at (x, y), the function sets:
      • grid[0, y, x] = 1.0   (occupancy becomes wall)
      • grid[1, y, x] = 0.0   (hp cleared)
      • grid[2, y, x] = -1.0  (agent_id cleared)

    This enforces a strong invariant:
      “A wall cell cannot simultaneously contain an agent or HP state.”
    """

    # Validate grid shape: must be at least 3 channels.
    assert grid.ndim == 3 and grid.size(0) >= 3, "grid must be (3,H,W)"

    # occ is a view into the occupancy channel for concise access.
    occ = grid[0]

    # Height and width are derived from the occupancy channel shape (H, W).
    H, W = int(occ.size(0)), int(occ.size(1))

    # --------------------------------------------------------------------------
    # Direction set: 8-connected neighborhood (dx, dy)
    # --------------------------------------------------------------------------
    # The random walk uses the standard 8 directions (including diagonals),
    # enabling “meandering” walls that can move diagonally as well as orthogonally.
    #
    # Storing this as a tensor on occ.device ensures:
    #   • no device mismatch if grid is on GPU
    #   • direction lookup is cheap and device-resident
    dirs8 = torch.tensor(
        [[ 0, -1],[ 1, -1],[ 1,  0],[ 1,  1],[ 0,  1],[-1,  1],[-1,  0],[-1, -1]],
        dtype=torch.long, device=occ.device
    )

    # --------------------------------------------------------------------------
    # Local helper: place a single wall cell with optional agent avoidance
    # --------------------------------------------------------------------------
    def _place_wall_cell(x: int, y: int) -> None:
        """
        Write a wall cell into the grid at (x, y), subject to bounds and policy.

        Bounds:
        -------
        The function guards against out-of-range coordinates.

        Agent-overwrite policy:
        -----------------------
        If allow_over_agents is False, and the current occupancy at (x, y)
        indicates an agent cell (2.0 or 3.0), the placement is skipped.

        Invariant enforcement:
        ----------------------
        When a wall is placed, the function clears hp and agent_id at that cell,
        ensuring internal consistency between grid channels.
        """
        if 0 <= x < W and 0 <= y < H:
            if not allow_over_agents:
                v = float(occ[y, x].item())
                if v in (2.0, 3.0):  # skip unit cells
                    return
            occ[y, x] = 1.0
            grid[1, y, x] = 0.0
            grid[2, y, x] = -1.0

    # --------------------------------------------------------------------------
    # Compute allowable start region for wall traces
    # --------------------------------------------------------------------------
    # The code respects an avoid_margin while also keeping at least a 1-cell
    # border clear. This is important because many grid generators pre-wall the
    # boundary to prevent agents from leaving the world.
    #
    # x0_min/x0_max and y0_min/y0_max define a valid inclusive range for random
    # starting points. Note that random.randint(a, b) includes both endpoints.
    x0_min, x0_max = max(1, avoid_margin), W - max(1, avoid_margin) - 1
    y0_min, y0_max = max(1, avoid_margin), H - max(1, avoid_margin) - 1

    # If the grid is too small for the margins, or no segments requested, exit.
    if x0_min >= x0_max or y0_min >= y0_max or n_segments <= 0:
        return

    # --------------------------------------------------------------------------
    # Draw each segment as a biased random walk
    # --------------------------------------------------------------------------
    # For each segment:
    #   • sample a random start (x, y)
    #   • sample a random length L
    #   • place the initial cell
    #   • then for L steps:
    #       - choose direction with turn bias
    #       - step and clamp to interior
    #       - place cell
    #       - occasionally introduce a gap
    #
    # The bias is important: without it, walls can jitter chaotically and
    # self-intersect aggressively; with bias, they appear as smoother "traces".
    for _ in range(max(0, int(n_segments))):
        x = random.randint(x0_min, x0_max)
        y = random.randint(y0_min, y0_max)

        # Segment length L is sampled uniformly in [seg_min, seg_max], but both
        # are clamped to at least 1 to avoid degenerate segments.
        L = random.randint(max(1, int(seg_min)), max(1, int(seg_max)))

        # Place the starting cell of the segment.
        _place_wall_cell(x, y)

        # last_dir is a direction index in [0, 7].
        last_dir = random.randrange(8)

        for _step in range(L):
            # ------------------------------------------------------------------
            # Direction choice with “small turn bias”
            # ------------------------------------------------------------------
            # With probability 0.70, keep the same direction (continue forward).
            # Otherwise, turn slightly by ±1 or ±2 steps in the direction index.
            #
            # This creates gentle bends rather than noisy zig-zags.
            if random.random() < 0.70:
                d = last_dir
            else:
                d = (last_dir + random.choice([-2, -1, 1, 2])) % 8
            last_dir = d

            # Extract dx, dy from dirs8 tensor. .item() returns a Python scalar.
            dx, dy = int(dirs8[d, 0].item()), int(dirs8[d, 1].item())

            # ------------------------------------------------------------------
            # Step and clamp inside interior
            # ------------------------------------------------------------------
            # The wall is constrained to the interior grid to avoid painting over
            # boundary walls:
            #   x ∈ [1, W-2], y ∈ [1, H-2]
            #
            # This also prevents index errors.
            x = max(1, min(W - 2, x + dx))
            y = max(1, min(H - 2, y + dy))

            # Place wall cell at the new position.
            _place_wall_cell(x, y)

            # ------------------------------------------------------------------
            # Occasional deliberate gaps
            # ------------------------------------------------------------------
            # With probability 0.05, we do nothing (leave a gap).
            #
            # Design rationale:
            #   Continuous walls can partition the map into disconnected regions,
            #   potentially creating unreachable areas or degenerately trapping
            #   agents. Introducing random gaps reduces the risk of total
            #   partition while preserving the qualitative “maze-like” structure.
            if random.random() < 0.05:
                # leave a deliberate gap (do nothing)
                pass



# ------------------------------------------------------------------------------
# Base signed zones (+ legacy heal generation) and capture zones
# ------------------------------------------------------------------------------
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
    Create signed base-zone values plus separate capture-zone masks.

    Conceptual overview:
    --------------------
    The simulation includes special areas:
      • Base zones: signed scalar tiles that currently originate from legacy
        heal rectangles and therefore start at +1.0 where active.
      • Capture zones: rectangular “patches” that can be captured/contested.

    Rather than encoding these into the occupancy channel, this function produces
    separate tensors which can be consumed by:
      • the tick engine (to apply current heal-like base-zone rules and CP rules)
      • the renderer (to visualize positive base zones as legacy heal tiles)
      • the observation builder (to expose zone info to agents)

    Parameters:
    -----------
    H, W:
        Grid height and width.

    heal_count:
        Number of heal rectangles to place.

    heal_ratio:
        Side-length ratio for heal rectangles relative to grid dimensions.
        The actual rectangle side lengths are computed as:
            h_side = round(heal_ratio * H)
            w_side = round(heal_ratio * W)
        then clamped to at least 1.

    cp_count:
        Number of capture rectangles to place.

    cp_ratio:
        Side-length ratio for capture rectangles (same computation strategy).

    device:
        Optional torch.device controlling where masks live. If None, defaults to
        config.TORCH_DEVICE.

    Output:
    -------
    Zones object containing:
      • base_zone_value_map: shape (H, W), dtype float32, values in [-1, +1]
      • cp_masks: list of cp_count boolean masks, each shape (H, W)

    Current generation policy:
    --------------------------
    Patch 1 keeps map generation behavior as close as possible to the old repo:
    every legacy heal rectangle simply writes +1.0 into the canonical base-zone
    field. Negative values are supported by the data model but are not generated
    here yet.

    Overlap semantics:
    ------------------
    The docstring states that masks are "non-overlapping where possible", but
    the implementation does not enforce non-overlap constraints; it simply
    samples rectangles independently. Therefore, overlap may occur:
      • heal zones may overlap with capture zones
      • capture zones may overlap each other
    and the comment clarifies that overlap is acceptable (semantics additive).
    """

    # Default device: if none provided, use config.TORCH_DEVICE.
    device = device or config.TORCH_DEVICE

    # Canonical signed base-zone field. Patch 1 still seeds it from the legacy
    # heal-zone generator, so positive tiles are written as +1.0.
    base_zone_value_map = torch.zeros((H, W), dtype=torch.float32, device=device)

    # cp_masks is a list of distinct capture zone masks.
    cp_masks: List[torch.Tensor] = []

    # --------------------------------------------------------------------------
    # Helper: sample a rectangle fully contained within the interior
    # --------------------------------------------------------------------------
    def _sample_rect(h_side: int, w_side: int) -> Tuple[int, int, int, int]:
        """
        Sample an axis-aligned rectangle within the interior region of the grid.

        Returns:
        --------
        (y0, y1, x0, x1) defining a slice:
            rows y0:y1 and cols x0:x1

        Boundary policy:
        ----------------
        The sampler keeps a 1-cell border clear, consistent with the assumption
        that outer walls exist around the perimeter.

        Implementation detail:
        ----------------------
        random.randint(a, b) is inclusive at both ends. The max(...) ensures the
        upper bound is at least 1 to prevent ValueError for small grids.
        """
        # keep 1-cell border clear (outer walls)
        x0 = random.randint(1, max(1, W - w_side - 2))
        y0 = random.randint(1, max(1, H - h_side - 2))
        return y0, y0 + h_side, x0, x0 + w_side  # (y0, y1, x0, x1)

    # --------------------------------------------------------------------------
    # Legacy heal rectangles: write +1.0 into the canonical base-zone field
    # --------------------------------------------------------------------------
    if heal_count > 0 and heal_ratio > 0.0:
        # Compute rectangle side lengths proportional to grid dimensions.
        # round(...) allows proportional sizes to scale naturally with grid size.
        h_side = max(1, int(round(heal_ratio * H)))
        w_side = max(1, int(round(heal_ratio * W)))

        for _ in range(int(heal_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)

            # Repeated assignments naturally union together because writing the
            # same +1.0 value over overlapping rectangles is idempotent.
            base_zone_value_map[y0:y1, x0:x1] = 1.0

    # --------------------------------------------------------------------------
    # Capture zones: create a separate mask per rectangle and append to list
    # --------------------------------------------------------------------------
    if cp_count > 0 and cp_ratio > 0.0:
        h_side = max(1, int(round(cp_ratio * H)))
        w_side = max(1, int(round(cp_ratio * W)))

        for _ in range(int(cp_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)

            # Each capture zone is a distinct mask so the engine can track
            # ownership/contest state per zone independently (implied by design).
            m = torch.zeros((H, W), dtype=torch.bool, device=device)
            m[y0:y1, x0:x1] = True
            cp_masks.append(m)

    # Package into the canonical Zones dataclass and return.
    return Zones(base_zone_value_map=base_zone_value_map, cp_masks=cp_masks)

