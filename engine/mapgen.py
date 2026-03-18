from __future__ import annotations

# Standard library and typing imports
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random

# PyTorch import
import torch

# Project configuration
import config

# Grid channels documentation
# The code operates on a grid tensor of shape (3, H, W) with these semantics:
#   grid[0] occupancy (tile kind / team marker)
#       0 empty
#       1 wall
#       2 red occupancy marker
#       3 blue occupancy marker
#   grid[1] hp
#       typically 0..MAX_HP; for walls/empty tiles, it is usually 0 or unused.
#   grid[2] agent_id
#       -1 indicates no agent is present
#       >=0 indicates an agent exists in that cell (registry slot/id)
# Zone masks intentionally remain off-grid. This keeps special map semantics
# additive and avoids forcing unrelated renderer / engine / serialization changes.


@dataclass(frozen=True)
class HealZone:
    """
    Immutable definition of one base heal-zone patch.

    Fields:
    -------
    zone_id:
        Stable zone identity inside this Zones instance.

    mask:
        Boolean (H, W) tensor for the base geometry of this zone.

    bounds:
        Bounding rectangle in (y0, y1, x0, x1) slice form.
    """

    zone_id: str
    mask: torch.Tensor
    bounds: Tuple[int, int, int, int]


@dataclass(frozen=True)
class HealZoneCatastropheState:
    """
    Transient scheduler-owned suppression state for heal zones.

    Only one catastrophe slot is allowed in Zones at any time. This dataclass is
    intentionally small and explicit so later scheduler/viewer code has a single
    place to reason about catastrophe ownership.
    """

    event_id: str
    suppressed_zone_ids: Tuple[str, ...]


class Zones:
    """
    Special-tile model with an explicit per-zone heal foundation.

    Backward-compatible public contract:
    -----------------------------------
    - `zones.heal_mask` still exists and remains the single effective runtime
      heal mask consumed by old engine/viewer code.
    - `zones.cp_masks` remains a list of per-capture-point masks.
    - `Zones(heal_mask=..., cp_masks=...)` still works for legacy checkpoint and
      runtime paths.

    New internal separation:
    ------------------------
    A) base heal-zone geometry      -> `heal_zones`
    B) manual override layer        -> `manual_zone_enabled`
    C) catastrophe suppression slot -> `catastrophe_state`
    D) derived runtime heal truth   -> `heal_mask` / `effective_heal_mask`

    Manual-vs-catastrophe rule:
    ---------------------------
    Manual API methods clear active catastrophe state by default. This lays the
    architectural foundation for future viewer clicks where manual control must
    win cleanly instead of half-stacking with catastrophe suppression.
    """

    def __init__(
        self,
        heal_mask: Optional[torch.Tensor] = None,
        cp_masks: Optional[Sequence[torch.Tensor]] = None,
        *,
        heal_zones: Optional[Sequence[HealZone]] = None,
        manual_zone_enabled: Optional[Dict[str, bool]] = None,
        catastrophe_state: Optional[HealZoneCatastropheState] = None,
    ) -> None:
        cp_masks = list(cp_masks or [])
        self.cp_masks: List[torch.Tensor] = [
            self._normalize_bool_mask(m, name=f"cp_masks[{i}]")
            for i, m in enumerate(cp_masks)
        ]

        if heal_zones is not None:
            self.heal_zones: List[HealZone] = [self._normalize_heal_zone(z, i) for i, z in enumerate(heal_zones)]
        else:
            self.heal_zones = self._heal_zones_from_legacy_mask(heal_mask)

        if self.heal_zones:
            first_mask = self.heal_zones[0].mask
            self.H, self.W = int(first_mask.shape[0]), int(first_mask.shape[1])
            self.device = first_mask.device
        elif self.cp_masks:
            first_mask = self.cp_masks[0]
            self.H, self.W = int(first_mask.shape[0]), int(first_mask.shape[1])
            self.device = first_mask.device
        elif heal_mask is not None:
            hm = self._normalize_bool_mask(heal_mask, name="heal_mask")
            self.H, self.W = int(hm.shape[0]), int(hm.shape[1])
            self.device = hm.device
        else:
            self.H = self.W = 0
            self.device = getattr(config, "TORCH_DEVICE", torch.device("cpu"))

        self._ensure_shape_consistency()

        self.manual_zone_enabled: Dict[str, bool] = {}
        if manual_zone_enabled:
            for zone_id, enabled in manual_zone_enabled.items():
                zid = str(zone_id)
                if zid not in self._heal_zone_index:
                    raise KeyError(f"manual override references unknown heal zone id: {zid!r}")
                self.manual_zone_enabled[zid] = bool(enabled)

        self.catastrophe_state: Optional[HealZoneCatastropheState] = None
        self._baseline_heal_mask = self._build_baseline_heal_mask()
        self._effective_heal_mask = self._baseline_heal_mask.clone()
        self._active_heal_zone_ids: Tuple[str, ...] = tuple()
        self._runtime_heal_revision: int = 0

        if catastrophe_state is not None:
            self._validate_catastrophe_state(catastrophe_state)
            self.catastrophe_state = catastrophe_state

        self.rebuild_effective_heal_mask()

    # Public compatibility properties
    @property
    def heal_mask(self) -> torch.Tensor:
        """Single derived runtime truth for active heal cells."""
        return self._effective_heal_mask

    @property
    def effective_heal_mask(self) -> torch.Tensor:
        """Alias for the derived runtime heal mask."""
        return self._effective_heal_mask

    @property
    def base_heal_mask(self) -> torch.Tensor:
        """Union of base heal-zone geometry before runtime overrides."""
        return self._baseline_heal_mask

    @property
    def cp_count(self) -> int:
        """Number of distinct capture-point patches."""
        return len(self.cp_masks)

    @property
    def heal_zone_count(self) -> int:
        """Number of individually addressable heal zones."""
        return len(self.heal_zones)

    @property
    def active_heal_zone_ids(self) -> Tuple[str, ...]:
        """Zone ids whose geometry contributes to the current effective mask."""
        return self._active_heal_zone_ids

    @property
    def runtime_heal_revision(self) -> int:
        """Monotonic revision for effective-heal-mask changes only."""
        return int(self._runtime_heal_revision)

    # Public heal-zone API required by catastrophe foundation work
    def rebuild_effective_heal_mask(self) -> torch.Tensor:
        """
        Recompute the effective heal mask from base geometry + runtime state.

        Precedence:
        1) base zone is active by default
        2) catastrophe suppression can deactivate it
        3) manual override wins last if present
        """
        if self.H <= 0 or self.W <= 0:
            self._effective_heal_mask = torch.zeros((0, 0), dtype=torch.bool, device=self.device)
            self._active_heal_zone_ids = tuple()
            return self._effective_heal_mask

        effective = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
        active_zone_ids: List[str] = []
        suppressed = set(self.catastrophe_state.suppressed_zone_ids) if self.catastrophe_state is not None else set()

        for zone in self.heal_zones:
            enabled = zone.zone_id not in suppressed
            manual = self.manual_zone_enabled.get(zone.zone_id)
            if manual is not None:
                enabled = bool(manual)
            if enabled:
                effective |= zone.mask
                active_zone_ids.append(zone.zone_id)

        previous_effective = self._effective_heal_mask
        self._effective_heal_mask = effective
        self._active_heal_zone_ids = tuple(active_zone_ids)
        self._validate_effective_mask()

        if (
            previous_effective is None
            or tuple(previous_effective.shape) != tuple(self._effective_heal_mask.shape)
            or not torch.equal(previous_effective, self._effective_heal_mask)
        ):
            self._runtime_heal_revision += 1

        if not self.manual_zone_enabled and self.catastrophe_state is None:
            if not torch.equal(self._effective_heal_mask, self._baseline_heal_mask):
                raise RuntimeError("baseline heal-mask equivalence failed with no overrides active")

        return self._effective_heal_mask

    def get_heal_zone(self, zone_id: str) -> HealZone:
        """Return one heal-zone definition by id."""
        zid = str(zone_id)
        try:
            return self._heal_zone_index[zid]
        except KeyError as exc:
            raise KeyError(f"unknown heal zone id: {zid!r}") from exc

    def get_heal_zones_containing_cell(
        self,
        x: int,
        y: int,
        *,
        active_only: bool = False,
    ) -> List[HealZone]:
        """Return all heal zones whose masks contain the given cell."""
        self._validate_xy(x, y)
        out: List[HealZone] = []
        active_set = set(self._active_heal_zone_ids) if active_only else None
        for zone in self.heal_zones:
            if active_set is not None and zone.zone_id not in active_set:
                continue
            if bool(zone.mask[y, x].item()):
                out.append(zone)
        return out

    def get_heal_zone_containing_cell(
        self,
        x: int,
        y: int,
        *,
        active_only: bool = False,
    ) -> Optional[HealZone]:
        """
        Return the first matching heal zone for a cell, or None.

        Base heal rectangles may overlap in the legacy baseline. In that case the
        first zone in definition order wins for this singular query helper.
        """
        zones = self.get_heal_zones_containing_cell(x, y, active_only=active_only)
        return zones[0] if zones else None

    def enable_zone_manually(self, zone_id: str, *, clear_catastrophe: bool = True) -> torch.Tensor:
        """Manually force one zone on, optionally clearing active catastrophe state."""
        return self._set_manual_zone_enabled(zone_id, True, clear_catastrophe=clear_catastrophe)

    def disable_zone_manually(self, zone_id: str, *, clear_catastrophe: bool = True) -> torch.Tensor:
        """Manually force one zone off, optionally clearing active catastrophe state."""
        return self._set_manual_zone_enabled(zone_id, False, clear_catastrophe=clear_catastrophe)

    def reset_manual_overrides(self) -> torch.Tensor:
        """Remove all persistent manual overrides and rebuild the effective mask."""
        self.manual_zone_enabled.clear()
        return self.rebuild_effective_heal_mask()

    def apply_catastrophe_suppression_state(
        self,
        suppressed_zone_ids: Sequence[str],
        *,
        event_id: str = "catastrophe",
    ) -> torch.Tensor:
        """
        Apply the single allowed catastrophe suppression slot.

        A second overlapping catastrophe is rejected until the current one is
        explicitly cleared.
        """
        if self.catastrophe_state is not None:
            raise RuntimeError(
                "catastrophe suppression already active; clear current catastrophe state before applying another"
            )

        normalized_ids = tuple(str(zid) for zid in suppressed_zone_ids)
        state = HealZoneCatastropheState(event_id=str(event_id), suppressed_zone_ids=normalized_ids)
        self._validate_catastrophe_state(state)
        self.catastrophe_state = state
        return self.rebuild_effective_heal_mask()

    def clear_catastrophe_suppression_state(self) -> torch.Tensor:
        """Clear the transient catastrophe suppression layer and rebuild."""
        self.catastrophe_state = None
        return self.rebuild_effective_heal_mask()

    def clear_current_catastrophe_state(self) -> torch.Tensor:
        """Alias kept explicit for future viewer/scheduler control paths."""
        return self.clear_catastrophe_suppression_state()

    def restore_all_zones_to_normal_effective_state(self) -> torch.Tensor:
        """Return to baseline by clearing manual overrides and catastrophe state."""
        self.manual_zone_enabled.clear()
        self.catastrophe_state = None
        return self.rebuild_effective_heal_mask()

    def to_checkpoint_payload(self) -> Dict[str, object]:
        """
        Build a backward-friendly serializable payload for future checkpoint work.

        The legacy `heal_mask` key is intentionally retained so older loading
        logic can still recover baseline behavior even if it ignores the richer
        per-zone fields.
        """
        catastrophe_payload: Optional[Dict[str, object]] = None
        if self.catastrophe_state is not None:
            catastrophe_payload = {
                "event_id": self.catastrophe_state.event_id,
                "suppressed_zone_ids": list(self.catastrophe_state.suppressed_zone_ids),
            }

        return {
            "heal_mask": self.base_heal_mask.detach().clone(),
            "effective_heal_mask": self.heal_mask.detach().clone(),
            "heal_zones": [
                {
                    "zone_id": zone.zone_id,
                    "mask": zone.mask.detach().clone(),
                    "bounds": tuple(int(v) for v in zone.bounds),
                }
                for zone in self.heal_zones
            ],
            "manual_zone_enabled": dict(self.manual_zone_enabled),
            "catastrophe_state": catastrophe_payload,
            "cp_masks": [m.detach().clone() for m in self.cp_masks],
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: Dict[str, object], *, device: Optional[torch.device] = None) -> "Zones":
        """Construct Zones from either a rich or legacy checkpoint-style payload."""
        if payload is None:
            return cls(heal_mask=None, cp_masks=[])

        device = device or getattr(config, "TORCH_DEVICE", torch.device("cpu"))
        cp_masks = [torch.as_tensor(m, device=device).bool() for m in payload.get("cp_masks", [])]

        heal_zone_payloads = payload.get("heal_zones") or []
        heal_zones: List[HealZone] = []
        for idx, zone_payload in enumerate(heal_zone_payloads):
            if not isinstance(zone_payload, dict):
                raise TypeError(f"heal_zones[{idx}] payload must be a dict")
            heal_zones.append(
                HealZone(
                    zone_id=str(zone_payload["zone_id"]),
                    mask=torch.as_tensor(zone_payload["mask"], device=device).bool(),
                    bounds=tuple(int(v) for v in zone_payload["bounds"]),
                )
            )

        catastrophe_payload = payload.get("catastrophe_state")
        catastrophe_state = None
        if catastrophe_payload is not None:
            if not isinstance(catastrophe_payload, dict):
                raise TypeError("catastrophe_state payload must be a dict or None")
            suppressed_zone_ids = tuple(str(z) for z in catastrophe_payload.get("suppressed_zone_ids", []))
            if suppressed_zone_ids:
                catastrophe_state = HealZoneCatastropheState(
                    event_id=str(catastrophe_payload["event_id"]),
                    suppressed_zone_ids=suppressed_zone_ids,
                )

        manual_zone_enabled = {str(k): bool(v) for k, v in dict(payload.get("manual_zone_enabled", {})).items()}

        if heal_zones:
            return cls(
                heal_zones=heal_zones,
                cp_masks=cp_masks,
                manual_zone_enabled=manual_zone_enabled,
                catastrophe_state=catastrophe_state,
            )

        heal_mask = payload.get("heal_mask")
        if heal_mask is None:
            return cls(heal_mask=None, cp_masks=cp_masks)

        return cls(
            heal_mask=torch.as_tensor(heal_mask, device=device).bool(),
            cp_masks=cp_masks,
        )

    # Internal helpers
    def _set_manual_zone_enabled(
        self,
        zone_id: str,
        enabled: bool,
        *,
        clear_catastrophe: bool,
    ) -> torch.Tensor:
        zid = str(zone_id)
        self.get_heal_zone(zid)  # validate existence
        if clear_catastrophe and self.catastrophe_state is not None:
            self.catastrophe_state = None
        self.manual_zone_enabled[zid] = bool(enabled)
        return self.rebuild_effective_heal_mask()

    def _normalize_heal_zone(self, zone: HealZone, idx: int) -> HealZone:
        if not isinstance(zone, HealZone):
            raise TypeError(f"heal_zones[{idx}] must be HealZone, got {type(zone).__name__}")
        zone_id = str(zone.zone_id)
        mask = self._normalize_bool_mask(zone.mask, name=f"heal_zones[{idx}].mask")
        if int(mask.to(torch.int64).sum().item()) <= 0:
            raise ValueError(f"heal_zones[{idx}] must contain at least one active cell")
        bounds = self._bounds_from_mask(mask) if zone.bounds is None else tuple(int(v) for v in zone.bounds)
        self._validate_bounds(bounds, mask.shape)
        return HealZone(zone_id=zone_id, mask=mask, bounds=bounds)

    def _normalize_bool_mask(self, mask: Optional[torch.Tensor], *, name: str) -> torch.Tensor:
        if mask is None:
            raise ValueError(f"{name} cannot be None")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if mask.ndim != 2:
            raise ValueError(f"{name} must have shape (H, W), got {tuple(mask.shape)}")
        return mask.bool()

    def _ensure_shape_consistency(self) -> None:
        zone_index: Dict[str, HealZone] = {}
        for i, zone in enumerate(self.heal_zones):
            if zone.zone_id in zone_index:
                raise ValueError(f"duplicate heal zone id detected: {zone.zone_id!r}")
            if i == 0:
                self.H, self.W = int(zone.mask.shape[0]), int(zone.mask.shape[1])
                self.device = zone.mask.device
            else:
                if tuple(zone.mask.shape) != (self.H, self.W):
                    raise ValueError(
                        f"heal zone {zone.zone_id!r} has shape {tuple(zone.mask.shape)} != ({self.H}, {self.W})"
                    )
                if zone.mask.device != self.device:
                    raise ValueError(
                        f"heal zone {zone.zone_id!r} device {zone.mask.device} != expected {self.device}"
                    )
            zone_index[zone.zone_id] = zone

        for i, mask in enumerate(self.cp_masks):
            if self.heal_zones:
                if tuple(mask.shape) != (self.H, self.W):
                    raise ValueError(f"cp_masks[{i}] has shape {tuple(mask.shape)} != ({self.H}, {self.W})")
                if mask.device != self.device:
                    raise ValueError(f"cp_masks[{i}] device {mask.device} != expected {self.device}")
            elif i == 0:
                self.H, self.W = int(mask.shape[0]), int(mask.shape[1])
                self.device = mask.device

        self._heal_zone_index = zone_index

    def _build_baseline_heal_mask(self) -> torch.Tensor:
        if self.H <= 0 or self.W <= 0:
            return torch.zeros((0, 0), dtype=torch.bool, device=self.device)
        base = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
        for zone in self.heal_zones:
            base |= zone.mask
        return base

    def _validate_effective_mask(self) -> None:
        if self._effective_heal_mask.dtype != torch.bool:
            raise RuntimeError("effective heal mask must have dtype torch.bool")
        if tuple(self._effective_heal_mask.shape) != (self.H, self.W):
            raise RuntimeError(
                f"effective heal mask shape mismatch: got {tuple(self._effective_heal_mask.shape)} expected ({self.H}, {self.W})"
            )
        if self._effective_heal_mask.device != self.device:
            raise RuntimeError(
                f"effective heal mask device mismatch: got {self._effective_heal_mask.device} expected {self.device}"
            )

    def _validate_catastrophe_state(self, state: HealZoneCatastropheState) -> None:
        if not isinstance(state, HealZoneCatastropheState):
            raise TypeError("catastrophe_state must be HealZoneCatastropheState")
        if not state.suppressed_zone_ids:
            raise ValueError("catastrophe suppression must disable at least one heal zone")
        seen = set()
        for zone_id in state.suppressed_zone_ids:
            zid = str(zone_id)
            if zid in seen:
                raise ValueError(f"catastrophe suppression repeats heal zone id: {zid!r}")
            if zid not in self._heal_zone_index:
                raise KeyError(f"catastrophe suppression references unknown heal zone id: {zid!r}")
            seen.add(zid)
        if self.heal_zone_count > 0 and len(seen) >= self.heal_zone_count:
            raise ValueError("catastrophe suppression cannot disable every heal zone")

    def _heal_zones_from_legacy_mask(self, heal_mask: Optional[torch.Tensor]) -> List[HealZone]:
        if heal_mask is None:
            return []
        mask = self._normalize_bool_mask(heal_mask, name="heal_mask")
        if int(mask.sum().item()) == 0:
            return []

        components = self._connected_components(mask)
        zones: List[HealZone] = []
        for i, comp in enumerate(components):
            bounds = self._bounds_from_mask(comp)
            zones.append(HealZone(zone_id=f"heal_{i:03d}", mask=comp, bounds=bounds))
        return zones

    def _connected_components(self, mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Split a legacy merged heal mask into synthetic per-component zones.

        This path is used only for backward compatibility with older code/checkpoints
        that stored a single merged heal mask. It does not claim to recover the exact
        original procedural rectangles if they had overlapped.
        """
        mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
        H, W = int(mask_cpu.shape[0]), int(mask_cpu.shape[1])
        visited = torch.zeros((H, W), dtype=torch.bool)
        components: List[torch.Tensor] = []

        for y in range(H):
            for x in range(W):
                if visited[y, x] or not bool(mask_cpu[y, x].item()):
                    continue

                stack = [(x, y)]
                visited[y, x] = True
                coords: List[Tuple[int, int]] = []

                while stack:
                    cx, cy = stack.pop()
                    coords.append((cx, cy))
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx] and bool(mask_cpu[ny, nx].item()):
                            visited[ny, nx] = True
                            stack.append((nx, ny))

                comp = torch.zeros((H, W), dtype=torch.bool)
                xs = [xy[0] for xy in coords]
                ys = [xy[1] for xy in coords]
                comp[ys, xs] = True
                components.append(comp.to(device=mask.device))

        return components

    @staticmethod
    def _bounds_from_mask(mask: torch.Tensor) -> Tuple[int, int, int, int]:
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if xs.numel() == 0:
            return (0, 0, 0, 0)
        return (
            int(ys.min().item()),
            int(ys.max().item()) + 1,
            int(xs.min().item()),
            int(xs.max().item()) + 1,
        )

    @staticmethod
    def _validate_bounds(bounds: Tuple[int, int, int, int], shape: Sequence[int]) -> None:
        H, W = int(shape[0]), int(shape[1])
        y0, y1, x0, x1 = (int(v) for v in bounds)
        if not (0 <= y0 <= y1 <= H and 0 <= x0 <= x1 <= W):
            raise ValueError(
                f"invalid heal-zone bounds {bounds}; expected within shape ({H}, {W})"
            )

    def _validate_xy(self, x: int, y: int) -> None:
        if not (0 <= int(x) < self.W and 0 <= int(y) < self.H):
            raise IndexError(f"cell ({x}, {y}) is outside heal grid bounds ({self.W}, {self.H})")


# Random thin gray walls (1-cell thick, meandering segments)
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

    assert grid.ndim == 3 and grid.size(0) >= 3, "grid must be (3,H,W)"
    occ = grid[0]
    H, W = int(occ.size(0)), int(occ.size(1))

    dirs8 = torch.tensor(
        [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]],
        dtype=torch.long,
        device=occ.device,
    )

    def _place_wall_cell(x: int, y: int) -> None:
        if 0 <= x < W and 0 <= y < H:
            if not allow_over_agents:
                v = float(occ[y, x].item())
                if v in (2.0, 3.0):
                    return
            occ[y, x] = 1.0
            grid[1, y, x] = 0.0
            grid[2, y, x] = -1.0

    x0_min, x0_max = max(1, avoid_margin), W - max(1, avoid_margin) - 1
    y0_min, y0_max = max(1, avoid_margin), H - max(1, avoid_margin) - 1
    if x0_min >= x0_max or y0_min >= y0_max or n_segments <= 0:
        return

    for _ in range(max(0, int(n_segments))):
        x = random.randint(x0_min, x0_max)
        y = random.randint(y0_min, y0_max)
        L = random.randint(max(1, int(seg_min)), max(1, int(seg_max)))
        _place_wall_cell(x, y)
        last_dir = random.randrange(8)

        for _step in range(L):
            if random.random() < 0.70:
                d = last_dir
            else:
                d = (last_dir + random.choice([-2, -1, 1, 2])) % 8
            last_dir = d

            dx, dy = int(dirs8[d, 0].item()), int(dirs8[d, 1].item())
            x = max(1, min(W - 2, x + dx))
            y = max(1, min(H - 2, y + dy))
            _place_wall_cell(x, y)

            if random.random() < 0.05:
                pass


# Heal & Capture zones (rectangular patches, scaled to grid)
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
    Create explicit heal-zone definitions plus capture-zone masks.

    Baseline behavior is preserved by deriving one effective heal mask from the
    union of all active base heal zones. The base generator still uses simple
    rectangles and still allows overlap exactly like the old implementation.
    """
    device = device or config.TORCH_DEVICE
    cp_masks: List[torch.Tensor] = []
    heal_zones: List[HealZone] = []

    def _sample_rect(h_side: int, w_side: int) -> Tuple[int, int, int, int]:
        x0 = random.randint(1, max(1, W - w_side - 2))
        y0 = random.randint(1, max(1, H - h_side - 2))
        return y0, y0 + h_side, x0, x0 + w_side

    if heal_count > 0 and heal_ratio > 0.0:
        h_side = max(1, int(round(heal_ratio * H)))
        w_side = max(1, int(round(heal_ratio * W)))

        for idx in range(int(heal_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)
            mask = torch.zeros((H, W), dtype=torch.bool, device=device)
            mask[y0:y1, x0:x1] = True
            heal_zones.append(
                HealZone(
                    zone_id=f"heal_{idx:03d}",
                    mask=mask,
                    bounds=(y0, y1, x0, x1),
                )
            )

    if cp_count > 0 and cp_ratio > 0.0:
        h_side = max(1, int(round(cp_ratio * H)))
        w_side = max(1, int(round(cp_ratio * W)))

        for _ in range(int(cp_count)):
            y0, y1, x0, x1 = _sample_rect(h_side, w_side)
            m = torch.zeros((H, W), dtype=torch.bool, device=device)
            m[y0:y1, x0:x1] = True
            cp_masks.append(m)

    return Zones(heal_zones=heal_zones, cp_masks=cp_masks)
