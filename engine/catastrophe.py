from __future__ import annotations

"""
Catastrophe runtime framework for Neural-Abyss.

Design goals
------------
This module introduces a *runtime* catastrophe layer that sits on top of the
canonical signed base-zone map without mutating it.

Core separation of concerns:
- base-zone state lives in `engine.mapgen.Zones.base_zone_value_map`
- catastrophe state lives here as transient runtime state
- the effective zone field is derived from base + override + apply mask
- manual viewer edits remain targeted at the canonical base layer

This module intentionally does NOT implement a full catastrophe pack.
It only provides the framework needed for fixed-duration runtime overrides so a
later patch can add concrete catastrophe generators / manual triggers cleanly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


GridShape = Tuple[int, int]


def _normalize_grid_shape(shape: GridShape) -> GridShape:
    """Return a validated `(H, W)` grid shape tuple."""
    if len(tuple(shape)) != 2:
        raise ValueError(f"grid shape must be length-2, got {shape!r}")
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"grid shape must be positive, got {(h, w)}")
    return h, w


def _normalize_float_grid(t: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate one rank-2 float grid and clamp it into [-1, +1]."""
    if not torch.is_tensor(t):
        raise TypeError(f"{name} must be a torch.Tensor")
    if int(t.ndim) != 2:
        raise ValueError(f"{name} must be rank-2 (H,W), got shape={tuple(t.shape)}")
    return t.to(dtype=torch.float32).clamp(-1.0, 1.0)


def _normalize_bool_grid(t: torch.Tensor, *, name: str, shape: GridShape) -> torch.Tensor:
    """Validate one rank-2 boolean grid against the expected world shape."""
    if not torch.is_tensor(t):
        raise TypeError(f"{name} must be a torch.Tensor")
    if int(t.ndim) != 2:
        raise ValueError(f"{name} must be rank-2 (H,W), got shape={tuple(t.shape)}")
    if tuple(int(v) for v in t.shape) != tuple(int(v) for v in shape):
        raise ValueError(
            f"{name} shape {tuple(int(v) for v in t.shape)} does not match expected shape {tuple(int(v) for v in shape)}"
        )
    return t.to(dtype=torch.bool)


def _clone_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a detached shallow copy of user-provided metadata."""
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict when provided, got {type(metadata).__name__}")
    return dict(metadata)


@dataclass
class CatastropheSpec:
    """
    Pure catastrophe definition used to *activate* a runtime catastrophe.

    One spec defines:
    - catastrophe type name
    - duration in ticks
    - override value map (signed float grid)
    - apply mask selecting where the override replaces base values
    - edit-lock mask for manual base-zone editing while active
    - optional metadata for debugging / later UI / later catastrophe packs
    """

    type_name: str
    duration_ticks: int
    override_value_map: torch.Tensor
    apply_mask: torch.Tensor
    edit_lock_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        type_name = str(self.type_name).strip()
        if not type_name:
            raise ValueError("CatastropheSpec.type_name must be a non-empty string")
        self.type_name = type_name

        duration_ticks = int(self.duration_ticks)
        if duration_ticks <= 0:
            raise ValueError(f"CatastropheSpec.duration_ticks must be > 0, got {duration_ticks}")
        self.duration_ticks = duration_ticks

        self.override_value_map = _normalize_float_grid(self.override_value_map, name="override_value_map")
        shape = tuple(int(v) for v in self.override_value_map.shape)
        self.apply_mask = _normalize_bool_grid(self.apply_mask, name="apply_mask", shape=shape)

        if self.edit_lock_mask is None:
            self.edit_lock_mask = self.apply_mask.clone()
        else:
            self.edit_lock_mask = _normalize_bool_grid(
                self.edit_lock_mask,
                name="edit_lock_mask",
                shape=shape,
            )

        self.metadata = _clone_metadata(self.metadata)

    @property
    def shape(self) -> GridShape:
        """Return `(H, W)` for the catastrophe grids."""
        return tuple(int(v) for v in self.override_value_map.shape)

    @property
    def active_cell_count(self) -> int:
        """Return how many cells are covered by the override apply mask."""
        return int(self.apply_mask.sum().item())

    @property
    def locked_cell_count(self) -> int:
        """Return how many cells are locked against base-zone editing."""
        return int(self.edit_lock_mask.sum().item())

    def to_device(self, device: torch.device) -> "CatastropheSpec":
        """Return a copy moved onto `device`."""
        return CatastropheSpec(
            type_name=self.type_name,
            duration_ticks=self.duration_ticks,
            override_value_map=self.override_value_map.to(device),
            apply_mask=self.apply_mask.to(device),
            edit_lock_mask=self.edit_lock_mask.to(device) if self.edit_lock_mask is not None else None,
            metadata=_clone_metadata(self.metadata),
        )

    def summary_payload(self) -> Dict[str, Any]:
        """Return a small JSON-friendly summary for logs / manifests."""
        return {
            "type_name": str(self.type_name),
            "duration_ticks": int(self.duration_ticks),
            "shape": [int(v) for v in self.shape],
            "active_cell_count": int(self.active_cell_count),
            "locked_cell_count": int(self.locked_cell_count),
            "metadata": _clone_metadata(self.metadata),
        }


@dataclass
class ActiveCatastropheState:
    """
    Concrete active catastrophe runtime state.

    This is the serializable/resumable lifecycle-bearing form.
    """

    type_name: str
    start_tick: int
    duration_ticks: int
    remaining_ticks: int
    override_value_map: torch.Tensor
    apply_mask: torch.Tensor
    edit_lock_mask: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        type_name = str(self.type_name).strip()
        if not type_name:
            raise ValueError("ActiveCatastropheState.type_name must be a non-empty string")
        self.type_name = type_name

        self.start_tick = int(self.start_tick)
        self.duration_ticks = int(self.duration_ticks)
        self.remaining_ticks = int(self.remaining_ticks)
        if self.duration_ticks <= 0:
            raise ValueError(f"duration_ticks must be > 0, got {self.duration_ticks}")
        if self.remaining_ticks <= 0:
            raise ValueError(f"remaining_ticks must be > 0 while active, got {self.remaining_ticks}")
        if self.remaining_ticks > self.duration_ticks:
            raise ValueError(
                f"remaining_ticks ({self.remaining_ticks}) cannot exceed duration_ticks ({self.duration_ticks})"
            )

        self.override_value_map = _normalize_float_grid(self.override_value_map, name="override_value_map")
        shape = tuple(int(v) for v in self.override_value_map.shape)
        self.apply_mask = _normalize_bool_grid(self.apply_mask, name="apply_mask", shape=shape)
        self.edit_lock_mask = _normalize_bool_grid(self.edit_lock_mask, name="edit_lock_mask", shape=shape)
        self.metadata = _clone_metadata(self.metadata)

    @classmethod
    def from_spec(cls, spec: CatastropheSpec, *, start_tick: int, device: torch.device) -> "ActiveCatastropheState":
        spec_dev = spec.to_device(device)
        return cls(
            type_name=spec_dev.type_name,
            start_tick=int(start_tick),
            duration_ticks=int(spec_dev.duration_ticks),
            remaining_ticks=int(spec_dev.duration_ticks),
            override_value_map=spec_dev.override_value_map,
            apply_mask=spec_dev.apply_mask,
            edit_lock_mask=spec_dev.edit_lock_mask if spec_dev.edit_lock_mask is not None else spec_dev.apply_mask,
            metadata=_clone_metadata(spec_dev.metadata),
        )

    @property
    def shape(self) -> GridShape:
        return tuple(int(v) for v in self.override_value_map.shape)

    @property
    def active_cell_count(self) -> int:
        return int(self.apply_mask.sum().item())

    @property
    def locked_cell_count(self) -> int:
        return int(self.edit_lock_mask.sum().item())

    @property
    def elapsed_ticks(self) -> int:
        return int(self.duration_ticks - self.remaining_ticks)

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "type_name": str(self.type_name),
            "start_tick": int(self.start_tick),
            "duration_ticks": int(self.duration_ticks),
            "remaining_ticks": int(self.remaining_ticks),
            "override_value_map": self.override_value_map,
            "apply_mask": self.apply_mask,
            "edit_lock_mask": self.edit_lock_mask,
            "metadata": _clone_metadata(self.metadata),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: Dict[str, Any], *, device: torch.device) -> "ActiveCatastropheState":
        if not isinstance(payload, dict):
            raise TypeError(f"active catastrophe payload must be a dict, got {type(payload).__name__}")
        return cls(
            type_name=payload["type_name"],
            start_tick=int(payload["start_tick"]),
            duration_ticks=int(payload["duration_ticks"]),
            remaining_ticks=int(payload["remaining_ticks"]),
            override_value_map=payload["override_value_map"].to(device),
            apply_mask=payload["apply_mask"].to(device),
            edit_lock_mask=payload["edit_lock_mask"].to(device),
            metadata=_clone_metadata(payload.get("metadata", {})),
        )

    def summary_payload(self) -> Dict[str, Any]:
        return {
            "active": True,
            "type_name": str(self.type_name),
            "start_tick": int(self.start_tick),
            "duration_ticks": int(self.duration_ticks),
            "remaining_ticks": int(self.remaining_ticks),
            "elapsed_ticks": int(self.elapsed_ticks),
            "shape": [int(v) for v in self.shape],
            "active_cell_count": int(self.active_cell_count),
            "locked_cell_count": int(self.locked_cell_count),
            "metadata": _clone_metadata(self.metadata),
        }


class CatastropheController:
    """Fixed-duration catastrophe lifecycle controller."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = torch.device(device)
        self._active: Optional[ActiveCatastropheState] = None

    def is_active(self) -> bool:
        return self._active is not None

    @property
    def active_state(self) -> Optional[ActiveCatastropheState]:
        return self._active

    def summary_payload(self) -> Dict[str, Any]:
        if self._active is None:
            return {
                "active": False,
                "type_name": None,
                "start_tick": None,
                "duration_ticks": 0,
                "remaining_ticks": 0,
                "elapsed_ticks": 0,
                "shape": None,
                "active_cell_count": 0,
                "locked_cell_count": 0,
                "metadata": {},
            }
        return self._active.summary_payload()

    def activate(
        self,
        spec: CatastropheSpec,
        *,
        tick: int,
        world_shape: GridShape,
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        world_shape = _normalize_grid_shape(world_shape)
        if tuple(int(v) for v in spec.shape) != tuple(int(v) for v in world_shape):
            raise ValueError(
                f"catastrophe spec shape {tuple(spec.shape)} does not match world shape {tuple(world_shape)}"
            )
        if self._active is not None and not bool(replace_existing):
            raise RuntimeError(
                f"catastrophe {self._active.type_name!r} is already active; replace_existing=False blocks replacement"
            )
        self._active = ActiveCatastropheState.from_spec(spec, start_tick=int(tick), device=self.device)
        return {"phase": "activated", **self._active.summary_payload()}

    def clear(self, *, tick: int, reason: str = "cleared") -> Optional[Dict[str, Any]]:
        if self._active is None:
            return None
        payload = {"phase": "ended", "end_tick": int(tick), "reason": str(reason), **self._active.summary_payload()}
        self._active = None
        return payload

    def on_tick_end(self, tick: int) -> Optional[Dict[str, Any]]:
        if self._active is None:
            return None
        self._active.remaining_ticks -= 1
        if int(self._active.remaining_ticks) > 0:
            return None
        return self.clear(tick=int(tick), reason="duration_exhausted")

    def resolve_effective_layers(self, base_zone_value_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        base = _normalize_float_grid(base_zone_value_map, name="base_zone_value_map").to(self.device)
        shape = tuple(int(v) for v in base.shape)
        if self._active is None:
            false_mask = torch.zeros(shape, device=self.device, dtype=torch.bool)
            zero_override = torch.zeros(shape, device=self.device, dtype=torch.float32)
            return {"base": base, "effective": base, "override": zero_override, "apply_mask": false_mask, "edit_lock_mask": false_mask}
        if tuple(int(v) for v in self._active.shape) != shape:
            raise ValueError(
                f"active catastrophe shape {tuple(self._active.shape)} does not match base zone shape {shape}"
            )
        effective = base.clone()
        if bool(self._active.apply_mask.any().item()):
            effective[self._active.apply_mask] = self._active.override_value_map[self._active.apply_mask]
        effective = effective.clamp(-1.0, 1.0)
        return {
            "base": base,
            "effective": effective,
            "override": self._active.override_value_map,
            "apply_mask": self._active.apply_mask,
            "edit_lock_mask": self._active.edit_lock_mask,
        }

    def checkpoint_payload(self) -> Dict[str, Any]:
        if self._active is None:
            return {"active": False}
        return {"active": True, "active_state": self._active.checkpoint_payload()}

    def load_checkpoint_payload(self, payload: Optional[Dict[str, Any]], *, device: torch.device) -> None:
        self.device = torch.device(device)
        if not payload:
            self._active = None
            return
        if not isinstance(payload, dict):
            raise TypeError(f"catastrophe checkpoint payload must be a dict, got {type(payload).__name__}")
        if not bool(payload.get("active", False)):
            self._active = None
            return
        active_payload = payload.get("active_state", None)
        if not isinstance(active_payload, dict):
            raise ValueError("active catastrophe checkpoint payload is missing active_state")
        self._active = ActiveCatastropheState.from_checkpoint_payload(active_payload, device=self.device)
