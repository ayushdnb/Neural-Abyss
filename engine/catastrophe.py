from __future__ import annotations

"""
Catastrophe runtime framework for Neural-Abyss.

Design goals
------------
This module implements a transient catastrophe layer that sits on top of the
canonical signed base-zone field without mutating it.

Core separation of concerns:
- canonical persistent zone state lives in `engine.mapgen.Zones.base_zone_value_map`
- catastrophe state lives here as transient runtime state
- the effective zone field is derived from base + runtime override + apply mask
- viewer edits always target the canonical base layer, never the runtime layer

Patch rationale
---------------
The repo already had a correct foundation for:
- signed canonical base zones
- runtime catastrophe overrides
- fixed-duration active catastrophe persistence

The missing reconciliation work completed here is:
- an explainable catastrophe pack that is safe by default
- a deterministic dynamic trigger law with its own checkpointed RNG/state
- richer catastrophe/scheduler summaries for viewer + telemetry + checkpoints
- clean separation between safe default catastrophes and experimental ones
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import config


GridShape = Tuple[int, int]

_MANUAL_CATASTROPHE_EPS = 1e-6


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


def _resolve_duration_ticks(duration_ticks: Optional[int], *, key_hint: Optional[str] = None) -> int:
    """Resolve catastrophe duration using explicit value first, then config."""
    if duration_ticks is not None:
        duration = int(duration_ticks)
    else:
        durations = dict(getattr(config, "CATASTROPHE_PRESET_DURATION_TICKS", {}) or {})
        if key_hint is not None and key_hint in durations:
            duration = int(durations[key_hint])
        else:
            duration = int(getattr(config, "CATASTROPHE_DEFAULT_DURATION_TICKS", 0))
    if duration <= 0:
        raise ValueError(
            "catastrophe duration must be > 0; set duration_ticks explicitly or configure "
            "CATASTROPHE_DEFAULT_DURATION_TICKS / CATASTROPHE_PRESET_DURATION_TICKS"
        )
    return duration


def _canonical_base_zone_value_map(base_zone_value_map: torch.Tensor) -> torch.Tensor:
    """Normalize the canonical signed base-zone field used to derive catastrophes."""
    return _normalize_float_grid(base_zone_value_map, name="base_zone_value_map")


def _active_nonzero_base_mask(base_zone_value_map: torch.Tensor, *, eps: float = _MANUAL_CATASTROPHE_EPS) -> torch.Tensor:
    """Return a mask of cells whose canonical base-zone value is meaningfully non-zero."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    return base.abs() > float(eps)


def _positive_base_mask(base_zone_value_map: torch.Tensor, *, eps: float = _MANUAL_CATASTROPHE_EPS) -> torch.Tensor:
    base = _canonical_base_zone_value_map(base_zone_value_map)
    return base > float(eps)


def _negative_base_mask(base_zone_value_map: torch.Tensor, *, eps: float = _MANUAL_CATASTROPHE_EPS) -> torch.Tensor:
    base = _canonical_base_zone_value_map(base_zone_value_map)
    return base < -float(eps)


def _full_world_edit_lock_mask(base_zone_value_map: torch.Tensor) -> torch.Tensor:
    """Lock the whole canonical base layer while a catastrophe is active."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    return torch.ones_like(base, dtype=torch.bool)


def _regional_half_mask(*, shape: GridShape, device: torch.device, region: str) -> torch.Tensor:
    """Return a deterministic half-world boolean mask for one supported region label."""
    h, w = _normalize_grid_shape(shape)
    region_key = str(region).strip().lower()
    mask = torch.zeros((h, w), device=device, dtype=torch.bool)

    if region_key == "left":
        mask[:, : max(1, w // 2)] = True
        return mask
    if region_key == "right":
        mask[:, w // 2 :] = True
        return mask
    if region_key == "top":
        mask[: max(1, h // 2), :] = True
        return mask
    if region_key == "bottom":
        mask[h // 2 :, :] = True
        return mask

    raise ValueError(f"unsupported regional catastrophe region: {region!r}")


def _band_mask(*, shape: GridShape, axis: str, center_index: int, band_width: int, device: torch.device) -> torch.Tensor:
    """Return one contiguous horizontal/vertical band mask."""
    h, w = _normalize_grid_shape(shape)
    axis_key = str(axis).strip().lower()
    mask = torch.zeros((h, w), device=device, dtype=torch.bool)
    if axis_key not in {"vertical", "horizontal"}:
        raise ValueError(f"unsupported band axis: {axis!r}")

    if axis_key == "vertical":
        dim = w
        center = max(0, min(w - 1, int(center_index)))
        half = max(0, int(band_width) // 2)
        start = max(0, center - half)
        end = min(w, start + max(1, int(band_width)))
        start = max(0, end - max(1, int(band_width)))
        mask[:, start:end] = True
        return mask

    dim = h
    center = max(0, min(h - 1, int(center_index)))
    half = max(0, int(band_width) // 2)
    start = max(0, center - half)
    end = min(h, start + max(1, int(band_width)))
    start = max(0, end - max(1, int(band_width)))
    mask[start:end, :] = True
    return mask


def _generator_rand_index(gen: torch.Generator, upper_exclusive: int) -> int:
    upper = max(1, int(upper_exclusive))
    return int(torch.randint(0, upper, (1,), generator=gen, device="cpu").item())


def _generator_rand_float(gen: torch.Generator) -> float:
    return float(torch.rand((1,), generator=gen, device="cpu").item())


def _weighted_choice_index(weights: List[float], gen: torch.Generator) -> int:
    if len(weights) == 0:
        raise ValueError("weighted choice requires at least one weight")
    total = float(sum(max(0.0, float(w)) for w in weights))
    if total <= 0.0:
        raise ValueError("weighted choice requires positive total weight")
    threshold = _generator_rand_float(gen) * total
    running = 0.0
    for idx, w in enumerate(weights):
        running += max(0.0, float(w))
        if threshold <= running:
            return idx
    return len(weights) - 1


@dataclass
class CatastropheSpec:
    """
    Pure catastrophe definition used to activate a runtime catastrophe.

    One spec defines:
    - catastrophe type name
    - duration in ticks
    - override value map (signed float grid)
    - apply mask selecting where the override replaces base values
    - edit-lock mask for manual base-zone editing while active
    - optional metadata for debugging / UI / telemetry / scheduler forensics
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
        return tuple(int(v) for v in self.override_value_map.shape)

    @property
    def active_cell_count(self) -> int:
        return int(self.apply_mask.sum().item())

    @property
    def locked_cell_count(self) -> int:
        return int(self.edit_lock_mask.sum().item())

    def to_device(self, device: torch.device) -> "CatastropheSpec":
        return CatastropheSpec(
            type_name=self.type_name,
            duration_ticks=self.duration_ticks,
            override_value_map=self.override_value_map.to(device),
            apply_mask=self.apply_mask.to(device),
            edit_lock_mask=self.edit_lock_mask.to(device) if self.edit_lock_mask is not None else None,
            metadata=_clone_metadata(self.metadata),
        )

    def summary_payload(self) -> Dict[str, Any]:
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
    """Concrete active catastrophe runtime state."""

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


class DynamicCatastropheScheduler:
    """
    Deterministic hazard-law catastrophe scheduler.

    Design properties:
    - own checkpointed RNG state
    - pressure accumulation rather than dumb fixed timers
    - cooldown + min/max interval guard rails
    - weighted catastrophe type selection
    - explainable status payload for viewer / telemetry / checkpoints
    - one-active-catastrophe-at-a-time by default
    """

    version: str = "pressure_hazard_v1"

    def __init__(self, *, device: torch.device, seed: Optional[int] = None) -> None:
        self.device = torch.device(device)
        base_seed = int(getattr(config, "RNG_SEED", getattr(config, "SEED", 42)) if seed is None else seed)
        seed_offset = int(getattr(config, "CATASTROPHE_SCHEDULER_SEED_OFFSET", 7919))
        self.seed = int(base_seed + seed_offset)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

        # NOTE: `enabled` here means the auto-scheduler preference only.
        # The master catastrophe on/off state lives on CatastropheController.
        self.enabled: bool = bool(getattr(config, "CATASTROPHE_DYNAMIC_SCHEDULER_ENABLED", True))
        self.pressure: float = 0.0
        self.cooldown_remaining: int = 0
        self.ticks_since_last_end: int = int(getattr(config, "CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS", 0) or 0)
        self.last_roll: Optional[float] = None
        self.last_threshold: float = 0.0
        self.last_pressure_delta: float = 0.0
        self.last_trigger_tick: Optional[int] = None
        self.last_end_tick: Optional[int] = None
        self.last_selected_key: Optional[str] = None
        self.last_trigger_mode: Optional[str] = None
        self.last_state_label: str = "boot"
        self.last_hard_trigger_due: bool = False

    @property
    def generator(self) -> torch.Generator:
        return self._rng

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.last_state_label = "enabled" if self.enabled else "disabled"

    def _pressure_increment(self, base_zone_value_map: torch.Tensor) -> float:
        base = _canonical_base_zone_value_map(base_zone_value_map)
        total = float(base.numel() or 1)
        positive_frac = float((base > _MANUAL_CATASTROPHE_EPS).sum().item()) / total
        negative_frac = float((base < -_MANUAL_CATASTROPHE_EPS).sum().item()) / total
        nonzero_frac = float((base.abs() > _MANUAL_CATASTROPHE_EPS).sum().item()) / total
        mean_abs = float(base.abs().mean().item())
        return (
            float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_BASE_PER_TICK", 0.0))
            + nonzero_frac * float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_NONZERO_GAIN", 0.0))
            + positive_frac * float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_POSITIVE_GAIN", 0.0))
            + negative_frac * float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_NEGATIVE_GAIN", 0.0))
            + mean_abs * float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_MEAN_ABS_GAIN", 0.0))
        )

    def _hazard_probability(self) -> float:
        base_prob = float(getattr(config, "CATASTROPHE_DYNAMIC_BASE_HAZARD_PROB", 0.0))
        gain = float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_TO_PROB_GAIN", 0.0))
        cap = float(getattr(config, "CATASTROPHE_DYNAMIC_MAX_HAZARD_PROB", 1.0))
        return max(0.0, min(cap, base_prob + (self.pressure * gain)))

    def summary_payload(self) -> Dict[str, Any]:
        max_interval = int(getattr(config, "CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS", 0))
        return {
            "version": str(self.version),
            "enabled": bool(self.enabled),
            "cooldown_remaining": int(self.cooldown_remaining),
            "pressure": float(self.pressure),
            "last_pressure_delta": float(self.last_pressure_delta),
            "hazard_probability": float(self.last_threshold),
            "last_roll": (None if self.last_roll is None else float(self.last_roll)),
            "ticks_since_last_end": int(self.ticks_since_last_end),
            "last_trigger_tick": (None if self.last_trigger_tick is None else int(self.last_trigger_tick)),
            "last_end_tick": (None if self.last_end_tick is None else int(self.last_end_tick)),
            "last_selected_key": (None if self.last_selected_key is None else str(self.last_selected_key)),
            "last_trigger_mode": (None if self.last_trigger_mode is None else str(self.last_trigger_mode)),
            "state_label": str(self.last_state_label),
            "hard_trigger_due": bool(self.last_hard_trigger_due),
            "max_interval_ticks": int(max_interval),
        }

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "enabled": bool(self.enabled),
            "seed": int(self.seed),
            "rng_state": self._rng.get_state(),
            "pressure": float(self.pressure),
            "cooldown_remaining": int(self.cooldown_remaining),
            "ticks_since_last_end": int(self.ticks_since_last_end),
            "last_roll": self.last_roll,
            "last_threshold": float(self.last_threshold),
            "last_pressure_delta": float(self.last_pressure_delta),
            "last_trigger_tick": self.last_trigger_tick,
            "last_end_tick": self.last_end_tick,
            "last_selected_key": self.last_selected_key,
            "last_trigger_mode": self.last_trigger_mode,
            "last_state_label": self.last_state_label,
            "last_hard_trigger_due": bool(self.last_hard_trigger_due),
        }

    def load_checkpoint_payload(self, payload: Optional[Dict[str, Any]]) -> None:
        if not payload:
            return
        if not isinstance(payload, dict):
            raise TypeError(f"scheduler checkpoint payload must be dict, got {type(payload).__name__}")
        self.seed = int(payload.get("seed", self.seed))
        self._rng = torch.Generator(device="cpu")
        rng_state = payload.get("rng_state", None)
        if torch.is_tensor(rng_state):
            self._rng.set_state(rng_state.cpu())
        else:
            self._rng.manual_seed(self.seed)
        self.enabled = bool(payload.get("enabled", self.enabled))
        self.pressure = float(payload.get("pressure", 0.0))
        self.cooldown_remaining = int(payload.get("cooldown_remaining", 0))
        self.ticks_since_last_end = int(payload.get("ticks_since_last_end", 0))
        self.last_roll = payload.get("last_roll", None)
        self.last_threshold = float(payload.get("last_threshold", 0.0))
        self.last_pressure_delta = float(payload.get("last_pressure_delta", 0.0))
        self.last_trigger_tick = payload.get("last_trigger_tick", None)
        self.last_end_tick = payload.get("last_end_tick", None)
        self.last_selected_key = payload.get("last_selected_key", None)
        self.last_trigger_mode = payload.get("last_trigger_mode", None)
        self.last_state_label = str(payload.get("last_state_label", self.last_state_label))
        self.last_hard_trigger_due = bool(payload.get("last_hard_trigger_due", False))

    def note_active(self) -> None:
        self.last_state_label = "active"
        self.last_roll = None
        self.last_threshold = 0.0
        self.last_hard_trigger_due = False

    def note_catastrophe_end(self, tick: int) -> None:
        self.cooldown_remaining = int(getattr(config, "CATASTROPHE_DYNAMIC_COOLDOWN_TICKS", 0))
        self.pressure = 0.0
        self.ticks_since_last_end = 0
        self.last_end_tick = int(tick)
        self.last_state_label = "cooldown" if self.cooldown_remaining > 0 else "idle"
        self.last_roll = None
        self.last_threshold = 0.0
        self.last_hard_trigger_due = False

    def step_idle(self, *, base_zone_value_map: torch.Tensor, tick: int) -> Dict[str, Any]:
        tick_i = int(tick)
        status = {
            "trigger_now": False,
            "trigger_mode": None,
            "hazard_probability": 0.0,
            "hard_trigger_due": False,
            "pressure": float(self.pressure),
        }

        if not self.enabled:
            self.last_state_label = "disabled"
            self.last_roll = None
            self.last_threshold = 0.0
            self.last_hard_trigger_due = False
            return status

        if self.cooldown_remaining > 0:
            self.cooldown_remaining = max(0, int(self.cooldown_remaining) - 1)
            self.ticks_since_last_end += 1
            decay = float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_COOLDOWN_DECAY", 0.0))
            self.pressure = max(0.0, float(self.pressure) * max(0.0, min(1.0, 1.0 - decay)))
            self.last_state_label = "cooldown" if self.cooldown_remaining > 0 else "arming"
            self.last_roll = None
            self.last_threshold = 0.0
            self.last_hard_trigger_due = False
            return {**status, "pressure": float(self.pressure)}

        self.ticks_since_last_end += 1
        increment = self._pressure_increment(base_zone_value_map)
        pressure_cap = float(getattr(config, "CATASTROPHE_DYNAMIC_PRESSURE_CAP", 1e9))
        self.pressure = max(0.0, min(pressure_cap, float(self.pressure) + float(increment)))
        self.last_pressure_delta = float(increment)

        min_interval = max(0, int(getattr(config, "CATASTROPHE_DYNAMIC_MIN_INTERVAL_TICKS", 0)))
        max_interval = max(0, int(getattr(config, "CATASTROPHE_DYNAMIC_MAX_INTERVAL_TICKS", 0)))
        if self.ticks_since_last_end < min_interval:
            self.last_state_label = "arming"
            self.last_roll = None
            self.last_threshold = 0.0
            self.last_hard_trigger_due = False
            return {**status, "pressure": float(self.pressure)}

        hazard = self._hazard_probability()
        hard_due = bool(max_interval > 0 and self.ticks_since_last_end >= max_interval)
        self.last_threshold = float(hazard)
        self.last_hard_trigger_due = hard_due
        roll = None if hard_due else _generator_rand_float(self._rng)
        self.last_roll = roll

        if hard_due:
            self.last_state_label = "hard_due"
            return {
                **status,
                "trigger_now": True,
                "trigger_mode": "hard_interval",
                "hazard_probability": float(hazard),
                "hard_trigger_due": True,
                "pressure": float(self.pressure),
            }

        if roll is not None and roll < hazard:
            self.last_state_label = "hazard_trigger"
            return {
                **status,
                "trigger_now": True,
                "trigger_mode": "hazard_roll",
                "hazard_probability": float(hazard),
                "hard_trigger_due": False,
                "pressure": float(self.pressure),
            }

        self.last_state_label = "armed"
        return {
            **status,
            "hazard_probability": float(hazard),
            "hard_trigger_due": False,
            "pressure": float(self.pressure),
        }

    def note_catastrophe_start(self, *, tick: int, preset_key: str, trigger_mode: str) -> None:
        self.last_trigger_tick = int(tick)
        self.last_selected_key = str(preset_key)
        self.last_trigger_mode = str(trigger_mode)
        self.last_state_label = "active"
        self.last_hard_trigger_due = False
        self.pressure = 0.0


class CatastropheController:
    """Lifecycle controller for active catastrophes plus deterministic dynamic scheduling."""

    def __init__(self, *, device: torch.device, seed: Optional[int] = None) -> None:
        self.device = torch.device(device)
        self.global_enabled: bool = bool(getattr(config, "CATASTROPHE_ENABLED", True))
        self._active: Optional[ActiveCatastropheState] = None
        self.scheduler = DynamicCatastropheScheduler(device=self.device, seed=seed)

    def is_active(self) -> bool:
        return self._active is not None

    @property
    def active_state(self) -> Optional[ActiveCatastropheState]:
        return self._active

    def set_global_enabled(self, enabled: bool) -> bool:
        self.global_enabled = bool(enabled)
        return bool(self.global_enabled)

    def set_dynamic_enabled(self, enabled: bool) -> bool:
        self.scheduler.set_enabled(bool(enabled))
        return bool(self.scheduler.enabled)

    def summary_payload(self) -> Dict[str, Any]:
        if self._active is None:
            out = {
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
        else:
            out = self._active.summary_payload()
        out["system_enabled"] = bool(self.global_enabled)
        out["scheduler"] = self.scheduler.summary_payload()
        return out

    def activate(
        self,
        spec: CatastropheSpec,
        *,
        tick: int,
        world_shape: GridShape,
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        if not bool(self.global_enabled):
            raise RuntimeError("catastrophe system is disabled; activation is blocked")
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
        self.scheduler.note_active()
        payload = {"phase": "activated", **self._active.summary_payload()}
        payload["scheduler"] = self.scheduler.summary_payload()
        return payload

    def clear(self, *, tick: int, reason: str = "cleared") -> Optional[Dict[str, Any]]:
        if self._active is None:
            return None
        payload = {
            "phase": "ended",
            "end_tick": int(tick),
            "reason": str(reason),
            **self._active.summary_payload(),
        }
        self._active = None
        self.scheduler.note_catastrophe_end(int(tick))
        payload["scheduler"] = self.scheduler.summary_payload()
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
            return {
                "base": base,
                "effective": base,
                "override": zero_override,
                "apply_mask": false_mask,
                "edit_lock_mask": false_mask,
            }
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
        return {
            "global_enabled": bool(self.global_enabled),
            "active": self._active is not None,
            "active_state": (None if self._active is None else self._active.checkpoint_payload()),
            "scheduler": self.scheduler.checkpoint_payload(),
        }

    def load_checkpoint_payload(self, payload: Optional[Dict[str, Any]], *, device: torch.device) -> None:
        self.device = torch.device(device)
        self.global_enabled = bool(getattr(config, "CATASTROPHE_ENABLED", True))
        self.scheduler = DynamicCatastropheScheduler(device=self.device, seed=None)
        if not payload:
            self._active = None
            return
        if not isinstance(payload, dict):
            raise TypeError(f"catastrophe checkpoint payload must be a dict, got {type(payload).__name__}")
        self.global_enabled = bool(payload.get("global_enabled", self.global_enabled))
        self.scheduler.load_checkpoint_payload(payload.get("scheduler", None))
        if not bool(self.global_enabled) or not bool(payload.get("active", False)):
            self._active = None
            return
        active_payload = payload.get("active_state", None)
        if not isinstance(active_payload, dict):
            raise ValueError("active catastrophe checkpoint payload is missing active_state")
        self._active = ActiveCatastropheState.from_checkpoint_payload(active_payload, device=self.device)
        self._active.metadata.setdefault("restored_from_checkpoint", True)
        self._active.metadata.setdefault("restore_origin", "checkpoint")
        self.scheduler.note_active()

    def maybe_activate_dynamic(
        self,
        *,
        tick: int,
        base_zone_value_map: torch.Tensor,
        world_shape: GridShape,
        replace_existing: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not bool(self.global_enabled):
            return None
        if self._active is not None:
            self.scheduler.note_active()
            return None
        if not bool(getattr(config, "CATASTROPHE_DYNAMIC_SCHEDULER_ENABLED", True)):
            return None

        base = _canonical_base_zone_value_map(base_zone_value_map).to(self.device)
        step = self.scheduler.step_idle(base_zone_value_map=base, tick=int(tick))
        if not bool(step.get("trigger_now", False)):
            return None

        preset_key = self._choose_dynamic_preset_key(base)
        if preset_key is None:
            self.scheduler.last_state_label = "no_eligible_preset"
            return None

        spec = build_dynamic_catastrophe_spec(
            preset_key,
            base_zone_value_map=base,
            generator=self.scheduler.generator,
        )
        spec.metadata.update(
            {
                "trigger_source": "dynamic",
                "trigger_mode": str(step.get("trigger_mode", "hazard_roll")),
                "scheduler_pressure": float(step.get("pressure", 0.0)),
                "scheduler_hazard_probability": float(step.get("hazard_probability", 0.0)),
                "scheduler_hard_trigger_due": bool(step.get("hard_trigger_due", False)),
            }
        )
        payload = self.activate(spec, tick=int(tick), world_shape=world_shape, replace_existing=bool(replace_existing))
        self.scheduler.note_catastrophe_start(
            tick=int(tick),
            preset_key=preset_key,
            trigger_mode=str(step.get("trigger_mode", "hazard_roll")),
        )
        payload["scheduler"] = self.scheduler.summary_payload()
        return payload

    def _choose_dynamic_preset_key(self, base_zone_value_map: torch.Tensor) -> Optional[str]:
        candidates: List[str] = []
        weights: List[float] = []
        for key, weight in dynamic_catastrophe_weight_map().items():
            w = float(weight)
            if w <= 0.0:
                continue
            if not catastrophe_preset_is_eligible(key, base_zone_value_map=base_zone_value_map):
                continue
            candidates.append(str(key))
            weights.append(w)
        if not candidates:
            return None
        idx = _weighted_choice_index(weights, self.scheduler.generator)
        return candidates[idx]


# -----------------------------------------------------------------------------
# Concrete catastrophe pack
# -----------------------------------------------------------------------------

def dynamic_catastrophe_weight_map() -> Dict[str, float]:
    return dict(getattr(config, "CATASTROPHE_DYNAMIC_WEIGHTS", {}) or {})


def catastrophe_preset_is_eligible(preset_key: str, *, base_zone_value_map: torch.Tensor) -> bool:
    key = str(preset_key).strip().lower()
    base = _canonical_base_zone_value_map(base_zone_value_map)
    nonzero_mask = _active_nonzero_base_mask(base)
    positive_mask = _positive_base_mask(base)
    if key in {"global_attenuation", "polarity_split_left_negative", "polarity_split_right_negative", "inversion", "full_dormancy", "regional_attenuation_left", "regional_attenuation_right"}:
        if key == "regional_attenuation_left":
            mask = _regional_half_mask(shape=tuple(int(v) for v in base.shape), device=base.device, region="left")
            return bool((mask & nonzero_mask).any().item())
        if key == "regional_attenuation_right":
            mask = _regional_half_mask(shape=tuple(int(v) for v in base.shape), device=base.device, region="right")
            return bool((mask & nonzero_mask).any().item())
        return bool(nonzero_mask.any().item())
    if key == "positive_band_dormancy":
        positive_count = int(positive_mask.sum().item())
        return positive_count > 1
    return False


def build_inversion_catastrophe_spec(
    *,
    base_zone_value_map: torch.Tensor,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """Flip the sign of every active non-zero canonical base-zone cell."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    apply_mask = _active_nonzero_base_mask(base)
    if not bool(apply_mask.any().item()):
        raise ValueError("inversion catastrophe requires at least one non-zero base-zone cell")
    override_value_map = (-base).clamp(-1.0, 1.0)
    return CatastropheSpec(
        type_name="inversion",
        duration_ticks=_resolve_duration_ticks(duration_ticks, key_hint="inversion"),
        override_value_map=override_value_map,
        apply_mask=apply_mask,
        edit_lock_mask=_full_world_edit_lock_mask(base),
        metadata={
            "preset_key": "inversion",
            "display_name": "Inversion (Experimental)",
            "scope": "global_nonzero_base",
            "safety_tier": "experimental",
        },
    )


def build_full_dormancy_catastrophe_spec(
    *,
    base_zone_value_map: torch.Tensor,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """Collapse every active non-zero canonical base-zone cell to neutral/dormant."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    apply_mask = _active_nonzero_base_mask(base)
    if not bool(apply_mask.any().item()):
        raise ValueError("full dormancy catastrophe requires at least one non-zero base-zone cell")
    override_value_map = torch.zeros_like(base, dtype=torch.float32)
    return CatastropheSpec(
        type_name="full_dormancy",
        duration_ticks=_resolve_duration_ticks(duration_ticks, key_hint="full_dormancy"),
        override_value_map=override_value_map,
        apply_mask=apply_mask,
        edit_lock_mask=_full_world_edit_lock_mask(base),
        metadata={
            "preset_key": "full_dormancy",
            "display_name": "Full Dormancy (Experimental)",
            "scope": "global_nonzero_base",
            "safety_tier": "experimental",
        },
    )


def build_global_attenuation_catastrophe_spec(
    *,
    base_zone_value_map: torch.Tensor,
    factor: Optional[float] = None,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """Shrink all active signed-zone magnitudes toward zero without removing them."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    apply_mask = _active_nonzero_base_mask(base)
    if not bool(apply_mask.any().item()):
        raise ValueError("global attenuation requires at least one non-zero base-zone cell")
    factor_f = float(getattr(config, "CATASTROPHE_GLOBAL_ATTENUATION_FACTOR", 0.35) if factor is None else factor)
    if factor_f < 0.0 or factor_f > 1.0:
        raise ValueError(f"global attenuation factor must be in [0,1], got {factor_f}")
    override_value_map = (base * factor_f).clamp(-1.0, 1.0)
    return CatastropheSpec(
        type_name="global_attenuation",
        duration_ticks=_resolve_duration_ticks(duration_ticks, key_hint="global_attenuation"),
        override_value_map=override_value_map,
        apply_mask=apply_mask,
        edit_lock_mask=_full_world_edit_lock_mask(base),
        metadata={
            "preset_key": "global_attenuation",
            "display_name": "Global Attenuation",
            "attenuation_factor": float(factor_f),
            "scope": "global_nonzero_base",
            "safety_tier": "safe_default",
        },
    )


def build_regional_attenuation_catastrophe_spec(
    *,
    base_zone_value_map: torch.Tensor,
    region: str,
    factor: Optional[float] = None,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """Reduce signed zone intensity in one deterministic half-world region."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    region_key = str(region).strip().lower()
    factor_f = float(getattr(config, "CATASTROPHE_REGIONAL_ATTENUATION_FACTOR", 0.25) if factor is None else factor)
    if factor_f < 0.0 or factor_f > 1.0:
        raise ValueError(f"regional attenuation factor must be in [0,1], got {factor_f}")

    region_mask = _regional_half_mask(shape=tuple(int(v) for v in base.shape), device=base.device, region=region_key)
    apply_mask = region_mask & _active_nonzero_base_mask(base)
    if not bool(apply_mask.any().item()):
        raise ValueError(f"regional attenuation {region_key} has no eligible non-zero base-zone cells")

    override_value_map = base.clone()
    override_value_map[apply_mask] = (override_value_map[apply_mask] * factor_f).clamp(-1.0, 1.0)

    region_display = {
        "left": "Left Half",
        "right": "Right Half",
        "top": "Top Half",
        "bottom": "Bottom Half",
    }.get(region_key, region_key.title())

    return CatastropheSpec(
        type_name="regional_attenuation",
        duration_ticks=_resolve_duration_ticks(duration_ticks, key_hint=f"regional_attenuation_{region_key}"),
        override_value_map=override_value_map,
        apply_mask=apply_mask,
        edit_lock_mask=_full_world_edit_lock_mask(base),
        metadata={
            "preset_key": f"regional_attenuation_{region_key}",
            "display_name": f"Regional Attenuation · {region_display}",
            "scope": "regional_half",
            "region": region_key,
            "attenuation_factor": factor_f,
            "safety_tier": "safe_default",
        },
    )


def build_polarity_split_catastrophe_spec(
    *,
    base_zone_value_map: torch.Tensor,
    negative_side: str,
    magnitude_scale: Optional[float] = None,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """One side becomes harmful-biased while the opposite side becomes beneficial-biased."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    nonzero_mask = _active_nonzero_base_mask(base)
    if not bool(nonzero_mask.any().item()):
        raise ValueError("polarity split requires at least one non-zero base-zone cell")

    scale_f = float(getattr(config, "CATASTROPHE_POLARITY_SPLIT_MAGNITUDE_SCALE", 0.75) if magnitude_scale is None else magnitude_scale)
    if scale_f <= 0.0 or scale_f > 1.0:
        raise ValueError(f"polarity split magnitude_scale must be in (0,1], got {scale_f}")

    neg_side = str(negative_side).strip().lower()
    if neg_side not in {"left", "right"}:
        raise ValueError(f"polarity split negative_side must be 'left' or 'right', got {negative_side!r}")

    left_mask = _regional_half_mask(shape=tuple(int(v) for v in base.shape), device=base.device, region="left")
    right_mask = _regional_half_mask(shape=tuple(int(v) for v in base.shape), device=base.device, region="right")
    magnitude = base.abs().clamp(0.0, 1.0) * scale_f
    override_value_map = torch.zeros_like(base)
    if neg_side == "left":
        override_value_map[left_mask] = -magnitude[left_mask]
        override_value_map[right_mask] = magnitude[right_mask]
        preset_key = "polarity_split_left_negative"
        display_name = "Polarity Split · Left Harmful"
    else:
        override_value_map[left_mask] = magnitude[left_mask]
        override_value_map[right_mask] = -magnitude[right_mask]
        preset_key = "polarity_split_right_negative"
        display_name = "Polarity Split · Right Harmful"

    return CatastropheSpec(
        type_name="polarity_split",
        duration_ticks=_resolve_duration_ticks(duration_ticks, key_hint=preset_key),
        override_value_map=override_value_map,
        apply_mask=nonzero_mask,
        edit_lock_mask=_full_world_edit_lock_mask(base),
        metadata={
            "preset_key": preset_key,
            "display_name": display_name,
            "negative_side": neg_side,
            "magnitude_scale": float(scale_f),
            "scope": "global_nonzero_base",
            "safety_tier": "safe_default",
        },
    )


def build_positive_band_dormancy_catastrophe_spec(
    *,
    base_zone_value_map: torch.Tensor,
    axis: str,
    center_index: int,
    band_fraction: Optional[float] = None,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """Temporarily neutralize only a contiguous band of beneficial support."""
    base = _canonical_base_zone_value_map(base_zone_value_map)
    positive_mask = _positive_base_mask(base)
    positive_count = int(positive_mask.sum().item())
    if positive_count <= 1:
        raise ValueError("positive band dormancy requires at least two beneficial base-zone cells")

    axis_key = str(axis).strip().lower()
    if axis_key not in {"vertical", "horizontal"}:
        raise ValueError(f"band dormancy axis must be 'vertical' or 'horizontal', got {axis!r}")

    frac = float(getattr(config, "CATASTROPHE_POSITIVE_BAND_DORMANCY_FRACTION", 0.22) if band_fraction is None else band_fraction)
    if frac <= 0.0 or frac >= 1.0:
        raise ValueError(f"positive band dormancy fraction must be in (0,1), got {frac}")

    h, w = tuple(int(v) for v in base.shape)
    dim = w if axis_key == "vertical" else h
    band_width = max(1, int(round(dim * frac)))

    band_mask = _band_mask(shape=(h, w), axis=axis_key, center_index=int(center_index), band_width=band_width, device=base.device)
    apply_mask = band_mask & positive_mask

    while int(apply_mask.sum().item()) >= positive_count and band_width > 1:
        band_width = max(1, band_width // 2)
        band_mask = _band_mask(shape=(h, w), axis=axis_key, center_index=int(center_index), band_width=band_width, device=base.device)
        apply_mask = band_mask & positive_mask

    if not bool(apply_mask.any().item()) or int(apply_mask.sum().item()) >= positive_count:
        raise ValueError("positive band dormancy could not find a safe partial beneficial band to neutralize")

    override_value_map = base.clone()
    override_value_map[apply_mask] = 0.0

    return CatastropheSpec(
        type_name="positive_band_dormancy",
        duration_ticks=_resolve_duration_ticks(duration_ticks, key_hint="positive_band_dormancy"),
        override_value_map=override_value_map,
        apply_mask=apply_mask,
        edit_lock_mask=_full_world_edit_lock_mask(base),
        metadata={
            "preset_key": "positive_band_dormancy",
            "display_name": f"Positive Band Dormancy · {axis_key.title()}",
            "axis": axis_key,
            "center_index": int(center_index),
            "band_width": int(band_width),
            "band_fraction": float(frac),
            "scope": "beneficial_band_only",
            "safety_tier": "safe_default",
        },
    )


def build_manual_catastrophe_spec(
    preset_key: str,
    *,
    base_zone_value_map: torch.Tensor,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """Dispatch one operator-facing catastrophe preset by stable key."""
    key = str(preset_key).strip().lower()
    if key == "global_attenuation":
        return build_global_attenuation_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            duration_ticks=duration_ticks,
        )
    if key == "positive_band_dormancy":
        default_axis = str(getattr(config, "CATASTROPHE_MANUAL_POSITIVE_BAND_AXIS", "vertical")).strip().lower()
        base = _canonical_base_zone_value_map(base_zone_value_map)
        h, w = tuple(int(v) for v in base.shape)
        center_index = (w // 2) if default_axis == "vertical" else (h // 2)
        return build_positive_band_dormancy_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            axis=default_axis,
            center_index=int(center_index),
            duration_ticks=duration_ticks,
        )
    if key == "polarity_split_left_negative":
        return build_polarity_split_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            negative_side="left",
            duration_ticks=duration_ticks,
        )
    if key == "polarity_split_right_negative":
        return build_polarity_split_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            negative_side="right",
            duration_ticks=duration_ticks,
        )
    if key == "regional_attenuation_left":
        return build_regional_attenuation_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            region="left",
            duration_ticks=duration_ticks,
        )
    if key == "regional_attenuation_right":
        return build_regional_attenuation_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            region="right",
            duration_ticks=duration_ticks,
        )
    if key in {"inversion", "inversion_experimental"}:
        return build_inversion_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            duration_ticks=duration_ticks,
        )
    if key in {"full_dormancy", "dormancy", "full_dormancy_experimental", "dormancy_experimental"}:
        return build_full_dormancy_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            duration_ticks=duration_ticks,
        )
    raise KeyError(f"unknown manual catastrophe preset: {preset_key!r}")


def build_dynamic_catastrophe_spec(
    preset_key: str,
    *,
    base_zone_value_map: torch.Tensor,
    generator: torch.Generator,
    duration_ticks: Optional[int] = None,
) -> CatastropheSpec:
    """Build one scheduler-selected catastrophe spec using deterministic scheduler RNG."""
    key = str(preset_key).strip().lower()
    if key == "global_attenuation":
        return build_global_attenuation_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            duration_ticks=duration_ticks,
        )
    if key == "positive_band_dormancy":
        axis_mode = str(getattr(config, "CATASTROPHE_DYNAMIC_BAND_AXIS_MODE", "random")).strip().lower()
        base = _canonical_base_zone_value_map(base_zone_value_map)
        h, w = tuple(int(v) for v in base.shape)
        if axis_mode == "random":
            axis = "vertical" if _generator_rand_float(generator) < 0.5 else "horizontal"
        elif axis_mode in {"vertical", "horizontal"}:
            axis = axis_mode
        else:
            axis = "vertical"
        dim = w if axis == "vertical" else h
        center = _generator_rand_index(generator, dim)
        return build_positive_band_dormancy_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            axis=axis,
            center_index=center,
            duration_ticks=duration_ticks,
        )
    if key == "polarity_split_left_negative":
        return build_polarity_split_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            negative_side="left",
            duration_ticks=duration_ticks,
        )
    if key == "polarity_split_right_negative":
        return build_polarity_split_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            negative_side="right",
            duration_ticks=duration_ticks,
        )
    if key == "regional_attenuation_left":
        return build_regional_attenuation_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            region="left",
            duration_ticks=duration_ticks,
        )
    if key == "regional_attenuation_right":
        return build_regional_attenuation_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            region="right",
            duration_ticks=duration_ticks,
        )
    if key == "inversion":
        return build_inversion_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            duration_ticks=duration_ticks,
        )
    if key == "full_dormancy":
        return build_full_dormancy_catastrophe_spec(
            base_zone_value_map=base_zone_value_map,
            duration_ticks=duration_ticks,
        )
    raise KeyError(f"unknown dynamic catastrophe preset: {preset_key!r}")

