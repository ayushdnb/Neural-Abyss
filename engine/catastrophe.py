from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

import config
from engine.mapgen import HealZone, Zones


def _clamp_unit_interval(value: Any, *, default: float) -> float:
    """Clamp potentially-invalid config fractions into [0, 1]."""
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass(frozen=True)
class CatastropheEvent:
    """
    Full scheduler/runtime description of one catastrophe event.

    The Zones object still stores only the single minimal catastrophe slot
    (event_id + suppressed zone ids). This richer event object lives in the
    controller so runtime, checkpoint, and future UI/debug paths have explicit
    metadata without polluting unrelated systems.
    """

    event_id: str
    pattern_name: str
    suppressed_zone_ids: Tuple[str, ...]
    started_at_tick: Optional[int] = None
    duration_ticks: Optional[int] = None
    expires_at_tick: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "scheduler"

    def to_checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "pattern_name": str(self.pattern_name),
            "suppressed_zone_ids": [str(zid) for zid in self.suppressed_zone_ids],
            "started_at_tick": None if self.started_at_tick is None else int(self.started_at_tick),
            "duration_ticks": None if self.duration_ticks is None else int(self.duration_ticks),
            "expires_at_tick": None if self.expires_at_tick is None else int(self.expires_at_tick),
            "metadata": dict(self.metadata),
            "source": str(self.source),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: Dict[str, Any]) -> "CatastropheEvent":
        if not isinstance(payload, dict):
            raise TypeError("catastrophe event payload must be a dict")
        return cls(
            event_id=str(payload["event_id"]),
            pattern_name=str(payload.get("pattern_name", "unknown")),
            suppressed_zone_ids=tuple(str(zid) for zid in payload.get("suppressed_zone_ids", [])),
            started_at_tick=_opt_int(payload.get("started_at_tick")),
            duration_ticks=_opt_int(payload.get("duration_ticks")),
            expires_at_tick=_opt_int(payload.get("expires_at_tick")),
            metadata=dict(payload.get("metadata", {})),
            source=str(payload.get("source", "scheduler")),
        )

    @classmethod
    def from_zones_slot(
        cls,
        zones: Zones,
        *,
        tick_now: int,
        default_duration_ticks: Optional[int] = None,
    ) -> "CatastropheEvent":
        if zones.catastrophe_state is None:
            raise ValueError("zones do not currently hold an active catastrophe slot")
        slot = zones.catastrophe_state
        duration_ticks = None
        expires_at_tick = None
        if default_duration_ticks is not None and int(default_duration_ticks) > 0:
            duration_ticks = int(default_duration_ticks)
            expires_at_tick = int(tick_now) + int(duration_ticks)
        return cls(
            event_id=str(slot.event_id),
            pattern_name="external_sync",
            suppressed_zone_ids=tuple(str(zid) for zid in slot.suppressed_zone_ids),
            started_at_tick=int(tick_now),
            duration_ticks=duration_ticks,
            expires_at_tick=expires_at_tick,
            metadata={"adopted_from_zones_slot": True},
            source="zones_sync",
        )


@dataclass(frozen=True)
class CatastropheRuntimeSignal:
    """
    Small interpretable runtime signal bundle for optional dynamic scheduling.
    """

    tick: int
    alive_count: int
    on_heal_count: int

    @property
    def heal_occupancy_ratio(self) -> float:
        if self.alive_count <= 0:
            return 0.0
        return float(self.on_heal_count) / float(self.alive_count)


class HealZoneCatastropheController:
    """
    Scheduler/runtime owner for the single catastrophe suppression slot.

    Design rules:
    - The controller never mutates PPO, observations, rewards, or other systems.
    - The controller only owns scheduler/runtime state and writes to
      `zones.apply_catastrophe_suppression_state(...)` / clear methods.
    - The controller tolerates external/manual clear operations by reconciling
      against the Zones slot on every update.
    """

    def __init__(self, zones: Optional[Zones]) -> None:
        self.zones = zones

        self.scheduler_enabled: bool = bool(getattr(config, "CATASTROPHE_SCHEDULER_ENABLED", False))
        self.scheduler_mode: str = str(getattr(config, "CATASTROPHE_SCHEDULER_MODE", "periodic")).strip().lower()
        if self.scheduler_mode not in ("periodic", "dynamic"):
            self.scheduler_mode = "periodic"

        self.cooldown_ticks: int = max(0, int(getattr(config, "CATASTROPHE_COOLDOWN_TICKS", 1200)))
        self.duration_ticks: int = max(1, int(getattr(config, "CATASTROPHE_DURATION_TICKS", 300)))
        self.min_active_zone_count: int = max(1, int(getattr(config, "CATASTROPHE_MIN_ACTIVE_HEAL_ZONES", 1)))
        self.min_zone_count_to_trigger: int = max(
            self.min_active_zone_count + 1,
            int(getattr(config, "CATASTROPHE_MIN_ZONE_COUNT_TO_TRIGGER", 3)),
        )
        self.require_both_halves_covered: bool = bool(
            getattr(config, "CATASTROPHE_REQUIRE_BOTH_HALVES_COVERED", False)
        )

        self.dynamic_heal_occupancy_threshold: float = _clamp_unit_interval(
            getattr(config, "CATASTROPHE_DYNAMIC_HEAL_OCCUPANCY_THRESHOLD", 0.35),
            default=0.35,
        )
        self.dynamic_sustain_ticks: int = max(
            1,
            int(getattr(config, "CATASTROPHE_DYNAMIC_SUSTAIN_TICKS", 120)),
        )

        self.small_suppress_fraction: float = _clamp_unit_interval(
            getattr(config, "CATASTROPHE_SMALL_SUPPRESS_FRACTION", 0.25),
            default=0.25,
        )
        self.medium_suppress_fraction: float = _clamp_unit_interval(
            getattr(config, "CATASTROPHE_MEDIUM_SUPPRESS_FRACTION", 0.50),
            default=0.50,
        )
        self.cluster_survivor_fraction: float = _clamp_unit_interval(
            getattr(config, "CATASTROPHE_CLUSTER_SURVIVOR_FRACTION", 0.25),
            default=0.25,
        )
        self.log_events: bool = bool(getattr(config, "CATASTROPHE_LOG_EVENTS", False))

        self.active_event: Optional[CatastropheEvent] = None
        self.last_trigger_tick: Optional[int] = None
        self.last_clear_tick: int = 0
        self.dynamic_signal_streak: int = 0
        self.next_event_serial: int = 1
        self.last_skip_reason: str = "init"

        # If the Zones object already carries an active catastrophe slot
        # (for example, checkpoint resume or future manual UI wiring), adopt it
        # so controller state and zone state start aligned.
        self.sync_with_zones(tick_now=0)

    def set_scheduler_enabled(self, enabled: bool) -> None:
        self.scheduler_enabled = bool(enabled)
        self.dynamic_signal_streak = 0
        self.last_skip_reason = "scheduler_enabled" if self.scheduler_enabled else "scheduler_disabled"

    def clear_active_catastrophe(self, *, tick_now: int, reason: str) -> bool:
        """
        Clear controller state and the Zones catastrophe slot explicitly.
        """
        zones_changed = False
        if self.zones is not None and self.zones.catastrophe_state is not None:
            self.zones.clear_current_catastrophe_state()
            zones_changed = True

        if self.active_event is not None or zones_changed:
            self.last_clear_tick = int(tick_now)

        self.active_event = None
        self.dynamic_signal_streak = 0
        self.last_skip_reason = str(reason)
        return zones_changed

    def update(self, *, tick_now: int, runtime_signal: Optional[CatastropheRuntimeSignal] = None) -> bool:
        """
        Advance catastrophe runtime/scheduler state by one simulation tick.

        Returns:
            True if the effective zone state changed and the engine should refresh
            cached zone tensors. False otherwise.
        """
        self.sync_with_zones(tick_now=tick_now)
        zones_changed = False

        if self.active_event is not None and self.active_event.expires_at_tick is not None:
            if int(tick_now) >= int(self.active_event.expires_at_tick):
                zones_changed |= self.clear_active_catastrophe(tick_now=tick_now, reason="expired")

        self._update_dynamic_signal(runtime_signal)

        if not self.scheduler_enabled:
            self.last_skip_reason = "scheduler_disabled"
            return zones_changed

        if self.active_event is not None:
            self.last_skip_reason = "catastrophe_already_active"
            return zones_changed

        if self.zones is None or self.zones.heal_zone_count < self.min_zone_count_to_trigger:
            self.last_skip_reason = "insufficient_heal_zones"
            return zones_changed

        if bool(getattr(self.zones, "manual_zone_enabled", {})):
            self.dynamic_signal_streak = 0
            self.last_skip_reason = "manual_overrides_active"
            return zones_changed

        if not self._cooldown_ready(tick_now):
            self.last_skip_reason = "cooldown_active"
            return zones_changed

        if not self._trigger_mode_ready():
            self.last_skip_reason = "trigger_not_ready"
            return zones_changed

        event = self._select_next_event(tick_now=tick_now)
        if event is None:
            self.last_skip_reason = "no_valid_catastrophe"
            return zones_changed

        zones_changed |= self._apply_event(event, tick_now=tick_now)
        return zones_changed

    def sync_with_zones(self, *, tick_now: int) -> None:
        """
        Reconcile controller state with the Zones catastrophe slot.

        This makes future manual viewer clears safe: if the UI clears the zone
        slot directly, the controller sees that and resets its own state instead
        of retaining stale scheduler ownership.
        """
        if self.zones is None:
            self.active_event = None
            self.dynamic_signal_streak = 0
            self.last_skip_reason = "no_zones_bound"
            return

        zone_state = self.zones.catastrophe_state
        if zone_state is None:
            if self.active_event is not None:
                self.active_event = None
                self.dynamic_signal_streak = 0
                self.last_clear_tick = int(tick_now)
                self.last_skip_reason = "zones_slot_cleared_externally"
            return

        if self.active_event is None:
            self.active_event = CatastropheEvent.from_zones_slot(
                self.zones,
                tick_now=int(tick_now),
                default_duration_ticks=int(self.duration_ticks),
            )
            self.last_skip_reason = "zones_slot_adopted"
            return

        same_event_id = str(zone_state.event_id) == str(self.active_event.event_id)
        same_suppressed = tuple(str(z) for z in zone_state.suppressed_zone_ids) == tuple(self.active_event.suppressed_zone_ids)
        if not (same_event_id and same_suppressed):
            self.active_event = CatastropheEvent.from_zones_slot(
                self.zones,
                tick_now=int(tick_now),
                default_duration_ticks=int(self.duration_ticks),
            )
            self.last_skip_reason = "zones_slot_resynced"

    def to_checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "scheduler_enabled": bool(self.scheduler_enabled),
            "scheduler_mode": str(self.scheduler_mode),
            "active_event": None if self.active_event is None else self.active_event.to_checkpoint_payload(),
            "last_trigger_tick": None if self.last_trigger_tick is None else int(self.last_trigger_tick),
            "last_clear_tick": int(self.last_clear_tick),
            "dynamic_signal_streak": int(self.dynamic_signal_streak),
            "next_event_serial": int(self.next_event_serial),
            "last_skip_reason": str(self.last_skip_reason),
        }

    def load_checkpoint_payload(
        self,
        payload: Optional[Dict[str, Any]],
        *,
        zones: Optional[Zones],
        current_tick: int,
    ) -> None:
        """
        Restore controller runtime state from checkpoint payload.

        Missing payload is tolerated for backward compatibility. In that case the
        controller simply adopts any catastrophe already present in the Zones
        checkpoint payload and otherwise falls back to config defaults.
        """
        self.zones = zones

        if not payload:
            self.active_event = None
            self.last_trigger_tick = None
            self.last_clear_tick = int(current_tick)
            self.dynamic_signal_streak = 0
            self.next_event_serial = 1
            self.last_skip_reason = "checkpoint_payload_missing"
            self.sync_with_zones(tick_now=int(current_tick))
            return

        if not isinstance(payload, dict):
            raise TypeError("catastrophe controller payload must be a dict or None")

        self.scheduler_enabled = bool(payload.get("scheduler_enabled", self.scheduler_enabled))
        mode = str(payload.get("scheduler_mode", self.scheduler_mode)).strip().lower()
        self.scheduler_mode = mode if mode in ("periodic", "dynamic") else "periodic"

        active_payload = payload.get("active_event")
        self.active_event = None if active_payload is None else CatastropheEvent.from_checkpoint_payload(active_payload)
        self.last_trigger_tick = _opt_int(payload.get("last_trigger_tick"))
        self.last_clear_tick = int(payload.get("last_clear_tick", current_tick))
        self.dynamic_signal_streak = max(0, int(payload.get("dynamic_signal_streak", 0)))
        self.next_event_serial = max(1, int(payload.get("next_event_serial", 1)))
        self.last_skip_reason = str(payload.get("last_skip_reason", "checkpoint_restore"))

        # Final reconciliation guarantees no silent mismatch with the actual
        # Zones catastrophe slot restored from the world payload.
        self.sync_with_zones(tick_now=int(current_tick))

    def debug_snapshot(self) -> Dict[str, Any]:
        return {
            "scheduler_enabled": bool(self.scheduler_enabled),
            "scheduler_mode": str(self.scheduler_mode),
            "cooldown_ticks": int(self.cooldown_ticks),
            "duration_ticks": int(self.duration_ticks),
            "min_active_zone_count": int(self.min_active_zone_count),
            "min_zone_count_to_trigger": int(self.min_zone_count_to_trigger),
            "dynamic_signal_streak": int(self.dynamic_signal_streak),
            "last_trigger_tick": self.last_trigger_tick,
            "last_clear_tick": int(self.last_clear_tick),
            "last_skip_reason": str(self.last_skip_reason),
            "active_event": None if self.active_event is None else self.active_event.to_checkpoint_payload(),
        }

    def manual_set_zone_enabled(
        self,
        *,
        zone_id: str,
        enabled: bool,
        tick_now: int,
        reason: str = "viewer_manual_zone_override",
    ) -> bool:
        """
        Apply a viewer/manual per-zone override through the controller.

        Manual viewer control must win over any currently active catastrophe.
        So this method clears the controller-owned catastrophe state first, then
        writes the manual zone override into Zones without re-clearing the slot
        a second time behind the controller's back.
        """
        if self.zones is None:
            return False

        self.sync_with_zones(tick_now=int(tick_now))
        zones_changed = False
        if self.active_event is not None or self.zones.catastrophe_state is not None:
            zones_changed |= self.clear_active_catastrophe(tick_now=int(tick_now), reason=str(reason))

        if bool(enabled):
            self.zones.enable_zone_manually(str(zone_id), clear_catastrophe=False)
            self.last_skip_reason = "manual_zone_enabled"
        else:
            self.zones.disable_zone_manually(str(zone_id), clear_catastrophe=False)
            self.last_skip_reason = "manual_zone_disabled"

        self.dynamic_signal_streak = 0
        self.sync_with_zones(tick_now=int(tick_now))
        return True

    def manual_toggle_zone(self, *, zone_id: str, tick_now: int) -> bool:
        """Toggle one heal zone based on its current effective active/dormant state."""
        if self.zones is None:
            return False
        active_now = str(zone_id) in set(self.zones.active_heal_zone_ids)
        return self.manual_set_zone_enabled(
            zone_id=str(zone_id),
            enabled=not active_now,
            tick_now=int(tick_now),
            reason="viewer_manual_zone_toggle",
        )

    def restore_all_zones_to_normal_effective_state(
        self,
        *,
        tick_now: int,
        reason: str = "viewer_restore_all",
    ) -> bool:
        """
        Clear any active catastrophe and all manual overrides, restoring baseline.
        """
        if self.zones is None:
            return False

        self.sync_with_zones(tick_now=int(tick_now))
        zones_changed = False
        if self.active_event is not None or self.zones.catastrophe_state is not None:
            zones_changed |= self.clear_active_catastrophe(tick_now=int(tick_now), reason=str(reason))

        self.zones.restore_all_zones_to_normal_effective_state()
        self.dynamic_signal_streak = 0
        self.last_skip_reason = str(reason)
        self.sync_with_zones(tick_now=int(tick_now))
        return True

    def trigger_manual_pattern(
        self,
        *,
        pattern_key: str,
        tick_now: int,
        source: str = "viewer_manual",
    ) -> bool:
        """
        Apply one explicit catastrophe pattern from the viewer/hotkeys.

        Policy:
        - clear any currently active catastrophe first
        - clear persistent manual overrides so the requested pattern is applied
          exactly, without stale per-zone state stacking on top of it
        - then apply the new pattern through the controller's single event path
        """
        if self.zones is None:
            return False

        builder_map = {
            "random_small": self._build_random_small_fraction_event,
            "random_medium": self._build_random_medium_fraction_event,
            "left_side": self._build_left_bias_event,
            "right_side": self._build_right_bias_event,
            "cluster_survives": self._build_cluster_survives_event,
        }
        key = str(pattern_key).strip().lower()
        if key not in builder_map:
            raise KeyError(f"unknown manual catastrophe pattern: {pattern_key!r}")

        self.sync_with_zones(tick_now=int(tick_now))
        zones_changed = False
        if self.active_event is not None or self.zones.catastrophe_state is not None:
            zones_changed |= self.clear_active_catastrophe(
                tick_now=int(tick_now),
                reason=f"manual_pattern_replace:{key}",
            )

        if bool(getattr(self.zones, "manual_zone_enabled", {})):
            self.zones.reset_manual_overrides()
            zones_changed = True

        template = builder_map[key](tick_now=int(tick_now))
        if template is None:
            self.last_skip_reason = f"manual_pattern_invalid:{key}"
            self.dynamic_signal_streak = 0
            return zones_changed

        event = CatastropheEvent(
            event_id=str(template.event_id),
            pattern_name=str(template.pattern_name),
            suppressed_zone_ids=tuple(str(zid) for zid in template.suppressed_zone_ids),
            started_at_tick=template.started_at_tick,
            duration_ticks=template.duration_ticks,
            expires_at_tick=template.expires_at_tick,
            metadata=dict(template.metadata),
            source=str(source),
        )
        zones_changed |= self._apply_event(event, tick_now=int(tick_now))
        self.last_skip_reason = f"manual_trigger:{key}"
        self.sync_with_zones(tick_now=int(tick_now))
        return zones_changed

    def ui_status_snapshot(self, *, tick_now: int) -> Dict[str, Any]:
        """Return a small viewer-facing catastrophe/heal-zone status snapshot."""
        self.sync_with_zones(tick_now=int(tick_now))

        zone_count = 0
        active_zone_ids: Tuple[str, ...] = tuple()
        manual_override_count = 0
        catastrophe_slot_active = False
        if self.zones is not None:
            zone_count = int(self.zones.heal_zone_count)
            active_zone_ids = tuple(str(zid) for zid in self.zones.active_heal_zone_ids)
            manual_override_count = int(len(getattr(self.zones, "manual_zone_enabled", {})))
            catastrophe_slot_active = self.zones.catastrophe_state is not None

        event = self.active_event
        remaining_ticks: Optional[int] = None
        expires_at_tick: Optional[int] = None
        if event is not None and event.expires_at_tick is not None:
            expires_at_tick = int(event.expires_at_tick)
            remaining_ticks = max(0, int(expires_at_tick) - int(tick_now))

        return {
            "scheduler_enabled": bool(self.scheduler_enabled),
            "scheduler_mode": str(self.scheduler_mode),
            "active_event": None if event is None else event.to_checkpoint_payload(),
            "event_pattern_name": None if event is None else str(event.pattern_name),
            "event_source": None if event is None else str(event.source),
            "event_expires_at_tick": expires_at_tick,
            "event_remaining_ticks": remaining_ticks,
            "zone_count": int(zone_count),
            "active_zone_count": int(len(active_zone_ids)),
            "dormant_zone_count": max(0, int(zone_count) - int(len(active_zone_ids))),
            "active_zone_ids": active_zone_ids,
            "manual_override_count": int(manual_override_count),
            "catastrophe_slot_active": bool(catastrophe_slot_active),
            "last_skip_reason": str(self.last_skip_reason),
        }

    # ------------------------------------------------------------------
    # Internal runtime helpers
    # ------------------------------------------------------------------
    def _update_dynamic_signal(self, runtime_signal: Optional[CatastropheRuntimeSignal]) -> None:
        if runtime_signal is None:
            return
        if runtime_signal.heal_occupancy_ratio >= float(self.dynamic_heal_occupancy_threshold):
            self.dynamic_signal_streak += 1
        else:
            self.dynamic_signal_streak = 0

    def _cooldown_ready(self, tick_now: int) -> bool:
        return int(tick_now) >= int(self.last_clear_tick) + int(self.cooldown_ticks)

    def _trigger_mode_ready(self) -> bool:
        if self.scheduler_mode == "periodic":
            return True
        return int(self.dynamic_signal_streak) >= int(self.dynamic_sustain_ticks)

    def _apply_event(self, event: CatastropheEvent, *, tick_now: int) -> bool:
        if self.zones is None:
            return False
        if self.active_event is not None:
            raise RuntimeError("cannot apply a catastrophe while another catastrophe is already active")
        self.zones.apply_catastrophe_suppression_state(
            event.suppressed_zone_ids,
            event_id=event.event_id,
        )
        self.active_event = event
        self.last_trigger_tick = int(tick_now)
        self.dynamic_signal_streak = 0
        self.last_skip_reason = "triggered"
        if self.log_events:
            print(
                f"[catastrophe] triggered event_id={event.event_id} pattern={event.pattern_name} "
                f"suppressed={list(event.suppressed_zone_ids)} start={event.started_at_tick} "
                f"expires={event.expires_at_tick}"
            )
        return True

    def _select_next_event(self, *, tick_now: int) -> Optional[CatastropheEvent]:
        if self.zones is None:
            return None

        pattern_builders = [
            self._build_random_small_fraction_event,
            self._build_random_medium_fraction_event,
            self._build_left_bias_event,
            self._build_right_bias_event,
            self._build_cluster_survives_event,
        ]

        # Shuffle builder order each trigger attempt to keep selection simple and
        # extensible without hard-wiring a fixed dominance order.
        for builder in random.sample(pattern_builders, k=len(pattern_builders)):
            event = builder(tick_now=tick_now)
            if event is not None:
                return event
        return None

    def _build_random_small_fraction_event(self, *, tick_now: int) -> Optional[CatastropheEvent]:
        return self._build_random_fraction_event(
            tick_now=tick_now,
            pattern_name="random_small_fraction_off",
            suppress_fraction=self.small_suppress_fraction,
        )

    def _build_random_medium_fraction_event(self, *, tick_now: int) -> Optional[CatastropheEvent]:
        return self._build_random_fraction_event(
            tick_now=tick_now,
            pattern_name="random_medium_fraction_off",
            suppress_fraction=self.medium_suppress_fraction,
        )

    def _build_random_fraction_event(
        self,
        *,
        tick_now: int,
        pattern_name: str,
        suppress_fraction: float,
    ) -> Optional[CatastropheEvent]:
        zone_ids = self._all_zone_ids()
        max_suppress = self._max_suppressible_zone_count()
        if max_suppress <= 0 or not zone_ids:
            return None

        target = max(1, int(round(float(suppress_fraction) * float(len(zone_ids)))))
        suppress_count = min(target, max_suppress)
        if suppress_count <= 0:
            return None

        suppressed = tuple(sorted(random.sample(zone_ids, k=suppress_count)))
        return self._make_event_if_valid(
            tick_now=tick_now,
            pattern_name=pattern_name,
            suppressed_zone_ids=suppressed,
            metadata={"suppress_count": int(suppress_count)},
        )

    def _build_left_bias_event(self, *, tick_now: int) -> Optional[CatastropheEvent]:
        if self.zones is None:
            return None
        max_suppress = self._max_suppressible_zone_count()
        if max_suppress <= 0:
            return None

        ordered = sorted(self.zones.heal_zones, key=lambda z: self._zone_center_x(z))
        suppress_count = min(max_suppress, max(1, int(round(0.5 * len(ordered)))))
        suppressed = tuple(zone.zone_id for zone in ordered[:suppress_count])
        return self._make_event_if_valid(
            tick_now=tick_now,
            pattern_name="left_side_bias_off",
            suppressed_zone_ids=suppressed,
            metadata={"suppress_count": int(suppress_count)},
        )

    def _build_right_bias_event(self, *, tick_now: int) -> Optional[CatastropheEvent]:
        if self.zones is None:
            return None
        max_suppress = self._max_suppressible_zone_count()
        if max_suppress <= 0:
            return None

        ordered = sorted(self.zones.heal_zones, key=lambda z: self._zone_center_x(z), reverse=True)
        suppress_count = min(max_suppress, max(1, int(round(0.5 * len(ordered)))))
        suppressed = tuple(zone.zone_id for zone in ordered[:suppress_count])
        return self._make_event_if_valid(
            tick_now=tick_now,
            pattern_name="right_side_bias_off",
            suppressed_zone_ids=suppressed,
            metadata={"suppress_count": int(suppress_count)},
        )

    def _build_cluster_survives_event(self, *, tick_now: int) -> Optional[CatastropheEvent]:
        if self.zones is None or not self.zones.heal_zones:
            return None

        zone_count = self.zones.heal_zone_count
        max_suppress = self._max_suppressible_zone_count()
        if zone_count <= self.min_active_zone_count or max_suppress <= 0:
            return None

        anchor = random.choice(self.zones.heal_zones)
        keep_count = max(
            self.min_active_zone_count,
            int(round(float(zone_count) * float(self.cluster_survivor_fraction))),
        )
        keep_count = min(max(1, keep_count), zone_count - 1)

        def _dist_sq(zone: HealZone) -> float:
            dx = self._zone_center_x(zone) - self._zone_center_x(anchor)
            dy = self._zone_center_y(zone) - self._zone_center_y(anchor)
            return (dx * dx) + (dy * dy)

        ordered = sorted(self.zones.heal_zones, key=_dist_sq)
        survivors = {zone.zone_id for zone in ordered[:keep_count]}
        suppressed = tuple(zone.zone_id for zone in self.zones.heal_zones if zone.zone_id not in survivors)
        if len(suppressed) > max_suppress:
            return None

        return self._make_event_if_valid(
            tick_now=tick_now,
            pattern_name="one_cluster_survives",
            suppressed_zone_ids=suppressed,
            metadata={
                "anchor_zone_id": str(anchor.zone_id),
                "survivor_count": int(keep_count),
            },
        )

    def _make_event_if_valid(
        self,
        *,
        tick_now: int,
        pattern_name: str,
        suppressed_zone_ids: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[CatastropheEvent]:
        normalized = tuple(str(zid) for zid in suppressed_zone_ids)
        if not self._is_valid_suppression_set(normalized):
            return None

        event_id = f"cat_{int(tick_now)}_{int(self.next_event_serial):06d}"
        self.next_event_serial += 1

        duration_ticks = int(self.duration_ticks) if int(self.duration_ticks) > 0 else None
        expires_at_tick = None if duration_ticks is None else int(tick_now) + int(duration_ticks)

        return CatastropheEvent(
            event_id=event_id,
            pattern_name=str(pattern_name),
            suppressed_zone_ids=normalized,
            started_at_tick=int(tick_now),
            duration_ticks=duration_ticks,
            expires_at_tick=expires_at_tick,
            metadata=dict(metadata or {}),
            source="scheduler",
        )

    def _is_valid_suppression_set(self, suppressed_zone_ids: Sequence[str]) -> bool:
        if self.zones is None:
            return False
        if self.active_event is not None:
            return False

        normalized = tuple(str(zid) for zid in suppressed_zone_ids)
        if not normalized:
            return False
        if len(set(normalized)) != len(normalized):
            return False

        zone_ids = set(self._all_zone_ids())
        if any(zid not in zone_ids for zid in normalized):
            return False

        remaining_count = int(self.zones.heal_zone_count) - int(len(normalized))
        if remaining_count < int(self.min_active_zone_count):
            return False
        if remaining_count <= 0:
            return False

        if not self.require_both_halves_covered:
            return True

        active_zones = [zone for zone in self.zones.heal_zones if zone.zone_id not in set(normalized)]
        if len(active_zones) <= 1:
            return True

        mid_x = 0.5 * max(1.0, float(self.zones.W))
        has_left = any(self._zone_center_x(zone) < mid_x for zone in active_zones)
        has_right = any(self._zone_center_x(zone) >= mid_x for zone in active_zones)
        return bool(has_left and has_right)

    def _all_zone_ids(self) -> List[str]:
        if self.zones is None:
            return []
        return [zone.zone_id for zone in self.zones.heal_zones]

    def _max_suppressible_zone_count(self) -> int:
        if self.zones is None:
            return 0
        return max(0, int(self.zones.heal_zone_count) - int(self.min_active_zone_count))

    @staticmethod
    def _zone_center_x(zone: HealZone) -> float:
        _, _, x0, x1 = zone.bounds
        return 0.5 * (float(x0) + float(x1 - 1))

    @staticmethod
    def _zone_center_y(zone: HealZone) -> float:
        y0, y1, _, _ = zone.bounds
        return 0.5 * (float(y0) + float(y1 - 1))


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    return int(v)

