from __future__ import annotations

from typing import Any, Dict


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y", "on", "t"}:
            return True
        if norm in {"0", "false", "no", "n", "off", "f"}:
            return False
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_viewer_checkpoint_state(raw: Any) -> Dict[str, Any]:
    """
    Normalize persisted viewer state into a stable schema.

    Backward compatibility:
    - Older checkpoints stored ``speed_mult`` instead of ``speed_multiplier``.
    - Score keys may be serialized as strings by JSON-like tooling.
    """
    src = raw if isinstance(raw, dict) else {}
    camera_src = src.get("camera", {}) if isinstance(src.get("camera"), dict) else {}

    agent_scores: Dict[int, float] = {}
    for key, value in dict(src.get("agent_scores", {})).items():
        try:
            agent_scores[int(key)] = float(value)
        except Exception:
            continue

    marked = []
    for item in list(src.get("marked", [])):
        try:
            slot = int(item)
        except Exception:
            continue
        if slot not in marked:
            marked.append(slot)
        if len(marked) >= 10:
            break

    speed = src.get("speed_multiplier", src.get("speed_mult", 1.0))

    return {
        "paused": _as_bool(src.get("paused"), False),
        "speed_multiplier": max(0.25, min(16.0, _as_float(speed, 1.0))),
        "show_rays": _as_bool(src.get("show_rays"), False),
        "threat_vision_mode": _as_bool(src.get("threat_vision_mode"), False),
        "battle_view_enabled": _as_bool(src.get("battle_view_enabled"), False),
        "show_brain_types": _as_bool(src.get("show_brain_types"), False),
        "marked": marked,
        "camera": {
            "offset_x": _as_float(camera_src.get("offset_x"), 0.0),
            "offset_y": _as_float(camera_src.get("offset_y"), 0.0),
            "zoom": max(0.25, min(8.0, _as_float(camera_src.get("zoom"), 1.0))),
        },
        "agent_scores": agent_scores,
    }
