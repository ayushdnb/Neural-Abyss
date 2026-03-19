from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Optional, Tuple
import warnings

try:
    import pygame as pygame
except ModuleNotFoundError:
    pygame = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PygameRuntimeInfo:
    pygame_module_version: str
    pygame_ce_distribution_version: Optional[str]
    pygame_distribution_version: Optional[str]


def detect_runtime() -> PygameRuntimeInfo:
    """Return import/runtime information for the active pygame module."""
    try:
        ce_ver = metadata.version("pygame-ce")
    except metadata.PackageNotFoundError:
        ce_ver = None

    try:
        pg_ver = metadata.version("pygame")
    except metadata.PackageNotFoundError:
        pg_ver = None

    return PygameRuntimeInfo(
        pygame_module_version=str(getattr(getattr(pygame, "version", None), "ver", "missing")),
        pygame_ce_distribution_version=ce_ver,
        pygame_distribution_version=pg_ver,
    )


def ensure_supported_runtime(*, strict: bool = True) -> PygameRuntimeInfo:
    """
    Validate that the viewer is running on pygame-ce.

    pygame-ce keeps the import name ``pygame`` but ships as the ``pygame-ce``
    distribution on PyPI. New users frequently end up with both ``pygame`` and
    ``pygame-ce`` installed, which makes support incidents difficult to reason
    about because the imported module name alone does not disambiguate the wheel.
    """
    info = detect_runtime()
    if pygame is None:
        raise RuntimeError("pygame is not installed. Install the UI dependency with `pip install pygame-ce`.")
    if info.pygame_ce_distribution_version is None:
        msg = (
            "The UI requires the pygame-ce distribution. "
            "Uninstall the legacy pygame package and install pygame-ce: "
            "`pip uninstall pygame && pip install pygame-ce`."
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if (
        info.pygame_ce_distribution_version is not None
        and info.pygame_distribution_version is not None
    ):
        warnings.warn(
            "Both `pygame` and `pygame-ce` distributions appear to be installed. "
            "Keep only `pygame-ce` to avoid import ambiguity.",
            RuntimeWarning,
            stacklevel=2,
        )
    return info


def primary_desktop_size() -> Tuple[int, int]:
    """
    Return the primary desktop size using the richest API available.
    """
    get_sizes = getattr(pygame.display, "get_desktop_sizes", None)
    if callable(get_sizes):
        try:
            sizes = get_sizes()
            if sizes:
                w, h = sizes[0]
                return int(w), int(h)
        except Exception:
            pass

    info = pygame.display.Info()
    return int(info.current_w), int(info.current_h)


def wheel_steps_from_event(event: Any) -> Optional[int]:
    """
    Return signed vertical wheel steps from a pygame event, if present.

    pygame 2 / pygame-ce emit dedicated ``MOUSEWHEEL`` events while still
    emitting legacy button 4/5 mouse events for backward compatibility.
    """
    ev_type = getattr(event, "type", None)

    if ev_type == getattr(pygame, "MOUSEWHEEL", object()):
        y = int(getattr(event, "y", 0))
        if bool(getattr(event, "flipped", False)):
            y = -y
        return y

    if not hasattr(pygame, "MOUSEWHEEL") and ev_type == getattr(pygame, "MOUSEBUTTONDOWN", object()):
        button = int(getattr(event, "button", 0))
        if button == 4:
            return 1
        if button == 5:
            return -1

    return None


def resize_from_event(event: Any) -> Optional[Tuple[int, int]]:
    """
    Return window size carried by either legacy or SDL2 window resize events.
    """
    ev_type = getattr(event, "type", None)

    if ev_type == getattr(pygame, "VIDEORESIZE", object()):
        return int(getattr(event, "w", 0)), int(getattr(event, "h", 0))

    window_events = {
        getattr(pygame, "WINDOWRESIZED", object()),
        getattr(pygame, "WINDOWSIZECHANGED", object()),
    }
    if ev_type in window_events:
        w = getattr(event, "x", getattr(event, "w", 0))
        h = getattr(event, "y", getattr(event, "h", 0))
        return int(w), int(h)

    return None
