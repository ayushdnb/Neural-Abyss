"""Optional UI surface."""

try:
    from .viewer import Viewer  # type: ignore
except Exception:
    Viewer = None

__all__ = ["Viewer"]
