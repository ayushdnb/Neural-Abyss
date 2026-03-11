"""
viewer.py – Real-time graphical interface for Neural Siege.

================================================================================
WHAT THIS FILE IS (HIGH-LEVEL)
--------------------------------------------------------------------------------
This module implements the *Viewer* for a grid-based simulation called "Neural
Siege". The Viewer is responsible for:

1) Creating and managing a Pygame window (the GUI).
2) Rendering the game world (terrain + agents) efficiently.
3) Handling user input (keyboard + mouse) for camera, toggles, selection, etc.
4) Controlling simulation speed (pause, single-step, speed multiplier).
5) Optional checkpointing (manual via hotkey + automatic periodic/trigger).

IMPORTANT ARCHITECTURE NOTE
--------------------------------------------------------------------------------
- The simulation itself is NOT implemented here. That lives in the Engine.
- The Viewer calls `engine.run_tick()` to advance the simulation.
- The Viewer reads simulation state (grid tensors + registry tensors) to draw.

PERFORMANCE PRINCIPLE (VERY IMPORTANT)
--------------------------------------------------------------------------------
In this project the simulation runs on the GPU (via PyTorch tensors), but the GUI
(Pygame) runs on the CPU. Copying data from GPU to CPU forces synchronization
and is expensive. If we do that every frame, the UI will stutter.

So we:
- Maintain CPU-side caches (numpy arrays) for only the minimal data needed to draw.
- Refresh those caches only once every N frames (configurable).

This is a common pattern in real-time systems:
- Producer (GPU simulation) generates fast state updates.
- Consumer (CPU renderer) samples the state at a manageable cadence.

================================================================================
BEGINNER-FRIENDLY NOTES
--------------------------------------------------------------------------------
- Pygame is a 2D graphics library: you "draw" shapes/images onto a Surface, then
  present it to the screen with `pygame.display.flip()`.

- A "grid world" here means:
  - There is a 2D map of H x W cells.
  - Each cell can contain terrain (empty, wall, zones) and/or agents.

- A "camera" converts coordinates:
  - World/grid coordinates: (x,y) in cells
  - Screen coordinates: pixel positions in the window
  This lets you pan and zoom.

================================================================================
ADVANCED NOTES
--------------------------------------------------------------------------------
- Static background caching: draw terrain once into a Surface, reuse it.
- UI caching: caching rendered text avoids expensive font rendering each frame.
- Monkey-patching PPO: hooking record_step allows tracking per-agent score without
  rewriting the PPO trainer (but must be done carefully to preserve behavior).

================================================================================
"""

from __future__ import annotations

import os
import collections
from typing import List, Tuple, Optional, Dict, Any
import math
from pathlib import Path  # Used for robust checkpoint file path operations

import pygame
import torch
import torch.nn as nn
import numpy as np

import config
from agent.mlp_brain import (
    brain_kind_from_module,
    brain_kind_display_name,
    brain_kind_short_label,
    describe_brain_module,
)
from .camera import Camera
from engine.agent_registry import (
    # These are column indices into `registry.agent_data` tensor.
    # The registry stores per-agent properties in a compact tensor for GPU efficiency.
    COL_ALIVE, COL_TEAM, COL_HP, COL_X, COL_Y, COL_UNIT,
    COL_HP_MAX, COL_VISION, COL_ATK, COL_AGENT_ID
)
from engine.catastrophe import build_manual_catastrophe_spec

# ==============================================================================
# Constants & colour palette
# ==============================================================================
FONT_NAME = str(getattr(config, "UI_FONT_NAME", "consolas"))

# Colors stored as (R, G, B). Some overlays use (R, G, B, A) for alpha blending.
# In Pygame:
# - RGB values range from 0..255
# - Alpha (A) also ranges from 0..255 (0 = fully transparent, 255 = opaque)
COLORS = {
    "bg": (20, 22, 28),
    "hud_bg": (12, 14, 18),
    "side_bg": (18, 20, 26),

    "grid": (40, 42, 48),
    "border": (70, 74, 82),
    "wall": (90, 94, 102),
    "empty": (24, 26, 32),

    # Red team
    "red_soldier": (231, 76, 60),
    "red_archer":  (211, 84, 0),
    "red":         (231, 76, 60),

    # Blue team
    "blue_soldier": (52, 152, 219),
    "blue_archer":  (22, 160, 133),
    "blue":         (52, 152, 219),

    # Signed-zone palette
    "zone_positive": (46, 204, 113),
    "zone_negative": (155, 89, 182),
    "zone_dormant": (108, 116, 132),
    "zone_locked": (241, 196, 15),

    # UI accents
    "archer_glyph": (245, 230, 90),
    "marker": (242, 228, 92),
    "selected": (242, 228, 92),

    "text": (230, 230, 230),
    "text_dim": (180, 186, 194),

    "green": (46, 204, 113),
    "warn": (243, 156, 18),

    "bar_bg": (38, 42, 48),
    "bar_fg": (46, 204, 113),

    # Graph fill colors with alpha (semi-transparent)
    "graph_red": (231, 76, 60, 150),
    "graph_blue": (52, 152, 219, 150),
    "graph_grid": (60, 60, 70),

    "pause_text": (241, 196, 15),
}

# Semi-transparent overlays (all include alpha channel).
OVERLAYS = {
    "cp": (210, 210, 230, 48),

    # Control point outline colors:
    "outline_red": (231, 76, 60, 160),
    "outline_blue": (52, 152, 219, 160),
    "outline_neutral": (160, 160, 170, 120),

    # Threat overlay colors (RGB only, alpha added dynamically later)
    "threat_enemy": (231, 76, 60),
    "threat_ally": (52, 152, 219),

    # Vision circle overlay
    "vision_range": (180, 180, 180, 40),
}

# Ray colors used when "show rays" is enabled
RAY_COLORS = {
    "ally": (52, 152, 219),
    "enemy": (231, 76, 60),
    "wall": (180, 180, 180),
    "empty": (100, 100, 110),
}


# ==============================================================================
# Small safety helpers for color handling
# ==============================================================================
def _clamp_u8(x) -> int:
    """
    Clamp any value to the integer range [0, 255].

    Why:
    - Pygame expects color channels in 0..255.
    - We might compute alpha or intensity dynamically (floats, numpy types).
    - Clamping prevents crashes or undefined behavior.

    Implementation detail:
    - We try to cast to int; if conversion fails, treat as 0.
    """
    try:
        xi = int(x)
    except Exception:
        xi = 0
    if xi < 0:
        return 0
    if xi > 255:
        return 255
    return xi


def _rgb(col) -> tuple[int, int, int]:
    """
    Return a safe (r, g, b) tuple from any iterable with ≥ 3 components.

    Why:
    - Some overlay entries are RGB, some are RGBA.
    - Some callers want only RGB to add alpha later.

    If `col` is invalid, returns (0,0,0).
    """
    r, g, b = (0, 0, 0)
    try:
        if len(col) >= 3:
            r, g, b = col[0], col[1], col[2]
    except Exception:
        pass
    return (_clamp_u8(r), _clamp_u8(g), _clamp_u8(b))


def _blend_rgb(base_rgb, tint_rgb, alpha_01: float) -> tuple[int, int, int]:
    """
    Alpha-blend `tint_rgb` over `base_rgb` using a normalized alpha in [0,1].

    This is used for signed-zone terrain rendering so the operator sees the
    underlying terrain while still reading positive / negative / dormant zone
    state honestly.
    """
    a = max(0.0, min(1.0, float(alpha_01)))
    br, bg, bb = _rgb(base_rgb)
    tr, tg, tb = _rgb(tint_rgb)
    return (
        _clamp_u8(round(br * (1.0 - a) + tr * a)),
        _clamp_u8(round(bg * (1.0 - a) + tg * a)),
        _clamp_u8(round(bb * (1.0 - a) + tb * a)),
    )


def _zone_state_label(value: float, eps: float = 1e-6) -> str:
    """Return a qualitative signed-zone label from one scalar value."""
    v = float(value)
    if v > eps:
        return "Beneficial"
    if v < -eps:
        return "Harmful"
    return "Dormant"


def _zone_overlay_rgb_alpha(value: float, *, dormant_alpha: float = 0.05) -> tuple[tuple[int, int, int], float]:
    """
    Convert one signed base-zone value into a tint colour and normalized alpha.

    Rendering policy for Patch 3 / Patch 5:
    - positive values: green tint, stronger with magnitude
    - negative values: purple tint, stronger with magnitude
    - zero / near-zero: faint neutral tint so dormant cells remain visually distinct
    - when a catastrophe is active, the renderer may feed this helper the runtime-effective
      field rather than the canonical base field so the operator sees the active override honestly
    """
    v = float(value)
    mag = max(0.0, min(1.0, abs(v)))
    if v > 1e-6:
        return COLORS["zone_positive"], 0.10 + 0.28 * mag
    if v < -1e-6:
        return COLORS["zone_negative"], 0.10 + 0.28 * mag
    return COLORS["zone_dormant"], dormant_alpha


def _catastrophe_display_name(status: Optional[Dict[str, Any]]) -> str:
    """Return the operator-facing display name for one catastrophe status payload."""
    if not status:
        return "None"
    metadata = dict(status.get("metadata", {}) or {})
    display_name = metadata.get("display_name", None)
    if display_name:
        return str(display_name)
    type_name = status.get("type_name", None)
    if not type_name:
        return "None"
    return str(type_name).replace("_", " ").title()


def _catastrophe_status_signature(status: Optional[Dict[str, Any]]) -> tuple:
    """Return a stable cache signature for catastrophe-driven zone visuals."""
    status = dict(status or {})
    metadata = dict(status.get("metadata", {}) or {})
    md_key = tuple(sorted((str(k), repr(v)) for k, v in metadata.items()))
    return (
        bool(status.get("active", False)),
        str(status.get("type_name", "")),
        int(status.get("start_tick") or -1),
        int(status.get("active_cell_count") or 0),
        int(status.get("locked_cell_count") or 0),
        md_key,
    )


# ==============================================================================
# Utility functions
# ==============================================================================
def _center_window():
    """
    Hint to SDL (the backend library used by Pygame) to center the window.

    This uses an environment variable read by SDL at window creation time.
    We set it only if it hasn't already been set externally.
    """
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")


def _get_model_summary(model: nn.Module) -> str:
    """
    Return a short string describing a PyTorch model architecture.

    Used in the side panel when an agent is selected.

    Heuristic approach:
    - If the class name indicates a Transformer, try to show embed dim and
      attention presence.
    - Otherwise, attempt to infer an MLP structure by collecting Linear layers.
    """
    try:
        return describe_brain_module(model)
    except Exception:
        return model.__class__.__name__


def _param_count(model: nn.Module) -> int:
    """
    Return number of trainable parameters in a PyTorch model.

    Why useful:
    - Gives quick sense of complexity.
    - Helps compare brain architectures.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# Text caching
# ==============================================================================
class TextCache:
    """
    Caches both pygame.Font objects (per size) and rendered text Surfaces.

    PERFORMANCE PROBLEM:
    - Creating fonts frequently is slow.
    - Rendering text is also slow (rasterization).
    - In a real-time UI, we may render the same label every frame.

    SOLUTION:
    - Keep a dictionary of Font objects keyed by size.
    - Keep a dictionary of rendered text surfaces keyed by:
        (text, size, color, antialias)

    This avoids repeated work, improving FPS and reducing CPU usage.
    """

    def __init__(self):
        # Pre-create a few common sizes to reduce first-frame latency.
        self.fonts = {
            13: self._mk_font(13),
            16: self._mk_font(16),
            18: self._mk_font(18),
        }

        # Cache for rendered surfaces
        # key: (text, size, color, antialias) -> pygame.Surface
        self.cache = {}

    def _mk_font(self, sz: int):
        """
        Create a pygame Font of a given size.

        We attempt to use a system font. If unavailable, we fall back to default.

        Notes:
        - SysFont uses system-installed fonts by name.
        - Font(None, size) uses a default built-in font.
        """
        try:
            return pygame.font.SysFont(FONT_NAME, sz)
        except Exception:
            return pygame.font.Font(None, sz)

    def render(self, text: str, size: int, color, aa: bool = True):
        """
        Return a cached Surface with the text rendered.

        Key behavior:
        - Normalizes text to string.
        - Ensures size is a positive int.
        - Creates missing font sizes on demand (fixes KeyError crashes).
        - Caches rendered surfaces for repeated use.
        """
        # Normalize inputs to stabilize cache keys
        if not isinstance(text, str):
            text = str(text)

        size = int(size)
        if size < 1:
            size = 1

        # Create font on-demand if missing (prevents crashes)
        if size not in self.fonts:
            self.fonts[size] = self._mk_font(size)

        key = (text, size, color, aa)
        if key not in self.cache:
            self.cache[key] = self.fonts[size].render(text, aa, color)

        return self.cache[key]


# ==============================================================================
# Layout management
# ==============================================================================
class LayoutManager:
    """
    Computes rectangles for the world view, right-side operator panels, and
    bottom HUD. The goal is to keep the entire operator surface visible on one
    screen without controls spilling off-panel.
    """

    def __init__(self, viewer):
        self.viewer = viewer

    def side_width(self):
        requested = int(getattr(config, "VIEWER_SIDE_PANEL_WIDTH", 460))
        return max(360, min(max(360, self.viewer.Wpix - 240), requested))

    def bottom_height(self):
        return max(120, int(getattr(config, "VIEWER_BOTTOM_PANEL_HEIGHT", 148)))

    def gap(self):
        return max(4, int(getattr(config, "VIEWER_PANEL_GAP", 8)))

    def world_rect(self):
        m = self.viewer.margin
        return pygame.Rect(
            m,
            m,
            max(64, self.viewer.Wpix - self.side_width() - (3 * m)),
            max(64, self.viewer.Hpix - self.bottom_height() - (2 * m)),
        )

    def side_rect(self):
        m, side_w = self.viewer.margin, self.side_width()
        return pygame.Rect(
            self.viewer.Wpix - side_w - m,
            m,
            side_w,
            max(64, self.viewer.Hpix - self.bottom_height() - (2 * m)),
        )

    def hud_rect(self):
        bottom_h = self.bottom_height()
        return pygame.Rect(0, self.viewer.Hpix - bottom_h, self.viewer.Wpix, bottom_h)

    def side_sections(self):
        s = self.side_rect()
        gap = self.gap()
        inner_w = max(100, s.width - 16)
        top_h = max(170, min(260, int(s.height * 0.30)))
        mid_h = max(170, min(250, int(s.height * 0.28)))
        remaining_h = max(140, s.height - top_h - mid_h - (2 * gap) - 12)
        agent = pygame.Rect(s.x + 8, s.y + 8, inner_w, top_h)
        zone = pygame.Rect(s.x + 8, agent.bottom + gap, inner_w, mid_h)
        help_rect = pygame.Rect(s.x + 8, zone.bottom + gap, inner_w, remaining_h)
        return {"agent": agent, "zone": zone, "help": help_rect}


# ==============================================================================
# World renderer
# ==============================================================================
class WorldRenderer:
    """
    Draws the world grid, agents, and overlays into the Pygame window.

    Patch-3 rendering policy:
    - signed base zones are rendered from the canonical base field
    - positive / negative / dormant cells are visually distinct
    - capture-point rendering stays separate
    - manual viewer edits target only the canonical base layer and invalidate the
      cached terrain surface deterministically
    """

    def __init__(self, viewer, grid, registry):
        self.viewer = viewer
        self.grid = grid
        self.registry = registry
        self.cam = viewer.cam

        # Cached background surface (static terrain)
        self.static_surf = None

        # Precomputed zone info extracted from engine.zones
        self._zone_cache = {
            "base_np": None,
            "render_np": None,
            "edit_locked_np": None,
            "draw_edit_locked_overlay": True,
            "cp_rects": [],
        }

    def build_static_cache(self, engine):
        """
        Extract signed base-zone values and control-point rectangles from `engine.zones`.

        Why:
        - the renderer wants CPU-side arrays so static terrain can be rebuilt
          without per-cell GPU reads
        - the canonical source of truth is now `base_zone_value_map`, not `heal_mask`
        - a later catastrophe patch can optionally provide an edit-lock mask
        """
        self.static_surf = None  # force rebuild of background

        H, W = int(self.grid.shape[1]), int(self.grid.shape[2])
        base_np = np.zeros((H, W), dtype=np.float32)
        render_np = base_np
        edit_locked_np = np.zeros((H, W), dtype=np.bool_)
        draw_edit_locked_overlay = True
        cp_rects: List[tuple[int, int, int, int]] = []

        zones = getattr(engine, "zones", None)
        if zones is not None:
            try:
                base_map = getattr(zones, "base_zone_value_map", None)
                if base_map is None and getattr(zones, "heal_mask", None) is not None:
                    base_map = getattr(zones, "heal_mask").to(dtype=torch.float32)
                if base_map is not None:
                    base_np = base_map.detach().to(dtype=torch.float32).cpu().numpy()
                    render_np = base_np
            except Exception:
                pass

            try:
                effective_map = getattr(engine, "_z_effective_values", None)
                if effective_map is not None:
                    render_np = effective_map.detach().to(dtype=torch.float32).cpu().numpy()
            except Exception:
                render_np = base_np

            try:
                if hasattr(zones, "base_zone_edit_locked_mask"):
                    edit_locked_np = (
                        zones.base_zone_edit_locked_mask
                        .detach()
                        .bool()
                        .cpu()
                        .numpy()
                    )
                    if bool(edit_locked_np.size > 0 and np.all(edit_locked_np)):
                        # Whole-world catastrophe lockout is already called out in the HUD / inspector.
                        # Drawing a border on every cell is visually noisy, so suppress that specific case.
                        draw_edit_locked_overlay = False
            except Exception:
                edit_locked_np = np.zeros((H, W), dtype=np.bool_)
                draw_edit_locked_overlay = True

            # Control points: each cp_mask -> bounding rectangle (x0,y0,x1,y1)
            for m in getattr(zones, "cp_masks", []):
                ys, xs = torch.nonzero(m, as_tuple=True)
                if xs.numel() > 0:
                    cp_rects.append(
                        (xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1)
                    )

        self._zone_cache = {
            "base_np": base_np,
            "render_np": render_np,
            "edit_locked_np": edit_locked_np,
            "draw_edit_locked_overlay": draw_edit_locked_overlay,
            "cp_rects": cp_rects,
        }

    def _draw_static_background(self):
        """
        Render walls, empty cells, and signed base-zone tint onto a static surface.

        Important details:
        - signed zone truth normally comes from the canonical base-zone value map
        - when a catastrophe is active, the overlay uses the runtime-effective zone field
          so the operator sees the actual in-force override rather than the hidden base layer
        - dormant cells receive a faint neutral tint so zero is still visible
        - capture points are NOT baked into this surface; they remain a separate overlay
        """
        wrect = self.viewer.layout.world_rect()
        self.static_surf = pygame.Surface(wrect.size)
        self.static_surf.fill(COLORS["bg"])

        H, W = self.grid.shape[1], self.grid.shape[2]
        occ_np = self.grid[0].detach().cpu().numpy()
        render_np = self._zone_cache.get("render_np", None)
        edit_locked_np = self._zone_cache.get("edit_locked_np", None)
        draw_edit_locked_overlay = bool(self._zone_cache.get("draw_edit_locked_overlay", True))
        show_zone_overlay = bool(getattr(self.viewer, "show_zone_overlay", True))

        for y in range(H):
            for x in range(W):
                occ = int(occ_np[y, x])
                terrain_color = COLORS["wall"] if occ == 1 else COLORS["empty"]
                color = terrain_color

                if show_zone_overlay and render_np is not None and occ != 1:
                    tint_rgb, tint_alpha = _zone_overlay_rgb_alpha(float(render_np[y, x]))
                    color = _blend_rgb(terrain_color, tint_rgb, tint_alpha)

                cx, cy = self.cam.world_to_screen(x, y)
                pygame.draw.rect(
                    self.static_surf,
                    color,
                    (cx, cy, self.cam.cell_px, self.cam.cell_px),
                )

                if (
                    show_zone_overlay
                    and draw_edit_locked_overlay
                    and edit_locked_np is not None
                    and occ != 1
                    and bool(edit_locked_np[y, x])
                    and self.cam.cell_px >= 5
                ):
                    pygame.draw.rect(
                        self.static_surf,
                        COLORS["zone_locked"],
                        (cx, cy, self.cam.cell_px, self.cam.cell_px),
                        1,
                    )

    def draw(self, surf, state_data):
        """
        Draw the entire world.

        Expected keys in state_data:
        - "occ_np": occupancy grid as numpy (H,W)
        - "id_np": id grid as numpy (H,W)
        - "alive_indices": list of slot ids alive
        - "agent_map": dict slot_id -> (x, y, unit, team, uid, brain_type_str)
        """
        wrect = self.viewer.layout.world_rect()

        if self.static_surf is None or self.static_surf.get_size() != wrect.size:
            self._draw_static_background()

        surf.blit(self.static_surf, wrect.topleft)
        c = self.cam.cell_px

        cp_overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for x0, y0, x1, y1 in self._zone_cache["cp_rects"]:
            patch = state_data["occ_np"][y0:y1, x0:x1]
            red_cnt = (patch == 2).sum()
            blue_cnt = (patch == 3).sum()

            if red_cnt > blue_cnt:
                b_col, label = OVERLAYS["outline_red"], ("R", COLORS["red"])
            elif blue_cnt > red_cnt:
                b_col, label = OVERLAYS["outline_blue"], ("B", COLORS["blue"])
            else:
                b_col, label = OVERLAYS["outline_neutral"], ("–", COLORS["text_dim"])

            cx0, cy0 = self.cam.world_to_screen(x0, y0)
            cx1, cy1 = self.cam.world_to_screen(x1, y1)
            rect = pygame.Rect(cx0, cy0, cx1 - cx0, cy1 - cy0)
            pygame.draw.rect(cp_overlay, b_col, rect, max(1, c // 2))
            lab_surf = self.viewer.text_cache.render(label[0], 13, label[1])
            cp_overlay.blit(lab_surf, lab_surf.get_rect(center=rect.center))

        surf.blit(cp_overlay, wrect.topleft)

        for slot_id in state_data["alive_indices"]:
            entry = state_data["agent_map"].get(slot_id, None)
            if entry is None:
                continue

            x, y, unit, team, _uid, _btype = entry
            color_key = f"{'red' if team == 2.0 else 'blue'}_{'archer' if unit == 2.0 else 'soldier'}"
            color = COLORS[color_key]
            cx, cy = self.cam.world_to_screen(int(x), int(y))
            pygame.draw.rect(surf, color, (wrect.x + cx, wrect.y + cy, c, c))

            if unit == 2.0 and c > 4:
                pygame.draw.circle(
                    surf,
                    COLORS["archer_glyph"],
                    (wrect.x + cx + c // 2, wrect.y + cy + c // 2),
                    max(2, c // 2 - 1),
                    max(1, c // 6),
                )

        if self.viewer.battle_view_enabled:
            self._draw_hp_bars(surf, wrect, c, state_data)
        if self.viewer.show_brain_types:
            self._draw_brain_labels(surf, wrect, c, state_data)
        if self.viewer.threat_vision_mode:
            self._draw_threat_vision(surf, wrect, c, state_data)
        if self.viewer.show_grid and c >= 6:
            self._draw_grid_lines(surf, wrect, c)

        self._draw_markers(surf, wrect, c, state_data["id_np"])
        if self.viewer.show_rays:
            self._draw_rays(surf, wrect, c, state_data)
        self._draw_selected_cell(surf, wrect, c)

    def _draw_hp_bars(self, surf, wrect, c, state_data):
        """
        Draw health bars above each agent.

        Rendering note:
        - Bars are drawn only if the cell is large enough, otherwise it becomes noise.
        """
        if c < 8:
            return

        hp_bar_surf = pygame.Surface(wrect.size, pygame.SRCALPHA)

        for slot_id in state_data["alive_indices"]:
            entry = state_data["agent_map"].get(slot_id, None)
            if entry is None:
                continue

            x, y, _unit, _team, _uid, _btype = entry
            hp_max = float(self.registry.agent_data[slot_id, COL_HP_MAX].item())
            hp = float(self.registry.agent_data[slot_id, COL_HP].item())
            hp_ratio_raw = (hp / hp_max) if hp_max > 0 else 0.0
            hp_ratio = max(0.0, min(1.0, hp_ratio_raw))
            cx, cy = self.cam.world_to_screen(int(x), int(y))
            bar_w, bar_h = c, max(1, c // 8)
            bar_y = cy - bar_h - 2
            if 0 <= bar_y < wrect.height and 0 <= cx < wrect.width:
                pygame.draw.rect(hp_bar_surf, (50, 50, 50), (cx, bar_y, bar_w, bar_h))
                pygame.draw.rect(hp_bar_surf, COLORS["green"], (cx, bar_y, bar_w * hp_ratio, bar_h))

        surf.blit(hp_bar_surf, wrect.topleft)

    def _draw_brain_labels(self, surf, wrect, c, state_data):
        """Draw small labels inside agent cells indicating brain type."""
        if c < 8:
            return

        for slot_id in state_data["alive_indices"]:
            entry = state_data["agent_map"].get(slot_id, None)
            if entry is None:
                continue

            x, y, _unit, _team, _uid, btype = entry
            cx, cy = self.cam.world_to_screen(int(x), int(y))
            if 0 <= cx < wrect.width and 0 <= cy < wrect.height:
                font_size = max(10, min(16, c // 2))
                lab_surf = self.viewer.text_cache.render(btype, font_size, COLORS["text"])
                rect = lab_surf.get_rect(center=(wrect.x + cx + c // 2, wrect.y + cy + c // 2))
                surf.blit(lab_surf, rect)

    def _draw_threat_vision(self, surf, wrect, c, state_data):
        """
        Threat vision mode:
        - draw a vision range circle for the selected agent
        - highlight agents within that range
        """
        slot_id = self.viewer.selected_slot_id
        if slot_id is None or slot_id not in state_data["agent_map"]:
            return

        my_x, my_y, _unit, my_team, _uid, _btype = state_data["agent_map"][slot_id]
        my_x = float(my_x)
        my_y = float(my_y)
        my_cx, my_cy = self.cam.world_to_screen(int(my_x), int(my_y))
        vision_range = float(self.registry.agent_data[slot_id, COL_VISION].item())
        vision_px = int(vision_range * c)

        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        center_px = (my_cx + c // 2, my_cy + c // 2)
        pygame.draw.circle(overlay, OVERLAYS["vision_range"], center_px, vision_px)

        for other_id, (ox, oy, _ou, o_team, _ouid, _obtype) in state_data["agent_map"].items():
            if slot_id == other_id:
                continue

            dist = float(np.hypot(ox - my_x, oy - my_y))
            if dist <= vision_range:
                o_cx, o_cy = self.cam.world_to_screen(int(ox), int(oy))
                hp_max = float(self.registry.agent_data[other_id, COL_HP_MAX].item())
                hp = float(self.registry.agent_data[other_id, COL_HP].item())
                hp_ratio = (hp / hp_max) if hp_max > 0 else 0.0

                if o_team != my_team:
                    radius = int(c * 0.7 * (0.5 + hp_ratio * 0.5))
                    alpha = _clamp_u8(50 + 150 * hp_ratio)
                    r, g, b = _rgb(OVERLAYS["threat_enemy"])
                    pygame.draw.circle(overlay, (r, g, b, alpha), (o_cx + c // 2, o_cy + c // 2), radius)
                else:
                    r, g, b = _rgb(OVERLAYS["threat_ally"])
                    pygame.draw.circle(overlay, (r, g, b, 100), (o_cx + c // 2, o_cy + c // 2), int(c * 0.4), 1)

        surf.blit(overlay, wrect.topleft)

    def _draw_grid_lines(self, surf, wrect, c):
        """Draw grid overlay lines aligned to cell boundaries."""
        ax, ay = self.cam.world_to_screen(0, 0)
        off_x = (c - (ax % c)) % c
        off_y = (c - (ay % c)) % c
        x = wrect.x + off_x
        y = wrect.y + off_y

        while x < wrect.right:
            pygame.draw.line(surf, COLORS["grid"], (x, wrect.y), (x, wrect.bottom))
            x += c

        while y < wrect.bottom:
            pygame.draw.line(surf, COLORS["grid"], (wrect.x, y), (wrect.right, y))
            y += c

        pygame.draw.rect(surf, COLORS["border"], wrect, 2)

    def _draw_markers(self, surf, wrect, c, id_np):
        """Draw a marker around explicitly marked agents."""
        for slot_id in self.viewer.marked:
            pos = np.argwhere(id_np == slot_id)
            if pos.size > 0:
                y, x = pos[0]
                cx, cy = self.cam.world_to_screen(int(x), int(y))
                pygame.draw.rect(
                    surf,
                    COLORS["marker"],
                    (wrect.x + cx, wrect.y + cy, c, c),
                    max(1, c // 8),
                )

    def _draw_rays(self, surf, wrect, c, state_data):
        """Draw line-of-sight rays from the selected agent."""
        slot_id = self.viewer.selected_slot_id
        if slot_id is None or slot_id not in state_data["agent_map"]:
            return

        agent_x, agent_y, _unit, my_team, _uid, _btype = state_data["agent_map"][slot_id]
        agent_x = float(agent_x)
        agent_y = float(agent_y)
        start_pos_screen = (
            wrect.x + self.cam.world_to_screen(int(agent_x), int(agent_y))[0] + c // 2,
            wrect.y + self.cam.world_to_screen(int(agent_x), int(agent_y))[1] + c // 2,
        )
        vision_range = int(float(self.registry.agent_data[slot_id, COL_VISION].item()))
        occ_grid = state_data["occ_np"]
        H, W = occ_grid.shape
        num_rays_to_draw = 32

        for i in range(num_rays_to_draw):
            angle = i * (2 * math.pi / num_rays_to_draw)
            dx, dy = math.cos(angle), math.sin(angle)
            end_x, end_y = agent_x, agent_y
            color = RAY_COLORS["empty"]

            for step in range(1, vision_range + 1):
                test_x = int(round(agent_x + dx * step))
                test_y = int(round(agent_y + dy * step))
                if not (0 <= test_x < W and 0 <= test_y < H):
                    end_x, end_y = test_x, test_y
                    break

                occupant = occ_grid[test_y, test_x]
                if occupant != 0:
                    end_x, end_y = test_x, test_y
                    if occupant == 1:
                        color = RAY_COLORS["wall"]
                    else:
                        hit_team = occupant
                        color = RAY_COLORS["enemy"] if my_team != hit_team else RAY_COLORS["ally"]
                    break
            else:
                end_x = agent_x + dx * vision_range
                end_y = agent_y + dy * vision_range

            end_pos_world = self.cam.world_to_screen(int(end_x), int(end_y))
            end_pos_screen = (
                wrect.x + end_pos_world[0] + c // 2,
                wrect.y + end_pos_world[1] + c // 2,
            )
            pygame.draw.line(surf, color, start_pos_screen, end_pos_screen, 1)

    def _draw_selected_cell(self, surf, wrect, c):
        """Draw an outline around the currently selected world cell."""
        info = self.viewer.get_selected_cell_zone_info()
        if not info:
            return

        gx, gy = info["cell"]
        cx, cy = self.cam.world_to_screen(gx, gy)
        rect = pygame.Rect(wrect.x + cx, wrect.y + cy, c, c)

        if info["edit_locked"]:
            outline_color = COLORS["zone_locked"]
        else:
            label = info["state_label"]
            if label == "Beneficial":
                outline_color = COLORS["zone_positive"]
            elif label == "Harmful":
                outline_color = COLORS["zone_negative"]
            else:
                outline_color = COLORS["selected"]

        pygame.draw.rect(surf, outline_color, rect, max(1, c // 4))
        if c >= 12:
            center = (rect.x + rect.width // 2, rect.y + rect.height // 2)
            pygame.draw.circle(surf, outline_color, center, max(1, c // 8))
# ==============================================================================
# HUD Panel (bottom)
# ==============================================================================
class HudPanel:
    """
    Bottom bar rendering: tick counter, pause/speed indicator, team stats,
    score graph, minimap.

    Maintains:
    - score_history: rolling buffer of score differences for graph.
    """

    def __init__(self, viewer, stats):
        self.viewer = viewer
        self.stats = stats

        # Deque keeps last N items efficiently (O(1) append/pop)
        self.score_history = collections.deque(maxlen=1000)

    def update(self):
        """Record the current score difference each tick (red - blue)."""
        self.score_history.append(self.stats.red.score - self.stats.blue.score)

    def draw(self, surf, state_data):
        hud = self.viewer.layout.hud_rect()
        surf.fill(COLORS["hud_bg"], hud)

        y, x = hud.y + 8, 12

        pause_str = "[ PAUSED ]" if self.viewer.paused else f"[ {self.viewer.speed_multiplier}x ]"

        surf.blit(
            self.viewer.text_cache.render(
                f"Tick {self.stats.tick} {pause_str}",
                16,
                COLORS["pause_text"] if self.viewer.paused else COLORS["text"],
            ),
            (x, y),
        )

        self._draw_team_stats(surf, y, x, state_data)
        self._draw_catastrophe_status(surf, y, x)
        self._draw_score_graph(surf, hud)

        # Minimap is a separate component
        self.viewer.minimap.draw(surf, hud, state_data)

    def _draw_team_stats(self, surf, y, x, state_data):
        """
        Show red/blue stats.

        We compute alive counts from `agent_map` (CPU cache).
        Each entry is:
          (x, y, unit, team, uid, btype)
        """
        r_alive = sum(1 for _, _, _, team, _, _ in state_data["agent_map"].values() if team == 2.0)
        b_alive = len(state_data["agent_map"]) - r_alive

        rs_alive = sum(
            1
            for _, _, unit, team, _, _ in state_data["agent_map"].values()
            if team == 2.0 and unit == 1.0
        )
        ra_alive = r_alive - rs_alive

        bs_alive = sum(
            1
            for _, _, unit, team, _, _ in state_data["agent_map"].values()
            if team == 3.0 and unit == 1.0
        )
        ba_alive = b_alive - bs_alive

        red_str = (
            f"Red  S:{self.stats.red.score:6.1f} CP:{self.stats.red.cp_points:4.1f} "
            f"K:{self.stats.red.kills:3d} D:{self.stats.red.deaths:3d} "
            f"Alive:{r_alive:3d} (S:{rs_alive} A:{ra_alive})"
        )
        blue_str = (
            f"Blue S:{self.stats.blue.score:6.1f} CP:{self.stats.blue.cp_points:4.1f} "
            f"K:{self.stats.blue.kills:3d} D:{self.stats.blue.deaths:3d} "
            f"Alive:{b_alive:3d} (S:{bs_alive} A:{ba_alive})"
        )

        surf.blit(self.viewer.text_cache.render(red_str, 16, COLORS["red"]), (x, y + 24))
        surf.blit(self.viewer.text_cache.render(blue_str, 16, COLORS["blue"]), (x, y + 48))

    def _draw_catastrophe_status(self, surf, y, x):
        """Draw catastrophe + scheduler state in a compact professional HUD strip."""
        status = self.viewer.get_catastrophe_status()
        scheduler = dict(status.get("scheduler", {}) or {})
        active = bool(status.get("active", False))
        system_enabled = bool(status.get("system_enabled", False))
        restored = bool(dict(status.get("metadata", {}) or {}).get("restored_from_checkpoint", False))
        if not active:
            line1 = (
                f"Catastrophe System: {'on' if system_enabled else 'off'} | "
                "Catastrophe: none | Base-zone editing: unlocked"
            )
            color1 = COLORS["text_dim"] if system_enabled else COLORS["warn"]
        else:
            restore_tag = " | Source: restored checkpoint" if restored else ""
            line1 = (
                f"Catastrophe System: {'on' if system_enabled else 'off'} | "
                f"Catastrophe: {status['display_name']} | Remaining: {int(status.get('remaining_ticks', 0))}/"
                f"{int(status.get('duration_ticks', 0))} | Cells: {int(status.get('active_cell_count', 0))} | "
                f"Base-zone editing: locked{restore_tag}"
            )
            color1 = COLORS["pause_text"] if system_enabled else COLORS["warn"]
        auto_state = 'on' if scheduler.get('enabled', False) else 'off'
        if not system_enabled:
            auto_state = f"blocked (pref {auto_state})"
        line2 = (
            f"Auto-Scheduler: {auto_state} | "
            f"State: {scheduler.get('state_label', 'idle')} | "
            f"Pressure: {float(scheduler.get('pressure', 0.0) or 0.0):.3f} | "
            f"Hazard: {float(scheduler.get('hazard_probability', 0.0) or 0.0):.3f} | "
            f"Cooldown: {int(scheduler.get('cooldown_remaining', 0) or 0)}"
        )
        surf.blit(self.viewer.text_cache.render(line1, 13, color1), (x, y + 72))
        if bool(getattr(config, "VIEWER_HUD_SHOW_SCHEDULER", True)):
            surf.blit(self.viewer.text_cache.render(line2, 13, COLORS["text_dim"]), (x, y + 92))

    def _draw_score_graph(self, surf, hud):
        """
        Draw a small graph of score difference over time.

        Mathematics:
        - We normalize scores by the maximum absolute score to fit in graph height:
            norm = score / max_abs_score
        - Then map normalized values to y-coordinates relative to graph center.

        Visualization:
        - Fill above zero in red (red lead)
        - Fill below zero in blue (blue lead)
        """
        graph_rect = pygame.Rect(hud.right - 540, hud.y + 10, 300, hud.height - 20)
        pygame.draw.rect(surf, COLORS["bg"], graph_rect)

        if len(self.score_history) < 2:
            return

        scores = np.array(self.score_history, dtype=np.float32)
        max_abs_score = float(np.max(np.abs(scores)) or 1.0)
        norm_scores = scores / max_abs_score

        denom = max(1, (len(scores) - 1))

        points = []
        for i, s in enumerate(norm_scores):
            s = float(s)  # numpy scalar -> python float

            px = graph_rect.x + (i / denom) * graph_rect.width
            py = graph_rect.centery - s * (graph_rect.height / 2.2)

            # Guard against NaN/inf
            if not (math.isfinite(px) and math.isfinite(py)):
                continue

            points.append((int(px), int(py)))

        if len(points) < 2:
            return

        # Grid lines
        for i in range(1, 4):
            pygame.draw.line(
                surf,
                COLORS["graph_grid"],
                (graph_rect.x, graph_rect.y + i * graph_rect.height / 4),
                (graph_rect.right, graph_rect.y + i * graph_rect.height / 4),
            )

        # Fill above/below center line
        red_poly_points = [(p[0], p[1]) for p in points if p[1] < graph_rect.centery]
        if red_poly_points:
            red_poly = (
                [(graph_rect.x, graph_rect.centery)]
                + red_poly_points
                + [(red_poly_points[-1][0], graph_rect.centery)]
            )
            pygame.draw.polygon(surf, COLORS["graph_red"], red_poly)

        blue_poly_points = [(p[0], p[1]) for p in points if p[1] > graph_rect.centery]
        if blue_poly_points:
            blue_poly = (
                [(graph_rect.x, graph_rect.centery)]
                + blue_poly_points
                + [(blue_poly_points[-1][0], graph_rect.centery)]
            )
            pygame.draw.polygon(surf, COLORS["graph_blue"], blue_poly)

        # Anti-aliased line for curve
        pygame.draw.aalines(surf, COLORS["text"], False, points)

        # Border and title
        pygame.draw.rect(surf, COLORS["border"], graph_rect, 1)
        surf.blit(
            self.viewer.text_cache.render("Score Lead", 13, COLORS["text_dim"]),
            (graph_rect.x + 5, graph_rect.y + 2),
        )


# ==============================================================================
# Side Panel (right)
# ==============================================================================
class SidePanel:
    """
    Right-side operator surface.

    The reconciled viewer uses three stacked cards so agent inspection, cell/zone
    inspection, and controls all remain visible without overflowing below the
    screen edge.
    """

    def __init__(self, viewer, registry):
        self.viewer = viewer
        self.registry = registry

    def _draw_card(self, surf, rect, title):
        pygame.draw.rect(surf, COLORS["hud_bg"], rect)
        pygame.draw.rect(surf, COLORS["border"], rect, 2)
        title_surf = self.viewer.text_cache.render(str(title), 16, COLORS["text"])
        surf.blit(title_surf, (rect.x + 12, rect.y + 10))
        return rect.x + 12, rect.y + 42

    def _draw_key_value_lines(self, surf, rect, lines, *, base_y, font_size=13, line_step=18):
        y = int(base_y)
        clip_prev = surf.get_clip()
        surf.set_clip(rect.inflate(-8, -8))
        for text_value, color in lines:
            surf.blit(self.viewer.text_cache.render(str(text_value), font_size, color), (rect.x + 12, y))
            y += line_step
            if y > rect.bottom - 16:
                break
        surf.set_clip(clip_prev)
        return y

    def _draw_help_card(self, surf, rect):
        x, y = self._draw_card(surf, rect, "Legend & Controls")
        font_size = max(11, int(getattr(config, "VIEWER_HELP_FONT_SIZE", 12)))
        row_step = max(14, font_size + 4)

        legend_items = [
            ("Red Soldier", COLORS["red_soldier"]),
            ("Red Archer", COLORS["red_archer"]),
            ("Blue Soldier", COLORS["blue_soldier"]),
            ("Blue Archer", COLORS["blue_archer"]),
            ("Beneficial Zone", COLORS["zone_positive"]),
            ("Harmful Zone", COLORS["zone_negative"]),
            ("Dormant Zone", COLORS["zone_dormant"]),
        ]
        col_gap = 10
        col_w = max(120, (rect.width - 36 - col_gap) // 2)
        for idx, (label, color) in enumerate(legend_items):
            col = idx % 2
            row = idx // 2
            lx = x + col * (col_w + col_gap)
            ly = y + row * row_step
            pygame.draw.rect(surf, color, (lx, ly + 2, 12, 12))
            surf.blit(self.viewer.text_cache.render(label, font_size, COLORS["text_dim"]), (lx + 18, ly))
        y = y + ((len(legend_items) + 1) // 2) * row_step + 12

        controls = [
            "LMB inspect/select", "[ / ] zone -/+", "0 reset zone", "Z zone overlay",
            "1 attenuation", "2 band dormancy", "3 left harmful", "4 right harmful",
            "5 left attenuate", "6 right attenuate", "Shift+I inversion exp.", "Shift+O dormancy exp.",
            "C clear catastrophe", "G catastrophe master", "Shift+G auto-scheduler", "Space pause", ". single-step",
            "+/- speed", "Wheel zoom", "WASD/arrows pan", "R rays / T threat",
            "B HP bars", "N brain labels", "M mark agent", "S save brain",
            "F9 checkpoint", "F11 fullscreen",
        ]
        cols = 2
        col_w = max(140, (rect.width - 36 - col_gap) // cols)
        rows = (len(controls) + cols - 1) // cols
        for idx, label in enumerate(controls):
            col = idx // rows
            row = idx % rows
            lx = x + col * (col_w + col_gap)
            ly = y + row * row_step
            if ly > rect.bottom - 18:
                continue
            surf.blit(self.viewer.text_cache.render(label, font_size, COLORS["text_dim"]), (lx, ly))

    def _draw_agent_section(self, surf, rect, state_data):
        x, y = self._draw_card(surf, rect, "Agent Inspector")
        slot_id = self.viewer.selected_slot_id

        if slot_id is None:
            lines = [("No live agent selected.", COLORS["text_dim"])]
            self._draw_key_value_lines(surf, rect, lines, base_y=y)
            return

        if slot_id not in state_data["agent_map"]:
            dead_label = (
                f"ID: {self.viewer.last_selected_uid} (Dead)"
                if self.viewer.last_selected_uid is not None
                else f"Slot: {slot_id} (Dead)"
            )
            self._draw_key_value_lines(surf, rect, [(dead_label, COLORS["warn"])], base_y=y)
            return

        _ax, _ay, _u, _t, unique_id, _btype = state_data["agent_map"][slot_id]
        uid = int(unique_id)
        agent_data = self.registry.agent_data[slot_id]
        pos = (int(agent_data[COL_X].item()), int(agent_data[COL_Y].item()))
        hp = float(agent_data[COL_HP].item())
        hp_max = float(agent_data[COL_HP_MAX].item())
        atk = float(agent_data[COL_ATK].item())
        vision = float(agent_data[COL_VISION].item())
        hp_ratio = (hp / hp_max) if hp_max > 0 else 0.0
        unit_val = float(agent_data[COL_UNIT].item())
        unit_name = "Archer" if unit_val == 2.0 else "Soldier"
        brain = self.registry.brains[slot_id]
        brain_name = describe_brain_module(brain)

        reward_total = None
        kill_credit = None
        if getattr(config, "VIEWER_SHOW_SELECTED_AGENT_SCORES", True):
            metrics = self.viewer.get_selected_agent_metrics(uid)
            reward_total = metrics.get("reward_total")
            kill_credit = metrics.get("kill_credit")

        lines = [(f"ID: {uid}", COLORS["green"])]
        if reward_total is not None:
            lines.append((f"Reward Σ: {reward_total:.2f}", COLORS["text_dim"]))
        if kill_credit is not None:
            lines.append((f"Kill Credit Σ: {kill_credit:.2f}", COLORS["text_dim"]))
        lines.extend([
            (f"Pos: ({pos[0]}, {pos[1]})", COLORS["text_dim"]),
            (f"Unit: {unit_name}", COLORS["text_dim"]),
            (f"Brain: {brain_name}", COLORS["text_dim"]),
        ])
        y = self._draw_key_value_lines(surf, rect, lines, base_y=y)

        bar_w = max(40, rect.width - 24)
        pygame.draw.rect(surf, COLORS["bar_bg"], (rect.x + 12, y, bar_w, 10))
        pygame.draw.rect(surf, COLORS["bar_fg"], (rect.x + 12, y, int(bar_w * hp_ratio), 10))
        y += 16

        lines = [
            (f"HP: {hp:.2f} / {hp_max:.2f}", COLORS["text_dim"]),
            (f"Attack: {atk:.2f}", COLORS["text_dim"]),
            (f"Vision: {vision:.0f} cells", COLORS["text_dim"]),
        ]
        if brain:
            lines.append((f"Model: {_get_model_summary(brain)}", COLORS["text_dim"]))
            lines.append((f"Params: {_param_count(brain):,}", COLORS["text_dim"]))
        self._draw_key_value_lines(surf, rect, lines, base_y=y)

    def _draw_zone_section(self, surf, rect):
        x, y = self._draw_card(surf, rect, "Cell / Zone Inspector")
        info = self.viewer.get_selected_cell_zone_info()
        if info is None:
            lines = [("Click the world to inspect a cell.", COLORS["text_dim"])]
            self._draw_key_value_lines(surf, rect, lines, base_y=y)
            return

        cp_indices = info["cp_indices"]
        cp_str = ", ".join(f"CP{idx}" for idx in cp_indices) if cp_indices else "None"
        catastrophe_line = (
            f"Catastrophe: {info['catastrophe_display_name']}"
            if info["catastrophe_active"]
            else "Catastrophe: None"
        )
        lines = [
            (f"Cell: ({info['cell'][0]}, {info['cell'][1]})", COLORS["text_dim"]),
            (f"Terrain: {info['terrain_label']}", COLORS["text_dim"]),
            ((info.get("occupant_label") or "Occupant: None"), COLORS["text_dim"]),
            (f"Base Value: {info['base_value']:+.2f} ({info['state_label']})", COLORS["zone_positive"] if info['state_label'] == 'Beneficial' else COLORS['zone_negative'] if info['state_label'] == 'Harmful' else COLORS['selected']),
            (f"Runtime Value: {info['effective_value']:+.2f} ({info['effective_state_label']})", COLORS["zone_positive"] if info['effective_state_label'] == 'Beneficial' else COLORS['zone_negative'] if info['effective_state_label'] == 'Harmful' else COLORS['selected']),
            (f"System: {'On' if info['system_enabled'] else 'Off'}", COLORS["green"] if info['system_enabled'] else COLORS['warn']),
            (catastrophe_line, COLORS["pause_text"] if info["catastrophe_active"] else COLORS["text_dim"]),
            (f"Catastrophe Here: {'Yes' if info['catastrophe_applies_here'] else 'No'}", COLORS["text_dim"]),
            (f"CP Masks: {cp_str}", COLORS["text_dim"]),
            (f"Edit Lock: {'Locked' if info['edit_locked'] else 'Unlocked'}", COLORS["zone_locked"] if info['edit_locked'] else COLORS['text_dim']),
            (f"Edit Step: ±{self.viewer.base_zone_edit_step:.2f}", COLORS["text_dim"]),
        ]
        scheduler = dict(info.get("scheduler", {}) or {})
        auto_scheduler_line = (
            f"Auto-Scheduler: {'Blocked (system off)' if not info['system_enabled'] else ('On' if scheduler.get('enabled', False) else 'Off')}"
        )
        lines.extend([
            (auto_scheduler_line, COLORS["text_dim"]),
            (f"Scheduler State: {scheduler.get('state_label', 'idle')}", COLORS["text_dim"]),
            (f"Pressure: {float(scheduler.get('pressure', 0.0) or 0.0):.3f}", COLORS["text_dim"]),
            (f"Hazard: {float(scheduler.get('hazard_probability', 0.0) or 0.0):.3f}", COLORS["text_dim"]),
            (f"Cooldown: {int(scheduler.get('cooldown_remaining', 0) or 0)}", COLORS["text_dim"]),
        ])
        y = self._draw_key_value_lines(surf, rect, lines, base_y=y)

        status = getattr(self.viewer, "zone_status_message", "")
        if status:
            status_color = COLORS.get(getattr(self.viewer, "zone_status_color_key", "text_dim"), COLORS["text_dim"])
            self._draw_key_value_lines(surf, rect, [(status, status_color)], base_y=y + 4, font_size=12, line_step=16)

    def draw(self, surf, state_data):
        srect = self.viewer.layout.side_rect()
        surf.fill(COLORS["side_bg"], srect)
        pygame.draw.rect(surf, COLORS["border"], srect, 2)
        sections = self.viewer.layout.side_sections()
        self._draw_agent_section(surf, sections["agent"], state_data)
        self._draw_zone_section(surf, sections["zone"])
        self._draw_help_card(surf, sections["help"])
# ==============================================================================
# Minimap
# ==============================================================================
class Minimap:
    """
    Displays a small overview of the map:

    - Agents drawn as tiny dots.
    - Camera viewport drawn as a rectangle.

    This helps navigation at high zoom levels.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.grid_w = viewer.grid.shape[2]
        self.grid_h = viewer.grid.shape[1]

    def draw(self, surf, hud_rect, state_data):
        # Fixed minimap width; height scales by aspect ratio H/W
        map_w = 200
        map_h = int(map_w * (self.grid_h / self.grid_w))

        map_rect = pygame.Rect(
            hud_rect.right - map_w - 20,
            hud_rect.y + (hud_rect.height - map_h) // 2,
            map_w,
            map_h
        )
        surf.fill(COLORS["empty"], map_rect)

        # Draw each agent as a tiny 2x2 pixel dot
        for x, y, _unit, team, _uid, _btype in state_data["agent_map"].values():
            dot_x = map_rect.x + int(float(x) / self.grid_w * map_w)
            dot_y = map_rect.y + int(float(y) / self.grid_h * map_h)
            color = COLORS["red"] if team == 2.0 else COLORS["blue"]
            pygame.draw.rect(surf, color, (dot_x, dot_y, 2, 2))

        # Draw camera viewport (frustum) rectangle
        if self.viewer.cam.cell_px > 0:
            cam_rect_world = pygame.Rect(
                self.viewer.cam.offset_x,
                self.viewer.cam.offset_y,
                self.viewer.layout.world_rect().width / self.viewer.cam.cell_px,
                self.viewer.layout.world_rect().height / self.viewer.cam.cell_px,
            )

            cam_rect_map = pygame.Rect(
                map_rect.x + (cam_rect_world.x / self.grid_w * map_w),
                map_rect.y + (cam_rect_world.y / self.grid_h * map_h),
                (cam_rect_world.width / self.grid_w * map_w),
                (cam_rect_world.height / self.grid_h * map_h),
            )
            pygame.draw.rect(surf, COLORS["marker"], cam_rect_map, 1)

        pygame.draw.rect(surf, COLORS["border"], map_rect, 1)


# ==============================================================================
# Input handling
# ==============================================================================
class InputHandler:
    """
    Central event handler for the Viewer.

    Responsibilities:
    - continuous key state handling (WASD/arrows for panning)
    - discrete events (keypress toggles, mouse clicks, resizing, quit)
    - signed-zone inspection / editing hotkeys
    """

    def __init__(self, viewer):
        self.viewer = viewer

    def handle(self):
        """
        Process input events for one frame.

        Returns:
            (running, advance_tick)
        """
        running, advance_tick = True, False
        keys = pygame.key.get_pressed()
        pan_speed = 10.0 / max(1.0, float(self.viewer.cam.cell_px))

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.viewer.cam.pan(-pan_speed, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.viewer.cam.pan(pan_speed, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.viewer.cam.pan(0, -pan_speed)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.viewer.cam.pan(0, pan_speed)

        wrect = self.viewer.layout.world_rect()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.VIDEORESIZE:
                if not self.viewer.fullscreen:
                    min_w = int(getattr(config, "VIEWER_MIN_WIDTH", 960))
                    min_h = int(getattr(config, "VIEWER_MIN_HEIGHT", 640))
                    self.viewer.Wpix = max(min_w, int(ev.w))
                    self.viewer.Hpix = max(min_h, int(ev.h))
                    self.viewer.windowed_size = (self.viewer.Wpix, self.viewer.Hpix)
                    self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                    self.viewer._rebuild_layout_after_window_change(auto_fit=True)

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    self.viewer.paused = not self.viewer.paused
                elif ev.key == pygame.K_PERIOD and self.viewer.paused:
                    advance_tick = True
                elif ev.key == pygame.K_m and self.viewer.selected_slot_id is not None:
                    if self.viewer.selected_slot_id in self.viewer.marked:
                        self.viewer.marked.remove(self.viewer.selected_slot_id)
                    elif len(self.viewer.marked) < 10:
                        self.viewer.marked.append(self.viewer.selected_slot_id)
                elif ev.key == pygame.K_r:
                    self.viewer.show_rays = not self.viewer.show_rays
                elif ev.key == pygame.K_t:
                    self.viewer.threat_vision_mode = not self.viewer.threat_vision_mode
                elif ev.key == pygame.K_b:
                    self.viewer.battle_view_enabled = not self.viewer.battle_view_enabled
                elif ev.key == pygame.K_n:
                    self.viewer.show_brain_types = not self.viewer.show_brain_types
                elif ev.key == pygame.K_z:
                    self.viewer.show_zone_overlay = not self.viewer.show_zone_overlay
                    self.viewer.world_renderer.static_surf = None
                    self.viewer.set_zone_status(
                        f"Signed-zone overlay {'enabled' if self.viewer.show_zone_overlay else 'disabled'}.",
                        "text_dim",
                    )
                elif ev.key in (pygame.K_1, pygame.K_KP1):
                    self.viewer.trigger_manual_catastrophe("global_attenuation")
                elif ev.key in (pygame.K_2, pygame.K_KP2):
                    self.viewer.trigger_manual_catastrophe("positive_band_dormancy")
                elif ev.key in (pygame.K_3, pygame.K_KP3):
                    self.viewer.trigger_manual_catastrophe("polarity_split_left_negative")
                elif ev.key in (pygame.K_4, pygame.K_KP4):
                    self.viewer.trigger_manual_catastrophe("polarity_split_right_negative")
                elif ev.key in (pygame.K_5, pygame.K_KP5):
                    self.viewer.trigger_manual_catastrophe("regional_attenuation_left")
                elif ev.key in (pygame.K_6, pygame.K_KP6):
                    self.viewer.trigger_manual_catastrophe("regional_attenuation_right")
                elif ev.key == pygame.K_i:
                    if ev.mod & pygame.KMOD_SHIFT:
                        self.viewer.trigger_manual_catastrophe("inversion", allow_experimental=True)
                    else:
                        self.viewer.set_zone_status("Experimental inversion is gated behind Shift+I.", "warn")
                elif ev.key == pygame.K_o:
                    if ev.mod & pygame.KMOD_SHIFT:
                        self.viewer.trigger_manual_catastrophe("full_dormancy", allow_experimental=True)
                    else:
                        self.viewer.set_zone_status("Experimental full dormancy is gated behind Shift+O.", "warn")
                elif ev.key == pygame.K_g:
                    if ev.mod & pygame.KMOD_SHIFT:
                        self.viewer.toggle_dynamic_catastrophe_scheduler()
                    else:
                        self.viewer.toggle_catastrophe_system()
                elif ev.key == pygame.K_c:
                    self.viewer.clear_manual_catastrophe()
                elif ev.key == pygame.K_LEFTBRACKET:
                    self.viewer.adjust_selected_cell_base_zone(-self.viewer.base_zone_edit_step)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    self.viewer.adjust_selected_cell_base_zone(self.viewer.base_zone_edit_step)
                elif ev.key in (pygame.K_0, pygame.K_KP0, pygame.K_BACKSPACE, pygame.K_DELETE):
                    self.viewer.reset_selected_cell_base_zone()
                elif ev.key == pygame.K_s:
                    self.viewer.save_selected_brain()
                elif ev.key == pygame.K_F9:
                    self.viewer.save_requested = True
                elif ev.key == pygame.K_F11:
                    self.viewer.fullscreen = not self.viewer.fullscreen
                    if self.viewer.fullscreen:
                        self.viewer.windowed_size = (int(self.viewer.Wpix), int(self.viewer.Hpix))
                        self.viewer.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        self.viewer.Wpix, self.viewer.Hpix = self.viewer.windowed_size
                        self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                    self.viewer._rebuild_layout_after_window_change(auto_fit=True)
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.viewer.speed_multiplier = min(16, self.viewer.speed_multiplier * 2)
                elif ev.key == pygame.K_MINUS:
                    self.viewer.speed_multiplier = max(0.25, self.viewer.speed_multiplier / 2)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1 and wrect.collidepoint(ev.pos):
                    gx, gy = self.viewer.cam.screen_to_world(ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                    self.viewer.selected_cell = (int(gx), int(gy))
                    slot_id = self.viewer.fast_grid_pick_slot(gx, gy)
                    self.viewer.selected_slot_id = slot_id if slot_id is not None and slot_id >= 0 else None
                    info = self.viewer.get_selected_cell_zone_info()
                    if info is not None:
                        self.viewer.set_zone_status(
                            f"Selected cell ({gx}, {gy}) {info['state_label'].lower()} {info['base_value']:+.2f}.",
                            "text_dim",
                        )
                    if self.viewer.selected_slot_id is not None:
                        if hasattr(self.viewer.registry, "agent_uids"):
                            self.viewer.last_selected_uid = int(self.viewer.registry.agent_uids[self.viewer.selected_slot_id].item())
                        else:
                            self.viewer.last_selected_uid = int(self.viewer.registry.agent_data[self.viewer.selected_slot_id, COL_AGENT_ID].item())
                elif ev.button == 4:
                    self.viewer.cam.zoom_at(1.12)
                    self.viewer.world_renderer.static_surf = None
                elif ev.button == 5:
                    self.viewer.cam.zoom_at(1 / 1.12)
                    self.viewer.world_renderer.static_surf = None

        return running, advance_tick
# ==============================================================================
# Animation manager (placeholder)
# ==============================================================================
class AnimationManager:
    """
    Simple placeholder for transient animations (e.g., damage flashes).

    Current behavior:
    - Stores a list of [type, position, lifetime]
    - Each update reduces lifetime; removes expired animations.
    """

    def __init__(self):
        self.animations = []

    def add(self, anim_type, pos):
        # Adds an animation with fixed lifetime of 10 frames
        self.animations.append([anim_type, pos, 10])

    def update(self):
        # Decrement lifetimes and keep those still alive
        self.animations = [[t, p, l - 1] for t, p, l in self.animations if l > 0]

    def draw(self, surf, wrect, cam):
        c = cam.cell_px
        for anim_type, pos, lifetime in self.animations:
            cx, cy = cam.world_to_screen(pos[0], pos[1])
            alpha = int(255 * (lifetime / 10))  # fade out linearly
            if anim_type == "damage":
                pygame.draw.circle(
                    surf,
                    (*COLORS["red"], alpha),
                    (wrect.x + cx + c // 2, wrect.y + cy + c // 2),
                    c // 2,
                    2
                )


# ==============================================================================
# Main Viewer class
# ==============================================================================
class Viewer:
    """
    Main application window: runs the Pygame loop and coordinates everything.

    Patch-3 additions:
    - signed-zone rendering toggle/state
    - selected cell inspector state
    - deterministic manual editing of the canonical base-zone layer
    - future-safe edit-lock hook support without implementing catastrophe logic
    """

    def __init__(self, grid: torch.Tensor, cell_size: Optional[int] = None, show_grid: bool = True):
        _center_window()
        pygame.init()
        pygame.display.set_caption("Neural Siege")

        self.grid = grid
        self.margin = 8
        self.show_grid = show_grid
        self.cam = Camera(int(cell_size or config.CELL_SIZE), grid.shape[2], grid.shape[1])

        H, W = grid.shape[1], grid.shape[2]
        side_min_w = int(getattr(config, "VIEWER_SIDE_PANEL_WIDTH", 460))
        hud_h = int(getattr(config, "VIEWER_BOTTOM_PANEL_HEIGHT", 148))
        min_w = int(getattr(config, "VIEWER_MIN_WIDTH", 1320))
        min_h = int(getattr(config, "VIEWER_MIN_HEIGHT", 820))
        try:
            display_info = pygame.display.Info()
            max_w = display_info.current_w - 80
            max_h = display_info.current_h - 120
        except pygame.error:
            max_w, max_h = 1440, 900

        world_px_w, world_px_h = W * self.cam.cell_px, H * self.cam.cell_px
        init_w = min(max_w, max(min_w, world_px_w + side_min_w + 3 * self.margin))
        init_h = min(max_h, max(min_h, world_px_h + hud_h + 2 * self.margin))
        self.Wpix, self.Hpix = int(init_w), int(init_h)
        self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)

        self.text_cache = TextCache()
        self.clock = pygame.time.Clock()
        self.engine = None
        self.registry = None
        self.stats = None

        self.selected_slot_id: Optional[int] = None
        self.last_selected_uid: Optional[int] = None
        self.selected_cell: Optional[Tuple[int, int]] = None
        self.marked: List[int] = []

        self.show_rays = False
        self.paused = False
        self.threat_vision_mode = False
        self.battle_view_enabled = False
        self.show_brain_types = False
        self.show_zone_overlay = bool(getattr(config, "VIEWER_ZONE_OVERLAY_DEFAULT", True))
        self.fullscreen = False
        self.speed_multiplier = 1.0
        self.windowed_size = (self.Wpix, self.Hpix)

        self.agent_scores: Dict[int, float] = collections.defaultdict(float)
        self.agent_kill_scores: Dict[int, float] = collections.defaultdict(float)
        self.zone_status_message: str = ""
        self.zone_status_color_key: str = "text_dim"

        self.save_requested: bool = False
        self._ckpt_last_status: str = ""

        self.STATE_REFRESH_EVERY_FRAMES = int(getattr(config, "VIEWER_STATE_REFRESH_EVERY", 2))
        self.PICK_REFRESH_EVERY_FRAMES = int(getattr(config, "VIEWER_PICK_REFRESH_EVERY", 2))

        step = float(getattr(config, "VIEWER_BASE_ZONE_EDIT_STEP", 0.25))
        if not math.isfinite(step) or step <= 0.0:
            step = 0.25
        self.base_zone_edit_step = max(0.01, min(1.0, step))

        self._cached_state_data = None
        self._cached_id_np = None
        self._cached_occ_np = None
        self._cached_alive_indices: List[int] = []
        self._cached_agent_map: Dict[int, tuple] = {}
        self._last_state_refresh_frame = -10
        self._last_pick_refresh_frame = -10
        self._last_zone_visual_signature = None

    def set_zone_status(self, message: str, color_key: str = "text_dim") -> None:
        """Store a short zone-related status line for the side-panel inspector."""
        self.zone_status_message = str(message)
        self.zone_status_color_key = str(color_key)

    def get_catastrophe_status(self) -> Dict[str, Any]:
        """Return a viewer-friendly catastrophe status payload."""
        default_scheduler = {
            "enabled": False,
            "pressure": 0.0,
            "cooldown_remaining": 0,
            "hazard_probability": 0.0,
            "state_label": "idle",
            "hard_trigger_due": False,
        }
        if self.engine is None or not hasattr(self.engine, "catastrophe_status"):
            return {
                "active": False,
                "type_name": None,
                "display_name": "None",
                "start_tick": None,
                "duration_ticks": 0,
                "remaining_ticks": 0,
                "elapsed_ticks": 0,
                "active_cell_count": 0,
                "locked_cell_count": 0,
                "metadata": {},
                "system_enabled": False,
                "scheduler": dict(default_scheduler),
            }
        status = dict(self.engine.catastrophe_status())
        status["display_name"] = _catastrophe_display_name(status)
        scheduler = dict(status.get("scheduler", {}) or {})
        for key, value in default_scheduler.items():
            scheduler.setdefault(key, value)
        status["scheduler"] = scheduler
        return status

    def _runtime_zone_visual_signature(self):
        """Return the cache key for catastrophe-driven world-overlay changes."""
        return _catastrophe_status_signature(self.get_catastrophe_status())

    def _refresh_zone_render_cache_if_needed(self) -> None:
        """Refresh the static terrain cache when catastrophe-driven visuals change."""
        signature = self._runtime_zone_visual_signature()
        if signature != self._last_zone_visual_signature:
            self._refresh_zone_render_cache()
            self._last_zone_visual_signature = signature

    def _rebuild_layout_after_window_change(self, *, auto_fit: bool = False) -> None:
        """Refresh layout-dependent surfaces after resize/fullscreen changes."""
        if self.screen is not None:
            self.Wpix, self.Hpix = self.screen.get_size()
        self.layout = LayoutManager(self)
        if getattr(self, "world_renderer", None) is not None:
            self.world_renderer.static_surf = None
        if auto_fit or bool(getattr(config, "VIEWER_AUTO_FIT_WORLD", True)):
            self._fit_camera_to_world_rect()

    def _fit_camera_to_world_rect(self) -> None:
        """Fit the world cleanly into the main viewport for a professional default view."""
        if not bool(getattr(config, "VIEWER_AUTO_FIT_WORLD", True)):
            return
        if getattr(self, "layout", None) is None:
            return
        world_rect = self.layout.world_rect()
        grid_h, grid_w = int(self.grid.shape[1]), int(self.grid.shape[2])
        if world_rect.width <= 0 or world_rect.height <= 0 or grid_w <= 0 or grid_h <= 0:
            return

        px_per_cell = min(world_rect.width / max(1, grid_w), world_rect.height / max(1, grid_h))
        px_per_cell = max(1.0, px_per_cell)
        target_zoom = px_per_cell / max(1.0, float(self.cam.base_cell))
        self.cam.zoom = max(0.25, min(8.0, float(target_zoom)))

        visible_w = world_rect.width / max(1, self.cam.cell_px)
        visible_h = world_rect.height / max(1, self.cam.cell_px)
        max_off_x = max(0.0, float(grid_w) - float(visible_w))
        max_off_y = max(0.0, float(grid_h) - float(visible_h))
        self.cam.offset_x = max(0.0, min(max_off_x, (float(grid_w) - float(visible_w)) * 0.5))
        self.cam.offset_y = max(0.0, min(max_off_y, (float(grid_h) - float(visible_h)) * 0.5))

    def _current_canonical_base_zone_map(self) -> Optional[torch.Tensor]:
        """Return the canonical signed base-zone map used to derive manual catastrophes."""
        if self.engine is None:
            return None
        zones = getattr(self.engine, "zones", None)
        if zones is not None:
            base_map = getattr(zones, "base_zone_value_map", None)
            if base_map is not None:
                return base_map
            heal_mask = getattr(zones, "heal_mask", None)
            if heal_mask is not None:
                return heal_mask.to(dtype=torch.float32)
        return getattr(self.engine, "_z_base_values", None)

    def trigger_manual_catastrophe(self, preset_key: str, *, allow_experimental: bool = False) -> bool:
        """Build and activate one concrete Patch-5 catastrophe preset from the viewer."""
        if self.engine is None or not hasattr(self.engine, "activate_catastrophe"):
            self.set_zone_status("No running engine is attached to the viewer.", "warn")
            return False

        current_status = self.get_catastrophe_status()
        if not bool(current_status.get("system_enabled", False)):
            self.set_zone_status("Catastrophe system is off. Press G to re-enable it before triggering catastrophes.", "warn")
            return False

        base_map = self._current_canonical_base_zone_map()
        if base_map is None:
            self.set_zone_status("No canonical base-zone map is available for catastrophe triggering.", "warn")
            return False

        previous_status = current_status
        try:
            spec = build_manual_catastrophe_spec(
                preset_key,
                base_zone_value_map=base_map,
            )
            metadata = dict(spec.metadata or {})
            if metadata.get("safety_tier") == "experimental" and not bool(allow_experimental):
                self.set_zone_status(
                    f"{metadata.get('display_name', spec.type_name)} is experimental and is gated behind a stronger hotkey.",
                    "warn",
                )
                return False
            replace_existing = bool(getattr(config, "CATASTROPHE_MANUAL_REPLACE_EXISTING", True))
            self.engine.activate_catastrophe(spec, replace_existing=replace_existing)
            self._refresh_zone_render_cache()
            self._last_zone_visual_signature = self._runtime_zone_visual_signature()
            current_status = self.get_catastrophe_status()
            if previous_status.get("active", False):
                msg = (
                    f"Replaced catastrophe {previous_status['display_name']} -> {current_status['display_name']} "
                    f"({int(current_status.get('remaining_ticks', 0))}/{int(current_status.get('duration_ticks', 0))} ticks)."
                )
            else:
                msg = (
                    f"Activated catastrophe {current_status['display_name']} "
                    f"({int(current_status.get('remaining_ticks', 0))}/{int(current_status.get('duration_ticks', 0))} ticks)."
                )
            self.set_zone_status(msg, "warn")
            return True
        except Exception as ex:
            self.set_zone_status(f"Catastrophe trigger failed: {type(ex).__name__}: {ex}", "warn")
            return False

    def clear_manual_catastrophe(self) -> bool:
        """Clear the currently active catastrophe from the viewer."""
        if self.engine is None or not hasattr(self.engine, "clear_active_catastrophe"):
            self.set_zone_status("No running engine is attached to the viewer.", "warn")
            return False
        if not bool(getattr(config, "CATASTROPHE_CLEAR_ENABLED", True)):
            self.set_zone_status("Manual catastrophe clearing is disabled by config.", "warn")
            return False
        try:
            payload = self.engine.clear_active_catastrophe(reason="manual_viewer_clear")
            if payload is None:
                self.set_zone_status("No active catastrophe to clear.", "text_dim")
                return False
            self._refresh_zone_render_cache()
            self._last_zone_visual_signature = self._runtime_zone_visual_signature()
            self.set_zone_status("Cleared active catastrophe.", "green")
            return True
        except Exception as ex:
            self.set_zone_status(f"Catastrophe clear failed: {type(ex).__name__}: {ex}", "warn")
            return False


    def toggle_catastrophe_system(self) -> bool:
        """Toggle the master catastrophe system kill-switch."""
        if self.engine is None or not hasattr(self.engine, "set_catastrophe_system_enabled"):
            self.set_zone_status("Catastrophe system control is unavailable.", "warn")
            return False
        current_status = self.get_catastrophe_status()
        enabled_now = bool(current_status.get("system_enabled", False))
        try:
            result = self.engine.set_catastrophe_system_enabled(not enabled_now, clear_active=True)
            if not bool(result.get("ok", False)):
                self.set_zone_status("Catastrophe system toggle failed.", "warn")
                return False
            status = dict(result.get("status", {}) or {})
            cleared = result.get("cleared", None)
            self._refresh_zone_render_cache()
            self._last_zone_visual_signature = self._runtime_zone_visual_signature()
            if bool(status.get("system_enabled", False)):
                self.set_zone_status("Catastrophe system enabled. Manual triggers and the auto-scheduler may run again.", "green")
            else:
                suffix = " Active catastrophe cleared." if cleared is not None else ""
                self.set_zone_status(f"Catastrophe system disabled. Manual triggers and the auto-scheduler are blocked.{suffix}", "warn")
            return True
        except Exception as ex:
            self.set_zone_status(f"Catastrophe system toggle failed: {type(ex).__name__}: {ex}", "warn")
            return False

    def toggle_dynamic_catastrophe_scheduler(self) -> bool:
        """Toggle the deterministic dynamic catastrophe scheduler."""
        if self.engine is None or not hasattr(self.engine, "set_dynamic_catastrophe_enabled"):
            self.set_zone_status("Dynamic catastrophe scheduler is unavailable.", "warn")
            return False
        current = self.get_catastrophe_status().get("scheduler", {})
        enabled_now = bool(current.get("enabled", False))
        try:
            new_value = not enabled_now
            ok = bool(self.engine.set_dynamic_catastrophe_enabled(new_value))
            if not ok:
                self.set_zone_status("Dynamic catastrophe scheduler toggle failed.", "warn")
                return False
            status = self.get_catastrophe_status().get("scheduler", {})
            system_enabled = bool(self.get_catastrophe_status().get("system_enabled", False))
            suffix = "" if system_enabled else " Catastrophe system is currently off, so auto-triggers remain blocked."
            self.set_zone_status(
                f"Auto-scheduler {'enabled' if status.get('enabled', False) else 'disabled'}.{suffix}",
                "green" if status.get("enabled", False) and system_enabled else "text_dim",
            )
            return True
        except Exception as ex:
            self.set_zone_status(f"Scheduler toggle failed: {type(ex).__name__}: {ex}", "warn")
            return False

    def get_selected_agent_metrics(self, uid: int) -> Dict[str, Optional[float]]:
        """Return honest per-agent cumulative metrics backed by engine-owned state."""
        out: Dict[str, Optional[float]] = {"reward_total": None, "kill_credit": None}
        try:
            reward_totals = getattr(self.engine, "agent_reward_totals", None) if self.engine is not None else None
            if reward_totals is not None and uid in reward_totals:
                out["reward_total"] = float(reward_totals[uid])
            elif uid in self.agent_scores:
                out["reward_total"] = float(self.agent_scores[uid])
        except Exception:
            pass
        try:
            kill_scores = getattr(self.engine, "agent_scores", None) if self.engine is not None else None
            if kill_scores is not None and uid in kill_scores:
                out["kill_credit"] = float(kill_scores[uid])
            elif uid in self.agent_kill_scores:
                out["kill_credit"] = float(self.agent_kill_scores[uid])
        except Exception:
            pass
        return out

    def save_selected_brain(self):
        """Save the PyTorch `state_dict` of the selected agent's brain to disk."""
        if self.selected_slot_id is None or not hasattr(self, "registry"):
            return

        brain = self.registry.brains[self.selected_slot_id]
        if brain:
            tick = self.stats.tick if hasattr(self, "stats") and self.stats is not None else 0
            uid = self.last_selected_uid if self.last_selected_uid is not None else self.selected_slot_id
            filename = f"brain_agent_{uid}_t_{tick}.pth"
            try:
                torch.save(brain.state_dict(), filename)
                kind = brain_kind_from_module(brain)
                label = brain_kind_display_name(kind) if kind else brain.__class__.__name__
                print(f"[Viewer] Saved {label} brain for agent {uid} to '{filename}'")
            except Exception as e:
                print(f"[Viewer] Error saving brain: {e}")

    def fast_grid_pick_slot(self, gx: int, gy: int) -> Optional[int]:
        """Return the agent slot ID at grid cell (gx, gy) using the cached id grid."""
        if self._cached_id_np is None:
            try:
                v = int(self.grid[2, gy, gx].item())
                return v if v >= 0 else None
            except Exception:
                return None

        if 0 <= gy < self._cached_id_np.shape[0] and 0 <= gx < self._cached_id_np.shape[1]:
            v = int(self._cached_id_np[gy, gx])
            return v if v >= 0 else None
        return None

    def get_selected_cell_zone_info(self) -> Optional[Dict[str, Any]]:
        """
        Return inspection information for the currently selected cell.

        Patch-5 extension:
        - keep canonical base-zone truth visible
        - also expose the runtime-effective value seen by agents during an active catastrophe
        - report whether the current catastrophe applies at this cell
        - report whether base-zone editing is locked right now
        """
        if self.selected_cell is None:
            return None

        gx, gy = int(self.selected_cell[0]), int(self.selected_cell[1])
        H, W = int(self.grid.shape[1]), int(self.grid.shape[2])
        if not (0 <= gx < W and 0 <= gy < H):
            return None

        if self._cached_occ_np is not None:
            occ = int(self._cached_occ_np[gy, gx])
        else:
            occ = int(self.grid[0, gy, gx].item())

        terrain_label = "Wall" if occ == 1 else "Traversable"
        slot_id = self.fast_grid_pick_slot(gx, gy)
        occupant_label = None
        occupant_uid = None
        if slot_id is not None and slot_id in self._cached_agent_map:
            _ax, _ay, _unit, team, uid, _btype = self._cached_agent_map[slot_id]
            team_label = "Red" if float(team) == 2.0 else "Blue"
            occupant_uid = int(uid)
            occupant_label = f"Occupant: {team_label} Agent #{occupant_uid}"
        elif occ == 2:
            occupant_label = "Occupant: Red Agent"
        elif occ == 3:
            occupant_label = "Occupant: Blue Agent"

        base_value = 0.0
        effective_value = 0.0
        cp_indices: List[int] = []
        edit_locked = False
        catastrophe_applies_here = False
        catastrophe_locked_here = False
        catastrophe_status = self.get_catastrophe_status()

        zones = getattr(self.engine, "zones", None) if self.engine is not None else None
        if zones is not None:
            try:
                if hasattr(zones, "get_base_zone_value_at"):
                    base_value = float(zones.get_base_zone_value_at(gx, gy))
                elif getattr(zones, "base_zone_value_map", None) is not None:
                    base_value = float(zones.base_zone_value_map[gy, gx].item())
                elif getattr(zones, "heal_mask", None) is not None:
                    base_value = 1.0 if bool(zones.heal_mask[gy, gx].item()) else 0.0
            except Exception:
                base_value = 0.0

            try:
                if hasattr(zones, "is_base_zone_edit_locked_at"):
                    edit_locked = bool(zones.is_base_zone_edit_locked_at(gx, gy))
            except Exception:
                edit_locked = False

            for idx, mask in enumerate(getattr(zones, "cp_masks", []) or []):
                try:
                    if bool(mask[gy, gx].item()):
                        cp_indices.append(idx)
                except Exception:
                    continue

        effective_value = float(base_value)
        if self.engine is not None:
            try:
                effective_map = getattr(self.engine, "_z_effective_values", None)
                if effective_map is not None:
                    effective_value = float(effective_map[gy, gx].item())
            except Exception:
                effective_value = float(base_value)

            try:
                apply_mask = getattr(self.engine, "_z_catastrophe_apply_mask", None)
                if apply_mask is not None:
                    catastrophe_applies_here = bool(apply_mask[gy, gx].item())
            except Exception:
                catastrophe_applies_here = False

            try:
                lock_mask = getattr(self.engine, "_z_catastrophe_edit_lock_mask", None)
                if lock_mask is not None:
                    catastrophe_locked_here = bool(lock_mask[gy, gx].item())
            except Exception:
                catastrophe_locked_here = False

        return {
            "cell": (gx, gy),
            "terrain_label": terrain_label,
            "occupant_label": occupant_label,
            "occupant_uid": occupant_uid,
            "slot_id": slot_id,
            "base_value": float(max(-1.0, min(1.0, base_value))),
            "state_label": _zone_state_label(base_value),
            "effective_value": float(max(-1.0, min(1.0, effective_value))),
            "effective_state_label": _zone_state_label(effective_value),
            "cp_indices": cp_indices,
            "edit_locked": bool(edit_locked),
            "catastrophe_active": bool(catastrophe_status.get("active", False)),
            "catastrophe_display_name": catastrophe_status.get("display_name", "None"),
            "catastrophe_applies_here": bool(catastrophe_applies_here),
            "catastrophe_locked_here": bool(catastrophe_locked_here),
            "system_enabled": bool(catastrophe_status.get("system_enabled", False)),
            "scheduler": dict(catastrophe_status.get("scheduler", {}) or {}),
        }

    def _refresh_zone_render_cache(self) -> None:
        """Rebuild cached signed-zone render data after a manual edit."""
        if getattr(self, "world_renderer", None) is not None and self.engine is not None:
            self.world_renderer.build_static_cache(self.engine)
        elif getattr(self, "world_renderer", None) is not None:
            self.world_renderer.static_surf = None

    def _set_selected_cell_base_zone_value(self, new_value: float) -> bool:
        """
        Write a new signed value into the canonical base-zone layer for the selected cell.

        Contract:
        - edits the base layer only
        - clamps into [-1, +1]
        - refreshes engine-side cached zone tensors
        - refreshes the renderer's static signed-zone cache
        """
        info = self.get_selected_cell_zone_info()
        if info is None:
            self.set_zone_status("Select a world cell before editing zones.", "warn")
            return False
        if self.engine is None or getattr(self.engine, "zones", None) is None:
            self.set_zone_status("No zone container is attached to the running engine.", "warn")
            return False

        catastrophe_status = self.get_catastrophe_status()
        if bool(catastrophe_status.get("active", False)):
            self.set_zone_status(
                f"Base-zone editing is locked while catastrophe {catastrophe_status['display_name']} is active.",
                "warn",
            )
            return False

        gx, gy = info["cell"]
        zones = self.engine.zones
        try:
            with torch.no_grad():
                if hasattr(zones, "set_base_zone_value_at"):
                    stored = float(zones.set_base_zone_value_at(gx, gy, float(new_value), respect_edit_lock=True))
                else:
                    base_map = getattr(zones, "base_zone_value_map", None)
                    if base_map is None:
                        raise RuntimeError("engine.zones has no canonical base_zone_value_map")
                    if hasattr(zones, "is_base_zone_edit_locked_at") and zones.is_base_zone_edit_locked_at(gx, gy):
                        raise RuntimeError(f"base-zone cell ({gx}, {gy}) is locked for manual editing")
                    base_map[gy, gx] = float(max(-1.0, min(1.0, float(new_value))))
                    stored = float(base_map[gy, gx].item())

                if hasattr(self.engine, "_ensure_zone_tensors"):
                    self.engine._ensure_zone_tensors()

            self._refresh_zone_render_cache()
            self.set_zone_status(
                f"Cell ({gx}, {gy}) base zone -> {stored:+.2f} ({_zone_state_label(stored).lower()}).",
                "green",
            )
            return True
        except Exception as ex:
            self.set_zone_status(f"Base-zone edit blocked: {type(ex).__name__}: {ex}", "warn")
            return False

    def adjust_selected_cell_base_zone(self, delta: float) -> bool:
        """Add a signed delta to the selected cell's canonical base-zone value."""
        info = self.get_selected_cell_zone_info()
        if info is None:
            self.set_zone_status("Select a world cell before editing zones.", "warn")
            return False
        return self._set_selected_cell_base_zone_value(info["base_value"] + float(delta))

    def reset_selected_cell_base_zone(self) -> bool:
        """Reset the selected cell's canonical base-zone value to dormant/neutral."""
        return self._set_selected_cell_base_zone_value(0.0)

    def _refresh_state_cpu(self):
        """
        Copy the minimal data needed for rendering/picking from GPU to CPU.
        """
        with torch.no_grad():
            grid_cpu = self.grid.detach().cpu()
            occ_np = grid_cpu[0].numpy().astype(np.int16, copy=False)
            id_np = grid_cpu[2].numpy().astype(np.int32, copy=False)
            ad = self.registry.agent_data
            alive_mask = ad[:, COL_ALIVE] > 0.5
            alive_idx_t = torch.nonzero(alive_mask).squeeze(1)

            if alive_idx_t.numel() == 0:
                alive_indices: List[int] = []
                agent_map: Dict[int, tuple] = {}
            else:
                alive_indices = alive_idx_t.cpu().tolist()
                cols = [COL_X, COL_Y, COL_UNIT, COL_TEAM, COL_AGENT_ID]
                alive_data = ad.index_select(0, alive_idx_t)[:, cols].detach().cpu().numpy()
                uid_data = None
                if hasattr(self.registry, "agent_uids"):
                    uid_data = self.registry.agent_uids.index_select(0, alive_idx_t).detach().cpu().numpy()

                brains = self.registry.brains
                agent_map = {}
                for k, slot_id in enumerate(alive_indices):
                    x, y, unit, team, uid = alive_data[k]
                    if uid_data is not None:
                        uid = uid_data[k]
                    br = brains[slot_id]
                    if br is None:
                        btype = "?"
                    else:
                        kind = brain_kind_from_module(br)
                        btype = brain_kind_short_label(kind) if kind else "?"
                    agent_map[slot_id] = (float(x), float(y), float(unit), float(team), float(uid), btype)

            self._cached_occ_np = occ_np
            self._cached_id_np = id_np
            self._cached_alive_indices = alive_indices
            self._cached_agent_map = agent_map
            self._cached_state_data = {
                "occ_np": self._cached_occ_np,
                "id_np": self._cached_id_np,
                "alive_indices": self._cached_alive_indices,
                "agent_map": self._cached_agent_map,
            }

    def _install_score_hook(self, engine, registry):
        """Bind viewer score displays to engine-owned cumulative metrics."""
        reward_totals = getattr(engine, "agent_reward_totals", None)
        if reward_totals is not None:
            self.agent_scores = reward_totals
        kill_scores = getattr(engine, "agent_scores", None)
        if kill_scores is not None:
            self.agent_kill_scores = kill_scores

    def run(
        self,
        engine,
        registry,
        stats,
        tick_limit: int = 0,
        target_fps: Optional[int] = None,
        run_dir: Optional[str] = None,
    ):
        """Start the viewer loop."""
        self.engine = engine
        self.registry = registry
        self.stats = stats

        ckpt_mgr = None
        if run_dir is not None:
            from utils.checkpointing import CheckpointManager
            ckpt_mgr = CheckpointManager(Path(run_dir))

        self.layout = LayoutManager(self)
        self._fit_camera_to_world_rect()
        self.world_renderer = WorldRenderer(self, self.grid, registry)
        self.hud_panel = HudPanel(self, stats)
        self.side_panel = SidePanel(self, registry)
        self.input_handler = InputHandler(self)
        self.anim_manager = AnimationManager()
        self.minimap = Minimap(self)

        self.world_renderer.build_static_cache(engine)
        self._last_zone_visual_signature = self._runtime_zone_visual_signature()
        self._install_score_hook(engine, registry)

        running = True
        frame_count = 0
        self._refresh_state_cpu()
        self._last_state_refresh_frame = frame_count
        self._last_pick_refresh_frame = frame_count

        while running:
            running, advance_tick = self.input_handler.handle()

            if self.save_requested:
                self.save_requested = False
                if ckpt_mgr is None:
                    self._ckpt_last_status = "Checkpoint save requested but run_dir is not set."
                    print("[checkpoint]", self._ckpt_last_status)
                else:
                    try:
                        viewer_state = {
                            "paused": bool(self.paused),
                            "speed_mult": float(self.speed_multiplier),
                            "camera": {
                                "offset_x": float(getattr(self.cam, "offset_x", 0.0)),
                                "offset_y": float(getattr(self.cam, "offset_y", 0.0)),
                                "zoom": float(getattr(self.cam, "zoom", 1.0)),
                            },
                            "agent_reward_totals": dict(self.agent_scores),
                            "agent_kill_scores": dict(self.agent_kill_scores),
                        }
                        out_dir = ckpt_mgr.save_atomic(
                            engine=engine,
                            registry=registry,
                            stats=stats,
                            viewer_state=viewer_state,
                            notes="manual_hotkey_F9",
                            pinned=bool(getattr(config, "CHECKPOINT_PIN_ON_MANUAL", True)),
                            pin_tag=str(getattr(config, "CHECKPOINT_PIN_TAG", "manual")),
                        )
                        ckpt_mgr.prune_keep_last_n(int(getattr(config, "CHECKPOINT_KEEP_LAST_N", 1)))
                        self._ckpt_last_status = f"Checkpoint saved: {out_dir.name}"
                        print("[checkpoint]", self._ckpt_last_status)
                    except Exception as ex:
                        self._ckpt_last_status = f"Checkpoint FAILED: {type(ex).__name__}: {ex}"
                        print("[checkpoint]", self._ckpt_last_status)

            num_ticks_this_frame = 0
            if not self.paused:
                if self.speed_multiplier >= 1:
                    num_ticks_this_frame = int(self.speed_multiplier)
                elif self.speed_multiplier > 0:
                    denom = int(1 / self.speed_multiplier)
                    if denom <= 1 or (frame_count % denom) == 0:
                        num_ticks_this_frame = 1
            elif advance_tick:
                num_ticks_this_frame = 1

            for _ in range(num_ticks_this_frame):
                engine.run_tick()
                self.hud_panel.update()
                if ckpt_mgr is not None:
                    trig = Path(run_dir) / str(getattr(config, "CHECKPOINT_TRIGGER_FILE", "checkpoint.now"))
                    ckpt_mgr.maybe_save_trigger_file(
                        trigger_path=trig,
                        engine=engine,
                        registry=registry,
                        stats=stats,
                        default_pin=bool(getattr(config, "CHECKPOINT_PIN_ON_MANUAL", True)),
                        pin_tag=str(getattr(config, "CHECKPOINT_PIN_TAG", "manual")),
                        keep_last_n=int(getattr(config, "CHECKPOINT_KEEP_LAST_N", 1)),
                    )
                    ckpt_mgr.maybe_save_periodic(
                        engine=engine,
                        registry=registry,
                        stats=stats,
                        every_ticks=int(getattr(config, "CHECKPOINT_EVERY_TICKS", 0)),
                        keep_last_n=int(getattr(config, "CHECKPOINT_KEEP_LAST_N", 1)),
                    )

            self._refresh_zone_render_cache_if_needed()

            if (frame_count - self._last_state_refresh_frame) >= self.STATE_REFRESH_EVERY_FRAMES:
                self._refresh_state_cpu()
                self._last_state_refresh_frame = frame_count

            if (frame_count - self._last_pick_refresh_frame) >= self.PICK_REFRESH_EVERY_FRAMES:
                self._last_pick_refresh_frame = frame_count

            state_data = self._cached_state_data
            if state_data is None:
                self._refresh_state_cpu()
                state_data = self._cached_state_data

            self.screen.fill(COLORS["bg"])
            self.world_renderer.draw(self.screen, state_data)
            self.hud_panel.draw(self.screen, state_data)
            self.side_panel.draw(self.screen, state_data)
            pygame.display.flip()
            self.clock.tick(int(target_fps or config.TARGET_FPS))
            frame_count += 1

            if tick_limit > 0 and stats.tick >= tick_limit:
                running = False

        pygame.quit()




