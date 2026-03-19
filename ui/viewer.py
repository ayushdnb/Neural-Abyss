"""Real-time viewer for Neural Abyss."""

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

# Constants & colour palette
FONT_NAME = "consolas"

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

    # UI accents
    "archer_glyph": (245, 230, 90),
    "marker": (242, 228, 92),

    "text": (230, 230, 230),
    "text_dim": (180, 186, 194),

    "green": (46, 204, 113),
    "warn": (243, 156, 18),
    "heal_active": (46, 204, 113),
    "heal_dormant": (214, 163, 72),
    "button_on": (46, 204, 113),
    "button_off": (192, 57, 43),
    "button_border": (96, 102, 112),

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
    "heal_active": (46, 204, 113, 78),
    "heal_dormant": (214, 163, 72, 110),
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


# Small safety helpers for color handling
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


# Utility functions
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


# Text caching
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


# Layout management
class LayoutManager:
    """
    Computes rectangles (pygame.Rect) for:
      - world view (main grid)
      - side panel (right)
      - bottom HUD (stats + minimap + graph)

    Why:
    - Centralizes layout logic.
    - Adapts to window resizing.
    - Avoids hard-coded coordinates scattered across the code.
    """

    def __init__(self, viewer):
        self.viewer = viewer

    def side_width(self):
        """
        Adaptive width for the side panel.

        Rule:
        - At least 320 px
        - At most 420 px
        - Roughly 27% of total width
        """
        return max(320, min(420, int(self.viewer.Wpix * 0.27)))

    def world_rect(self):
        """
        Rectangle where the game world is drawn.

        We subtract:
        - margin(s)
        - side panel width
        - HUD height (fixed 126 px)
        """
        m = self.viewer.margin
        return pygame.Rect(
            m,
            m,
            max(64, self.viewer.Wpix - self.side_width() - 3 * m),
            max(64, self.viewer.Hpix - 126 - 2 * m),
        )

    def side_rect(self):
        """Rectangle for the side panel (agent inspector + legend)."""
        m, side_w = self.viewer.margin, self.side_width()
        return pygame.Rect(
            self.viewer.Wpix - side_w - m,
            m,
            side_w,
            max(64, self.viewer.Hpix - 126 - 2 * m),
        )

    def hud_rect(self):
        """Rectangle for the bottom HUD bar."""
        return pygame.Rect(0, self.viewer.Hpix - 126, self.viewer.Wpix, 126)


# World renderer
class WorldRenderer:
    """
    Draws the world grid, agents, and overlays into the Pygame window.

    PERFORMANCE STRATEGY:
    - We draw static terrain (walls, empties, heal zones) into a cached Surface.
      This is expensive because it loops over grid cells.
    - We rebuild this cache only when needed:
        - viewport size changes
        - zoom changes
        - terrain changes
    - Dynamic items (agents, HP bars, rays) are drawn every frame.

    Inputs:
    - viewer (has camera, flags, text cache)
    - grid tensor (PyTorch)
    - registry (per-agent tensor state)
    """

    def __init__(self, viewer, grid, registry):
        self.viewer = viewer
        self.grid = grid
        self.registry = registry
        self.cam = viewer.cam

        # Cached background surface (static terrain)
        self.static_surf = None

        # Precomputed zone info extracted from engine.zones.
        # We keep active and dormant heal cells separate so the viewer can make
        # catastrophe suppression visually obvious at a glance.
        self._zone_cache = {
            "heal_active_tiles": [],
            "heal_dormant_tiles": [],
            "cp_rects": [],
        }
        self._zone_render_signature = None

    def invalidate_static_background(self):
        """Force the terrain/heal static surface to rebuild on next draw."""
        self.static_surf = None

    def _compute_zone_render_signature(self, engine):
        zones = getattr(engine, "zones", None)
        if zones is None:
            return ("no_zones",)
        catastrophe_sig = None
        if getattr(zones, "catastrophe_state", None) is not None:
            catastrophe_sig = (
                str(zones.catastrophe_state.event_id),
                tuple(str(zid) for zid in zones.catastrophe_state.suppressed_zone_ids),
            )
        return (
            tuple((str(zone.zone_id), tuple(int(v) for v in zone.bounds)) for zone in getattr(zones, "heal_zones", [])),
            tuple(str(zid) for zid in getattr(zones, "active_heal_zone_ids", tuple())),
            tuple(sorted((str(k), bool(v)) for k, v in dict(getattr(zones, "manual_zone_enabled", {})).items())),
            catastrophe_sig,
            int(len(getattr(zones, "cp_masks", []) or [])),
        )

    def maybe_refresh_zone_cache(self, engine, *, force: bool = False):
        """Rebuild cached heal-zone rendering data only when zone state changed."""
        sig = self._compute_zone_render_signature(engine)
        if force or sig != self._zone_render_signature:
            self.build_static_cache(engine)

    def build_static_cache(self, engine):
        """
        Extract active/dormant heal tiles and control-point bounds from engine.zones.

        Active vs dormant heal cells are cached separately so catastrophe/manual
        state changes can invalidate and redraw them explicitly instead of relying
        on stale merged overlays.
        """
        self.invalidate_static_background()
        self._zone_cache = {
            "heal_active_tiles": [],
            "heal_dormant_tiles": [],
            "cp_rects": [],
        }

        zones = getattr(engine, "zones", None)
        self._zone_render_signature = self._compute_zone_render_signature(engine)
        if not zones:
            return

        active_zone_ids = set(str(zid) for zid in getattr(zones, "active_heal_zone_ids", tuple()))
        active_tiles = set()
        dormant_tiles = set()

        for zone in getattr(zones, "heal_zones", []):
            ys, xs = torch.nonzero(zone.mask, as_tuple=True)
            coords = set(zip(xs.cpu().tolist(), ys.cpu().tolist()))
            if str(zone.zone_id) in active_zone_ids:
                active_tiles |= coords
            else:
                dormant_tiles |= coords

        dormant_tiles.difference_update(active_tiles)
        self._zone_cache["heal_active_tiles"] = sorted(active_tiles)
        self._zone_cache["heal_dormant_tiles"] = sorted(dormant_tiles)

        # Control points: each cp_mask -> bounding rectangle (x0,y0,x1,y1)
        for m in getattr(zones, "cp_masks", []):
            ys, xs = torch.nonzero(m, as_tuple=True)
            if xs.numel() > 0:
                self._zone_cache["cp_rects"].append(
                    (xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1)
                )

    def _draw_static_background(self):
        """
        Render walls, empty cells, and heal zones onto a static surface.

        KEY DETAIL:
        - We draw into a surface sized to the world_rect.
        - We use camera transforms world_to_screen to place cells.

        NOTE:
        - This loops over every cell in the grid (O(H*W)).
        - Therefore we do NOT want to call it every frame.
        """
        wrect = self.viewer.layout.world_rect()
        self.static_surf = pygame.Surface(wrect.size)
        self.static_surf.fill(COLORS["bg"])

        # Grid layout: grid tensor is typically (C, H, W)
        H, W = self.grid.shape[1], self.grid.shape[2]

        # We only need occupancy channel for terrain rendering.
        # This is an expensive GPU->CPU sync; done only when rebuilding static surf.
        occ_np = self.grid[0].detach().cpu().numpy()

        # Draw each cell: empty (0) or wall (1)
        for y in range(H):
            for x in range(W):
                occ = occ_np[y, x]
                if occ in {0, 1}:
                    color = COLORS["empty"] if occ == 0 else COLORS["wall"]
                    cx, cy = self.cam.world_to_screen(x, y)
                    pygame.draw.rect(
                        self.static_surf,
                        color,
                        (cx, cy, self.cam.cell_px, self.cam.cell_px),
                    )

        # Heal zones drawn as alpha overlay (separate surface with SRCALPHA).
        # Dormant tiles are intentionally visible; active tiles are then drawn on
        # top in the normal heal color so current effective support is obvious.
        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for key, overlay_name in (("heal_dormant_tiles", "heal_dormant"), ("heal_active_tiles", "heal_active")):
            for x, y in self._zone_cache[key]:
                cx, cy = self.cam.world_to_screen(x, y)
                pygame.draw.rect(
                    overlay,
                    OVERLAYS[overlay_name],
                    (cx, cy, self.cam.cell_px, self.cam.cell_px),
                )
        self.static_surf.blit(overlay, (0, 0))

    def draw(self, surf, state_data):
        """
        Draw the entire world.

        Parameters:
        - surf: the main screen Surface
        - state_data: CPU cached dictionary from Viewer._refresh_state_cpu()

        Expected keys in state_data:
        - "occ_np": occupancy grid as numpy (H,W)
        - "id_np": id grid as numpy (H,W)
        - "alive_indices": list of slot ids alive
        - "agent_map": dict slot_id -> (x, y, unit, team, uid, brain_type_str)
        """
        wrect = self.viewer.layout.world_rect()

        # Ensure static background exists and matches current viewport size
        if self.static_surf is None or self.static_surf.get_size() != wrect.size:
            self._draw_static_background()

        # 1) Blit static background into the world rect
        surf.blit(self.static_surf, wrect.topleft)

        # Cell size in pixels (depends on camera zoom)
        c = self.cam.cell_px

        # Control point outlines
        # Draw into an alpha surface so we can overlay semi-transparent outlines.
        cp_overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)

        for x0, y0, x1, y1 in self._zone_cache["cp_rects"]:
            # Determine which team controls the control point by counting tiles
            # inside this region in occupancy map:
            #   2 => red team, 3 => blue team (project convention)
            patch = state_data["occ_np"][y0:y1, x0:x1]
            red_cnt = (patch == 2).sum()
            blue_cnt = (patch == 3).sum()

            # Pick outline color and label
            if red_cnt > blue_cnt:
                b_col, label = OVERLAYS["outline_red"], ("R", COLORS["red"])
            elif blue_cnt > red_cnt:
                b_col, label = OVERLAYS["outline_blue"], ("B", COLORS["blue"])
            else:
                b_col, label = OVERLAYS["outline_neutral"], ("–", COLORS["text_dim"])

            # Convert world rectangle corners to screen positions
            cx0, cy0 = self.cam.world_to_screen(x0, y0)
            cx1, cy1 = self.cam.world_to_screen(x1, y1)

            rect = pygame.Rect(cx0, cy0, cx1 - cx0, cy1 - cy0)

            # Outline thickness scales with zoom/cell size (at least 1 px)
            pygame.draw.rect(cp_overlay, b_col, rect, max(1, c // 2))

            # Label in the center of the rect
            lab_surf = self.viewer.text_cache.render(label[0], 13, label[1])
            cp_overlay.blit(lab_surf, lab_surf.get_rect(center=rect.center))

        surf.blit(cp_overlay, wrect.topleft)

        # Agents
        for slot_id in state_data["alive_indices"]:
            entry = state_data["agent_map"].get(slot_id, None)
            if entry is None:
                continue

            x, y, unit, team, _uid, _btype = entry

            # Choose color based on (team, unit)
            # Convention:
            # - team 2.0 = red
            # - team 3.0 = blue
            # - unit 1.0 = soldier
            # - unit 2.0 = archer
            color_key = f"{'red' if team == 2.0 else 'blue'}_{'archer' if unit == 2.0 else 'soldier'}"
            color = COLORS[color_key]

            # Convert world cell to pixel coords
            cx, cy = self.cam.world_to_screen(int(x), int(y))

            # NOTE: For agents we draw directly into `surf`, but offset by world_rect
            # because world_to_screen is relative to the world surface coordinates.
            pygame.draw.rect(surf, color, (wrect.x + cx, wrect.y + cy, c, c))

            # If archer and cell is big enough, draw an inner circle glyph
            if unit == 2.0 and c > 4:
                pygame.draw.circle(
                    surf,
                    COLORS["archer_glyph"],
                    (wrect.x + cx + c // 2, wrect.y + cy + c // 2),
                    max(2, c // 2 - 1),
                    max(1, c // 6),
                )

        # Optional overlays controlled by flags/toggles
        if self.viewer.battle_view_enabled:
            self._draw_hp_bars(surf, wrect, c, state_data)
        if self.viewer.show_brain_types:
            self._draw_brain_labels(surf, wrect, c, state_data)
        if self.viewer.threat_vision_mode:
            self._draw_threat_vision(surf, wrect, c, state_data)
        if self.viewer.show_grid and c >= 6:
            self._draw_grid_lines(surf, wrect, c)

        # Markers always drawn (but only if viewer.marked is non-empty)
        self._draw_markers(surf, wrect, c, state_data["id_np"])

        # Rays depend on selected agent + toggle
        if self.viewer.show_rays:
            self._draw_rays(surf, wrect, c, state_data)

    def _draw_hp_bars(self, surf, wrect, c, state_data):
        """
        Draw health bars above each agent.

        Rendering note:
        - Bars are drawn only if the cell is large enough, otherwise it becomes noise.

        Math note:
        - hp_ratio = hp / hp_max in [0,1]
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
            hp_ratio = max(0.0, min(1.0, hp_ratio_raw))  # clamp [0,1]

            cx, cy = self.cam.world_to_screen(int(x), int(y))

            bar_w, bar_h = c, max(1, c // 8)
            bar_y = cy - bar_h - 2

            # Basic bounds check to avoid drawing off-surface
            if 0 <= bar_y < wrect.height and 0 <= cx < wrect.width:
                # background
                pygame.draw.rect(hp_bar_surf, (50, 50, 50), (cx, bar_y, bar_w, bar_h))
                # foreground proportionally filled
                pygame.draw.rect(hp_bar_surf, COLORS["green"], (cx, bar_y, bar_w * hp_ratio, bar_h))

        surf.blit(hp_bar_surf, wrect.topleft)

    def _draw_brain_labels(self, surf, wrect, c, state_data):
        """
        Draw small labels inside agent cells indicating brain type.

        - We use a scaled font size relative to cell size.
        - btype is a short string like "T", "M", "Tr", "U", etc.
        """
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
                rect = lab_surf.get_rect(
                    center=(wrect.x + cx + c // 2, wrect.y + cy + c // 2)
                )
                surf.blit(lab_surf, rect)

    def _draw_threat_vision(self, surf, wrect, c, state_data):
        """
        Threat vision mode:

        - Draw a vision range circle for the selected agent.
        - Highlight agents within that range.
        - Enemies get filled circles whose size/alpha depends on their HP ratio.
        - Allies get a small outlined circle.

        Geometry note:
        - Distance computed using Euclidean norm:
            dist = sqrt((dx)^2 + (dy)^2)
          implemented via numpy.hypot(dx, dy).
        """
        slot_id = self.viewer.selected_slot_id
        if slot_id is None or slot_id not in state_data["agent_map"]:
            return

        my_x, my_y, _unit, my_team, _uid, _btype = state_data["agent_map"][slot_id]
        my_x = float(my_x)
        my_y = float(my_y)

        my_cx, my_cy = self.cam.world_to_screen(int(my_x), int(my_y))

        vision_range = float(self.registry.agent_data[slot_id, COL_VISION].item())
        vision_px = int(vision_range * c)  # convert cells -> pixels

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
                    # Enemy: circle scales with hp_ratio; alpha scales with hp_ratio
                    radius = int(c * 0.7 * (0.5 + hp_ratio * 0.5))
                    alpha = _clamp_u8(50 + 150 * hp_ratio)
                    r, g, b = _rgb(OVERLAYS["threat_enemy"])
                    pygame.draw.circle(
                        overlay, (r, g, b, alpha),
                        (o_cx + c // 2, o_cy + c // 2),
                        radius
                    )
                else:
                    # Ally: small outlined circle
                    r, g, b = _rgb(OVERLAYS["threat_ally"])
                    pygame.draw.circle(
                        overlay, (r, g, b, 100),
                        (o_cx + c // 2, o_cy + c // 2),
                        int(c * 0.4),
                        1
                    )

        surf.blit(overlay, wrect.topleft)

    def _draw_grid_lines(self, surf, wrect, c):
        """
        Draw grid overlay lines.

        Math note:
        - We compute offsets so grid lines align with world cell boundaries even
          when camera has non-integer offsets.
        """
        ax, ay = self.cam.world_to_screen(0, 0)

        # We want the first grid line to start at the next multiple of `c`
        # relative to the world rect, hence modulo arithmetic.
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
        """
        Draw a marker around marked agents (toggled by 'M').

        Implementation note:
        - We locate marked slot ids in the id grid (id_np) using numpy.
        - This is O(H*W) per agent if done naively; here we use argwhere which is
          also potentially heavy, but the marked list is limited (<= 10).
        """
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
        """
        Draw line-of-sight rays from selected agent.

        Algorithm:
        - Cast N rays in evenly spaced angles around 2π radians.
        - For each ray:
            - Step forward cell-by-cell up to vision_range.
            - Stop at boundary, wall, or agent.
            - Choose color based on what it hits.
        - Draw a line from agent center to hit point.

        Complexity:
        - O(num_rays * vision_range)
        Typical values are small (e.g., 32 rays, range maybe ~8-20).
        """
        slot_id = self.viewer.selected_slot_id
        if slot_id is None or slot_id not in state_data["agent_map"]:
            return

        agent_x, agent_y, _unit, my_team, _uid, _btype = state_data["agent_map"][slot_id]
        agent_x = float(agent_x)
        agent_y = float(agent_y)

        # Start position in screen pixels (center of agent cell)
        start_pos_screen = (
            wrect.x + self.cam.world_to_screen(int(agent_x), int(agent_y))[0] + c // 2,
            wrect.y + self.cam.world_to_screen(int(agent_x), int(agent_y))[1] + c // 2,
        )

        vision_range = int(float(self.registry.agent_data[slot_id, COL_VISION].item()))

        occ_grid = state_data["occ_np"]  # occupancy values
        H, W = occ_grid.shape

        num_rays_to_draw = 32
        for i in range(num_rays_to_draw):
            angle = i * (2 * math.pi / num_rays_to_draw)
            dx, dy = math.cos(angle), math.sin(angle)

            end_x, end_y = agent_x, agent_y
            color = RAY_COLORS["empty"]

            for step in range(1, vision_range + 1):
                # Ray marching: test at each integer step
                test_x = int(round(agent_x + dx * step))
                test_y = int(round(agent_y + dy * step))

                # Stop if outside map
                if not (0 <= test_x < W and 0 <= test_y < H):
                    end_x, end_y = test_x, test_y
                    break

                occupant = occ_grid[test_y, test_x]
                if occupant != 0:
                    end_x, end_y = test_x, test_y
                    if occupant == 1:
                        # Hit wall
                        color = RAY_COLORS["wall"]
                    else:
                        # Hit an agent/team-occupied cell
                        hit_team = occupant
                        color = RAY_COLORS["enemy"] if my_team != hit_team else RAY_COLORS["ally"]
                    break
            else:
                # No obstacle found within vision range: extend to max
                end_x = agent_x + dx * vision_range
                end_y = agent_y + dy * vision_range

            end_pos_world = self.cam.world_to_screen(int(end_x), int(end_y))
            end_pos_screen = (
                wrect.x + end_pos_world[0] + c // 2,
                wrect.y + end_pos_world[1] + c // 2,
            )
            pygame.draw.line(surf, color, start_pos_screen, end_pos_screen, 1)


# HUD Panel (bottom)
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
        self._draw_catastrophe_status(surf, hud)
        self._draw_scheduler_button(surf, hud)
        self._draw_score_graph(surf, hud)

        # Minimap is a separate component
        self.viewer.minimap.draw(surf, hud, state_data)

    def scheduler_button_rect(self, hud_rect: Optional[pygame.Rect] = None) -> pygame.Rect:
        hud = self.viewer.layout.hud_rect() if hud_rect is None else hud_rect
        return pygame.Rect(hud.right - 680, hud.y + 10, 120, 28)

    def _draw_scheduler_button(self, surf, hud):
        rect = self.scheduler_button_rect(hud)
        enabled = bool(self.viewer.get_catastrophe_status()["scheduler_enabled"])
        fill = COLORS["button_on"] if enabled else COLORS["button_off"]
        pygame.draw.rect(surf, fill, rect, border_radius=6)
        pygame.draw.rect(surf, COLORS["button_border"], rect, 2, border_radius=6)
        label = "Scheduler: ON" if enabled else "Scheduler: OFF"
        text = self.viewer.text_cache.render(label, 13, COLORS["bg"])
        surf.blit(text, text.get_rect(center=rect.center))

    def _draw_catastrophe_status(self, surf, hud):
        status = self.viewer.get_catastrophe_status()
        x = 12
        y = hud.y + 72

        scheduler_text = f"Catastrophe: {'ON' if status['scheduler_enabled'] else 'OFF'} ({status['scheduler_mode']})"
        surf.blit(self.viewer.text_cache.render(scheduler_text, 14, COLORS["text"]), (x, y))
        y += 18

        if status["active_event"] is None:
            event_text = f"Event: none | last: {status['last_skip_reason']}"
        else:
            pattern = status["event_pattern_name"] or "unknown"
            remaining = status["event_remaining_ticks"]
            expiry = status["event_expires_at_tick"]
            if remaining is None or expiry is None:
                tail = "persistent"
            else:
                tail = f"remain {remaining} (t{expiry})"
            event_text = f"Event: {pattern} | {tail}"
        surf.blit(self.viewer.text_cache.render(event_text, 14, COLORS["text_dim"]), (x, y))
        y += 18

        zone_text = (
            f"Heal A/D: {status['active_zone_count']}/{status['dormant_zone_count']}"
            f" | Manual: {status['manual_override_count']}"
        )
        surf.blit(self.viewer.text_cache.render(zone_text, 14, COLORS["text_dim"]), (x, y))

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


# Side Panel (right)
class SidePanel:
    """
    Right panel:

    - Shows selected agent details (pos, HP, atk, vision, brain info, param count).
    - Shows legend and keyboard controls.
    """

    def __init__(self, viewer, registry):
        self.viewer = viewer
        self.registry = registry

    def _draw_legend(self, surf, x, y):
        """Draw the legend and controls list."""
        y += 20
        surf.blit(self.viewer.text_cache.render("Legend & Controls", 18, COLORS["text"]), (x, y))
        y += 30

        legend_items = [
            ("Red Soldier", COLORS["red_soldier"]),
            ("Red Archer", COLORS["red_archer"]),
            ("Blue Soldier", COLORS["blue_soldier"]),
            ("Blue Archer", COLORS["blue_archer"]),
            ("Heal Zone Active", COLORS["heal_active"]),
            ("Heal Zone Dormant", COLORS["heal_dormant"]),
        ]

        for label, color in legend_items:
            pygame.draw.rect(surf, color, (x, y, 15, 15))
            surf.blit(self.viewer.text_cache.render(label, 13, COLORS["text_dim"]), (x + 25, y))
            y += 22

        y += 20

        controls = [
            "WASD / Arrows: Pan Camera",
            "Mouse Wheel: Zoom",
            "Left Click Agent: Inspect",
            "Left Click Heal Zone: Select Zone",
            "Right Click / Shift+Click Zone: Toggle Zone",
            "Y: Scheduler ON/OFF",
            "0: Restore All Zones",
            "1/2/3/4/5: Small/Med/Left/Right/Cluster",
            "SPACE: Pause Simulation",
            "+/-: Change Speed",
            "R/T/B/N: Overlay Toggles",
            "S: Save Selected Brain",
            "F9: Manual Checkpoint Save",
            "F11: Toggle Fullscreen",
        ]

        for line in controls:
            surf.blit(self.viewer.text_cache.render(line, 13, COLORS["text_dim"]), (x, y))
            y += 18

        return y

    def _draw_catastrophe_panel(self, surf, x, y):
        status = self.viewer.get_catastrophe_status()
        zone_status = self.viewer.get_selected_zone_status()

        y += 22
        surf.blit(self.viewer.text_cache.render("Catastrophe Control", 18, COLORS["text"]), (x, y))
        y += 28

        lines = [
            f"Scheduler: {'ON' if status['scheduler_enabled'] else 'OFF'} ({status['scheduler_mode']})",
            f"Active Event: {status['event_pattern_name'] or 'none'}",
        ]
        if status['active_event'] is not None:
            if status['event_remaining_ticks'] is None:
                lines.append("Remaining: persistent")
            else:
                lines.append(
                    f"Remaining: {status['event_remaining_ticks']} ticks (t{status['event_expires_at_tick']})"
                )
        else:
            lines.append(f"Last State: {status['last_skip_reason']}")

        lines.append(
            f"Heal Zones: active {status['active_zone_count']} | dormant {status['dormant_zone_count']}"
        )
        lines.append(f"Manual Overrides: {status['manual_override_count']}")

        if zone_status is None:
            lines.append("Selected Zone: none")
        else:
            lines.append(
                f"Selected Zone: {zone_status['zone_id']} [{zone_status['effective_state_label']}]"
            )
            lines.append(
                f"Manual Override: {zone_status['manual_override_label']} | overlaps here: {zone_status['overlap_count']}"
            )
            lines.append(f"Bounds: {zone_status['bounds_label']}")

        if self.viewer._last_zone_status_message:
            lines.append(f"Last Action: {self.viewer._last_zone_status_message}")

        for line in lines:
            surf.blit(self.viewer.text_cache.render(line, 13, COLORS["text_dim"]), (x, y))
            y += 18
        return y

    def draw(self, surf, state_data):
        srect = self.viewer.layout.side_rect()

        # Background and border
        surf.fill(COLORS["side_bg"], srect)
        pygame.draw.rect(surf, COLORS["border"], srect, 2)

        pad, y = 12, srect.y + 12
        x = srect.x + pad

        surf.blit(self.viewer.text_cache.render("Agent Inspector", 18, COLORS["text"]), (x, y))
        y += 30

        slot_id = self.viewer.selected_slot_id

        # CASE 1: No selection
        if slot_id is None:
            surf.blit(self.viewer.text_cache.render("Click an agent to inspect.", 13, COLORS["text_dim"]), (x, y))
            y += 30

        # CASE 2: Selected slot but agent died (not in current agent_map)
        elif slot_id not in state_data["agent_map"]:
            uid_str = (
                f"ID: {self.viewer.last_selected_uid} (Dead)"
                if self.viewer.last_selected_uid is not None
                else f"Slot: {slot_id} (Dead)"
            )
            surf.blit(self.viewer.text_cache.render(uid_str, 13, COLORS["warn"]), (x, y))
            y += 30

        # CASE 3: Selected and alive
        else:
            _ax, _ay, _u, _t, unique_id, _btype = state_data["agent_map"][slot_id]
            surf.blit(self.viewer.text_cache.render(f"ID: {int(unique_id)}", 16, COLORS["green"]), (x, y))
            y += 24

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

            # Per-agent accumulated score (tracked via PPO hook)
            agent_score = self.viewer.agent_scores.get(int(unique_id), 0.0)

            surf.blit(self.viewer.text_cache.render(f"Score: {agent_score:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Pos: ({pos[0]}, {pos[1]})", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Unit: {unit_name}", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Brain: {brain_name}", 13, COLORS["text_dim"]), (x, y)); y += 18

            # HP bar
            bar_w = srect.width - 2 * pad
            pygame.draw.rect(surf, COLORS["bar_bg"], (x, y, bar_w, 10))
            pygame.draw.rect(surf, COLORS["bar_fg"], (x, y, bar_w * hp_ratio, 10))
            y += 14

            surf.blit(self.viewer.text_cache.render(f"HP: {hp:.2f} / {hp_max:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 20
            surf.blit(self.viewer.text_cache.render(f"Attack: {atk:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Vision: {vision} cells", 13, COLORS["text_dim"]), (x, y)); y += 22

            # Model summary and param count if brain exists
            if brain:
                surf.blit(self.viewer.text_cache.render(f"Model: {_get_model_summary(brain)}", 13, COLORS["text_dim"]), (x, y)); y += 18
                surf.blit(self.viewer.text_cache.render(f"Params: {_param_count(brain):,}", 13, COLORS["text_dim"]), (x, y)); y += 18

        y = self._draw_catastrophe_panel(surf, x, y)

        # Separator line
        pygame.draw.line(surf, COLORS["border"], (srect.x, y + 10), (srect.right, y + 10), 2)

        # Legend and controls
        self._draw_legend(surf, x, y)


# Minimap
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


# Input handling
class InputHandler:
    """
    Central event handler for the Viewer.

    Responsibilities:
    - Continuous key state handling (WASD/arrows for panning).
    - Discrete events (keypress toggles, mouse clicks, resizing, quit).
    """

    def __init__(self, viewer):
        self.viewer = viewer

    def handle(self):
        """
        Process input events for one frame.

        Returns:
            (running, advance_tick)
            - running=False means the viewer should exit
            - advance_tick=True means if paused, we should advance one simulation tick
        """
        running, advance_tick = True, False

        # Continuous keyboard state for smooth camera movement
        keys = pygame.key.get_pressed()

        # Pan speed inversely proportional to zoom (cell size).
        # If cell is large (zoomed in), we pan less per frame for precision.
        # If cell is small (zoomed out), we pan more per frame for speed.
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
                # Only allow resizable mode when not fullscreen
                if not self.viewer.fullscreen:
                    self.viewer.Wpix, self.viewer.Hpix = max(800, ev.w), max(520, ev.h)
                    self.viewer.screen = pygame.display.set_mode(
                        (self.viewer.Wpix, self.viewer.Hpix),
                        pygame.RESIZABLE
                    )
                    self.viewer.world_renderer.invalidate_static_background()

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False

                elif ev.key == pygame.K_SPACE:
                    # Toggle pause
                    self.viewer.paused = not self.viewer.paused

                elif ev.key == pygame.K_PERIOD and self.viewer.paused:
                    # Single-step when paused
                    advance_tick = True

                elif ev.key == pygame.K_m and self.viewer.selected_slot_id is not None:
                    # Mark/unmark current selection (max 10 markers)
                    if self.viewer.selected_slot_id in self.viewer.marked:
                        self.viewer.marked.remove(self.viewer.selected_slot_id)
                    elif len(self.viewer.marked) < 10:
                        self.viewer.marked.append(self.viewer.selected_slot_id)

                # Toggle overlays
                elif ev.key == pygame.K_r:
                    self.viewer.show_rays = not self.viewer.show_rays
                elif ev.key == pygame.K_t:
                    self.viewer.threat_vision_mode = not self.viewer.threat_vision_mode
                elif ev.key == pygame.K_b:
                    self.viewer.battle_view_enabled = not self.viewer.battle_view_enabled
                elif ev.key == pygame.K_n:
                    self.viewer.show_brain_types = not self.viewer.show_brain_types

                # Catastrophe viewer controls
                elif ev.key == pygame.K_y:
                    self.viewer.toggle_catastrophe_scheduler()
                elif ev.key == pygame.K_0:
                    self.viewer.restore_all_zones_to_normal_effective_state()
                elif ev.key == pygame.K_1:
                    self.viewer.manual_trigger_catastrophe("random_small")
                elif ev.key == pygame.K_2:
                    self.viewer.manual_trigger_catastrophe("random_medium")
                elif ev.key == pygame.K_3:
                    self.viewer.manual_trigger_catastrophe("left_side")
                elif ev.key == pygame.K_4:
                    self.viewer.manual_trigger_catastrophe("right_side")
                elif ev.key == pygame.K_5:
                    self.viewer.manual_trigger_catastrophe("cluster_survives")

                # Save selected agent brain weights
                elif ev.key == pygame.K_s:
                    self.viewer.save_selected_brain()

                # Manual checkpoint save request (actual save happens in Viewer.run loop)
                elif ev.key == pygame.K_F9:
                    self.viewer.save_requested = True

                # Fullscreen toggle
                elif ev.key == pygame.K_F11:
                    self.viewer.fullscreen = not self.viewer.fullscreen
                    if self.viewer.fullscreen:
                        self.viewer.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        self.viewer.screen = pygame.display.set_mode(
                            (self.viewer.Wpix, self.viewer.Hpix),
                            pygame.RESIZABLE
                        )
                    self.viewer.world_renderer.invalidate_static_background()

                # Speed multiplier adjustments
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.viewer.speed_multiplier = min(16, self.viewer.speed_multiplier * 2)
                elif ev.key == pygame.K_MINUS:
                    self.viewer.speed_multiplier = max(0.25, self.viewer.speed_multiplier / 2)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    button_rect = self.viewer.hud_panel.scheduler_button_rect()
                    if button_rect.collidepoint(ev.pos):
                        self.viewer.toggle_catastrophe_scheduler()
                        continue

                if ev.button in (1, 3) and wrect.collidepoint(ev.pos):
                    gx, gy = self.viewer.cam.screen_to_world(
                        ev.pos[0] - wrect.x,
                        ev.pos[1] - wrect.y,
                    )
                    if not self.viewer.is_valid_world_cell(gx, gy):
                        continue

                    shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    if ev.button == 3 or (ev.button == 1 and shift_held):
                        self.viewer.manual_toggle_zone_at_cell(gx, gy)
                        continue

                    slot_id = self.viewer.fast_grid_pick_slot(gx, gy)
                    if slot_id is not None and slot_id >= 0:
                        self.viewer.select_agent_slot(slot_id)
                    elif self.viewer.select_heal_zone_at_cell(gx, gy):
                        self.viewer.selected_slot_id = None
                    else:
                        self.viewer.selected_slot_id = None
                        self.viewer.clear_selected_zone()

                # Mouse wheel zoom
                elif ev.button == 4:
                    self.viewer.cam.zoom_at(1.12)
                    self.viewer.world_renderer.invalidate_static_background()
                elif ev.button == 5:
                    self.viewer.cam.zoom_at(1 / 1.12)
                    self.viewer.world_renderer.invalidate_static_background()

        return running, advance_tick


# Animation manager (placeholder)
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


# Main Viewer class
class Viewer:
    """
    Main application window: runs the Pygame loop and coordinates everything.

    Key features:
    - Efficient GPU->CPU sampling (state refresh interval).
    - Camera panning/zooming.
    - Selection + inspector panel.
    - Optional visual overlays (HP bars, rays, threat vision, etc.).
    - Optional checkpointing with atomic saves.

    Checkpointing concept:
    - "Atomic save" typically means:
        write checkpoint to temp directory -> rename/move to final name
      so partial writes don't corrupt the main checkpoint.
    """

    def __init__(self, grid: torch.Tensor, cell_size: Optional[int] = None, show_grid: bool = True):
        _center_window()

        # Pygame init sets up SDL + subsystems
        pygame.init()
        pygame.display.set_caption("Neural Abyss")

        self.grid = grid
        self.margin = 8
        self.show_grid = show_grid

        # Camera uses cell size in pixels plus world dimensions (W,H in cells)
        self.cam = Camera(int(cell_size or config.CELL_SIZE), grid.shape[2], grid.shape[1])

        H, W = grid.shape[1], grid.shape[2]
        side_min_w, hud_h = 280, 126

        # Determine initial window size while respecting display constraints
        try:
            display_info = pygame.display.Info()
            max_w = display_info.current_w - 80
            max_h = display_info.current_h - 120
        except pygame.error:
            # Safe fallback when display info is unavailable
            max_w, max_h = 1280, 720

        world_px_w, world_px_h = W * self.cam.cell_px, H * self.cam.cell_px
        init_w = min(max_w, max(1280, world_px_w + side_min_w + 3 * self.margin))
        init_h = min(max_h, max(720, world_px_h + hud_h + 2 * self.margin))

        self.Wpix, self.Hpix = int(init_w), int(init_h)

        # Resizable window mode by default
        self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)

        self.text_cache = TextCache()
        self.clock = pygame.time.Clock()

        # Selection state
        self.selected_slot_id: Optional[int] = None
        self.last_selected_uid: Optional[int] = None
        self.marked: List[int] = []
        self.selected_heal_zone_id: Optional[str] = None
        self.selected_heal_zone_overlap_count: int = 0
        self._last_zone_status_message: str = ""

        # Runtime bindings populated when Viewer.run(...) starts.
        self.engine = None
        self.registry = None
        self.stats = None

        # Display toggles
        self.show_rays = False
        self.paused = False
        self.threat_vision_mode = False
        self.battle_view_enabled = False
        self.show_brain_types = False
        self.fullscreen = False
        self.speed_multiplier = 1.0

        # Per-agent cumulative scores from PPO rewards.
        self.agent_scores: Dict[int, float] = collections.defaultdict(float)

        # Checkpointing
        self.save_requested: bool = False
        self._ckpt_last_status: str = ""

        # Performance knobs
        self.STATE_REFRESH_EVERY_FRAMES = int(getattr(config, "VIEWER_STATE_REFRESH_EVERY", 2))
        self.PICK_REFRESH_EVERY_FRAMES = int(getattr(config, "VIEWER_PICK_REFRESH_EVERY", 2))

        # Cached CPU data to avoid GPU sync each frame
        self._cached_state_data = None
        self._cached_id_np = None
        self._cached_occ_np = None
        self._cached_alive_indices: List[int] = []
        self._cached_agent_map: Dict[int, tuple] = {}

        self._last_state_refresh_frame = -10
        self._last_pick_refresh_frame = -10

    def save_selected_brain(self):
        """
        Save the PyTorch `state_dict` of the selected agent's brain to disk.

        Notes:
        - `state_dict()` is the standard PyTorch method returning model weights.
        - Saved as .pth file.
        - If no selection or registry not set, returns safely.
        """
        if self.selected_slot_id is None or not hasattr(self, "registry"):
            return

        brain = self.registry.brains[self.selected_slot_id]
        if brain:
            tick = self.stats.tick if hasattr(self, "stats") else 0
            uid = self.last_selected_uid if self.last_selected_uid is not None else self.selected_slot_id
            filename = f"brain_agent_{uid}_t_{tick}.pth"
            try:
                torch.save(brain.state_dict(), filename)
                kind = brain_kind_from_module(brain)
                label = brain_kind_display_name(kind) if kind else brain.__class__.__name__
                print(f"[Viewer] Saved {label} brain for agent {uid} to '{filename}'")
            except Exception as e:
                print(f"[Viewer] Error saving brain: {e}")

    def current_tick(self) -> int:
        return int(getattr(self.stats, "tick", 0)) if self.stats is not None else 0

    def is_valid_world_cell(self, gx: int, gy: int) -> bool:
        return 0 <= int(gx) < int(self.grid.shape[2]) and 0 <= int(gy) < int(self.grid.shape[1])

    def get_catastrophe_controller(self):
        return None if self.engine is None else getattr(self.engine, "catastrophe_controller", None)

    def get_catastrophe_status(self) -> Dict[str, Any]:
        controller = self.get_catastrophe_controller()
        if controller is None:
            return {
                "scheduler_enabled": False,
                "scheduler_mode": "periodic",
                "active_event": None,
                "event_pattern_name": None,
                "event_source": None,
                "event_expires_at_tick": None,
                "event_remaining_ticks": None,
                "zone_count": 0,
                "active_zone_count": 0,
                "dormant_zone_count": 0,
                "active_zone_ids": tuple(),
                "manual_override_count": 0,
                "catastrophe_slot_active": False,
                "last_skip_reason": "controller_unavailable",
            }
        return controller.ui_status_snapshot(tick_now=self.current_tick())

    def request_zone_render_refresh(self):
        if self.engine is not None and hasattr(self.engine, "refresh_zone_runtime_cache"):
            self.engine.refresh_zone_runtime_cache()
        if hasattr(self, "world_renderer") and self.world_renderer is not None:
            if self.engine is not None:
                self.world_renderer.maybe_refresh_zone_cache(self.engine, force=True)
            else:
                self.world_renderer.invalidate_static_background()

    def _set_zone_status_message(self, message: str) -> None:
        self._last_zone_status_message = str(message)

    def select_agent_slot(self, slot_id: Optional[int]) -> None:
        if slot_id is None or slot_id < 0:
            self.selected_slot_id = None
            return
        self.clear_selected_zone()
        self.selected_slot_id = int(slot_id)
        if self.registry is None:
            return
        if hasattr(self.registry, "agent_uids"):
            self.last_selected_uid = int(self.registry.agent_uids[self.selected_slot_id].item())
        else:
            self.last_selected_uid = int(self.registry.agent_data[self.selected_slot_id, COL_AGENT_ID].item())

    def clear_selected_zone(self) -> None:
        self.selected_heal_zone_id = None
        self.selected_heal_zone_overlap_count = 0

    def _zone_at_cell(self, gx: int, gy: int):
        if self.engine is None or not self.is_valid_world_cell(gx, gy):
            return None, 0
        zones = getattr(self.engine, "zones", None)
        if zones is None:
            return None, 0
        matches = zones.get_heal_zones_containing_cell(int(gx), int(gy), active_only=False)
        if not matches:
            return None, 0
        return matches[0], len(matches)

    def select_heal_zone_at_cell(self, gx: int, gy: int) -> bool:
        zone, overlap_count = self._zone_at_cell(gx, gy)
        if zone is None:
            return False
        self.selected_heal_zone_id = str(zone.zone_id)
        self.selected_heal_zone_overlap_count = int(overlap_count)
        self._set_zone_status_message(f"Selected heal zone {zone.zone_id}")
        return True

    def get_selected_zone_status(self) -> Optional[Dict[str, Any]]:
        if self.engine is None or self.selected_heal_zone_id is None:
            return None
        zones = getattr(self.engine, "zones", None)
        if zones is None:
            return None
        try:
            zone = zones.get_heal_zone(self.selected_heal_zone_id)
        except Exception:
            return None

        manual_value = dict(getattr(zones, "manual_zone_enabled", {})).get(str(zone.zone_id))
        if manual_value is None:
            manual_label = "none"
        else:
            manual_label = "forced ON" if bool(manual_value) else "forced OFF"

        active_now = str(zone.zone_id) in set(getattr(zones, "active_heal_zone_ids", tuple()))
        y0, y1, x0, x1 = zone.bounds
        return {
            "zone_id": str(zone.zone_id),
            "effective_state_label": "ACTIVE" if active_now else "DORMANT",
            "manual_override_label": manual_label,
            "overlap_count": int(max(1, self.selected_heal_zone_overlap_count)),
            "bounds_label": f"x=[{x0},{x1}) y=[{y0},{y1})",
            "status_message": str(self._last_zone_status_message),
        }

    def manual_toggle_zone_at_cell(self, gx: int, gy: int) -> bool:
        if not self.select_heal_zone_at_cell(gx, gy):
            self.clear_selected_zone()
            self._set_zone_status_message("No heal zone under cursor")
            return False

        self.selected_slot_id = None
        controller = self.get_catastrophe_controller()
        if controller is None:
            self._set_zone_status_message("Catastrophe controller unavailable")
            return False

        zone_id = str(self.selected_heal_zone_id)
        changed = bool(controller.manual_toggle_zone(zone_id=zone_id, tick_now=self.current_tick()))
        if changed:
            self.request_zone_render_refresh()
            zone_status = self.get_selected_zone_status()
            if zone_status is not None:
                self._set_zone_status_message(
                    f"Zone {zone_id} -> {zone_status['effective_state_label']} (manual override applied)"
                )
        return changed

    def toggle_catastrophe_scheduler(self) -> None:
        controller = self.get_catastrophe_controller()
        if controller is None:
            self._set_zone_status_message("Catastrophe controller unavailable")
            return
        enabled = not bool(controller.scheduler_enabled)
        controller.set_scheduler_enabled(enabled)
        self._set_zone_status_message(f"Scheduler {'enabled' if enabled else 'disabled'}")

    def restore_all_zones_to_normal_effective_state(self) -> bool:
        controller = self.get_catastrophe_controller()
        if controller is None:
            self._set_zone_status_message("Catastrophe controller unavailable")
            return False
        changed = bool(
            controller.restore_all_zones_to_normal_effective_state(
                tick_now=self.current_tick(),
                reason="viewer_restore_all",
            )
        )
        if changed:
            self.request_zone_render_refresh()
        self._set_zone_status_message("Restored all heal zones to normal effective state")
        return changed

    def manual_trigger_catastrophe(self, pattern_key: str) -> bool:
        controller = self.get_catastrophe_controller()
        if controller is None:
            self._set_zone_status_message("Catastrophe controller unavailable")
            return False
        changed = bool(
            controller.trigger_manual_pattern(
                pattern_key=str(pattern_key),
                tick_now=self.current_tick(),
                source="viewer_manual",
            )
        )
        if changed:
            self.request_zone_render_refresh()
            status = self.get_catastrophe_status()
            self._set_zone_status_message(
                f"Manual catastrophe triggered: {status['event_pattern_name'] or pattern_key}"
            )
        else:
            self._set_zone_status_message(f"Manual catastrophe rejected: {pattern_key}")
        return changed

    # Fast click picking (no GPU sync in the common case)
    def fast_grid_pick_slot(self, gx: int, gy: int) -> Optional[int]:
        """
        Return the agent slot ID at grid cell (gx, gy) using cached id grid.

        Why:
        - Reading from GPU tensor per click would force sync and can stutter.
        - Cached numpy `id_np` avoids GPU reads during user input.

        Fallback:
        - If cache doesn't exist, do a one-off slow GPU read.
        """
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

    # State refresh (bulk GPU->CPU copy)
    def _refresh_state_cpu(self):
        """
        Copy the minimal data needed for rendering/picking from GPU to CPU.

        This is the *core* performance optimization:
        - Instead of reading many scattered values from GPU each frame,
          we do a single bulk copy of a small set of tensors.

        Data copied:
        - grid occupancy channel (channel 0)
        - grid agent-id channel (channel 2)
        - alive agent list + columns (x,y,unit,team,uid)
        - brain type label from registry.brains (Python objects, no GPU sync)
        """
        with torch.no_grad():
            # Copy full grid to CPU once; then slice channels
            grid_cpu = self.grid.detach().cpu()

            # Convert to numpy views (copy=False encourages view when possible)
            occ_np = grid_cpu[0].numpy().astype(np.int16, copy=False)
            id_np = grid_cpu[2].numpy().astype(np.int32, copy=False)

            ad = self.registry.agent_data

            # Alive mask from tensor column
            alive_mask = ad[:, COL_ALIVE] > 0.5
            alive_idx_t = torch.nonzero(alive_mask).squeeze(1)

            if alive_idx_t.numel() == 0:
                alive_indices: List[int] = []
                agent_map: Dict[int, tuple] = {}
            else:
                alive_indices = alive_idx_t.cpu().tolist()

                # Bulk select needed columns for alive agents
                cols = [COL_X, COL_Y, COL_UNIT, COL_TEAM, COL_AGENT_ID]
                alive_data = ad.index_select(0, alive_idx_t)[:, cols].detach().cpu().numpy()
                uid_data = None
                if hasattr(self.registry, "agent_uids"):
                    uid_data = (
                        self.registry.agent_uids
                        .index_select(0, alive_idx_t)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                brains = self.registry.brains

                agent_map = {}
                for k, slot_id in enumerate(alive_indices):
                    x, y, unit, team, uid = alive_data[k]
                    if uid_data is not None:
                        uid = uid_data[k]

                    # Determine short brain label (string) for rendering
                    br = brains[slot_id]
                    if br is None:
                        btype = "?"
                    else:
                        kind = brain_kind_from_module(br)
                        btype = brain_kind_short_label(kind) if kind else "?"

                    agent_map[slot_id] = (
                        float(x), float(y), float(unit), float(team), float(uid), btype
                    )

            # Save into Viewer fields (used by renderer and picking)
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

    # PPO score hook
    def _install_score_hook(self, engine, registry):
        """
        Wrap the PPO runtime's ``record_step`` method to accumulate UI scores.

        The wrapper preserves the original call and adds a read-only score view
        keyed by persistent agent id.
        """
        if not hasattr(engine, "_ppo") or engine._ppo is None:
            return

        original_record_step = engine._ppo.record_step

        def record_step_with_score_tracking(*args, **kwargs):
            agent_ids = kwargs.get("agent_ids")
            rewards = kwargs.get("rewards")

            if agent_ids is not None and rewards is not None and agent_ids.numel() > 0:
                with torch.no_grad():
                    slot_ids = agent_ids.detach()

                    # Bulk lookup unique ids (GPU->CPU copy). Prefer int64 side-tensor.
                    if hasattr(registry, "agent_uids"):
                        uids = registry.agent_uids.index_select(0, slot_ids).detach().cpu().numpy()
                    else:
                        uids = registry.agent_data.index_select(0, slot_ids)[:, COL_AGENT_ID].detach().cpu().numpy()
                    r = rewards.detach().cpu().numpy()

                for uid, rv in zip(uids, r):
                    self.agent_scores[int(uid)] += float(rv)

            return original_record_step(*args, **kwargs)

        engine._ppo.record_step = record_step_with_score_tracking

    # Main run loop
    def run(
        self,
        engine,
        registry,
        stats,
        tick_limit: int = 0,
        target_fps: Optional[int] = None,
        run_dir: Optional[str] = None,
    ):
        """
        Start the viewer loop.

        Args:
            engine: simulation engine (must expose run_tick()).
            registry: agent registry (agent_data tensor, brains list).
            stats: stats object (tick, red/blue score etc.).
            tick_limit: stop after this many ticks (0 = unlimited).
            target_fps: render FPS cap. If None uses config.TARGET_FPS.
            run_dir: directory for checkpoints. If None, checkpointing disabled.
        """
        self.engine = engine
        self.registry = registry
        self.stats = stats

        # Lazy import checkpointing only if run_dir provided
        ckpt_mgr = None
        if run_dir is not None:
            from utils.checkpointing import CheckpointManager
            ckpt_mgr = CheckpointManager(Path(run_dir))

        # Initialize components that depend on Viewer existing
        self.layout = LayoutManager(self)
        self.world_renderer = WorldRenderer(self, self.grid, registry)
        self.hud_panel = HudPanel(self, stats)
        self.side_panel = SidePanel(self, registry)
        self.input_handler = InputHandler(self)
        self.anim_manager = AnimationManager()
        self.minimap = Minimap(self)

        # Build cached zone info for background
        self.world_renderer.build_static_cache(engine)

        # Install PPO hook if PPO exists
        self._install_score_hook(engine, registry)

        running = True
        frame_count = 0

        # Prime CPU caches once before loop
        self._refresh_state_cpu()
        self._last_state_refresh_frame = frame_count
        self._last_pick_refresh_frame = frame_count

        while running:
            # 1) Input phase (selection, toggles, camera)
            running, advance_tick = self.input_handler.handle()

            # 2) Manual checkpoint save (safe between ticks)
            if self.save_requested:
                self.save_requested = False

                if ckpt_mgr is None:
                    self._ckpt_last_status = "Checkpoint save requested but run_dir is not set."
                    print("[checkpoint]", self._ckpt_last_status)
                else:
                    try:
                        # Capture viewer state for restore (camera, pause, speed, scores)
                        viewer_state = {
                            "paused": bool(self.paused),
                            "speed_mult": float(self.speed_multiplier),  # attribute name mismatch noted
                            "camera": {
                                "offset_x": float(getattr(self.cam, "offset_x", 0.0)),
                                "offset_y": float(getattr(self.cam, "offset_y", 0.0)),
                                "zoom": float(getattr(self.cam, "zoom", 1.0)),
                            },
                            "agent_scores": dict(self.agent_scores),
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

                        # Prune old checkpoints to keep last N
                        ckpt_mgr.prune_keep_last_n(int(getattr(config, "CHECKPOINT_KEEP_LAST_N", 1)))

                        self._ckpt_last_status = f"Checkpoint saved: {out_dir.name}"
                        print("[checkpoint]", self._ckpt_last_status)
                    except Exception as ex:
                        self._ckpt_last_status = f"Checkpoint FAILED: {type(ex).__name__}: {ex}"
                        print("[checkpoint]", self._ckpt_last_status)

            # 3) Decide how many simulation ticks to run this frame
            num_ticks_this_frame = 0

            if not self.paused:
                if self.speed_multiplier >= 1:
                    # Run multiple ticks per frame (fast-forward)
                    num_ticks_this_frame = int(self.speed_multiplier)
                elif self.speed_multiplier > 0:
                    # Run less than 1 tick per frame (slow motion)
                    denom = int(1 / self.speed_multiplier)
                    if denom <= 1 or (frame_count % denom) == 0:
                        num_ticks_this_frame = 1
            elif advance_tick:
                # Single step when paused
                num_ticks_this_frame = 1

            # 4) Run simulation ticks
            for _ in range(num_ticks_this_frame):
                engine.run_tick()
                self.hud_panel.update()

                # 4a) Automatic checkpointing (trigger file + periodic)
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

            # 5) Refresh CPU state caches if needed (performance-critical)
            if (frame_count - self._last_state_refresh_frame) >= self.STATE_REFRESH_EVERY_FRAMES:
                self._refresh_state_cpu()
                self._last_state_refresh_frame = frame_count

            # Pick refresh is currently tied to full refresh, but kept separate for future tuning
            if (frame_count - self._last_pick_refresh_frame) >= self.PICK_REFRESH_EVERY_FRAMES:
                self._last_pick_refresh_frame = frame_count

            state_data = self._cached_state_data
            if state_data is None:
                # Extremely unlikely fallback
                self._refresh_state_cpu()
                state_data = self._cached_state_data

            # 6) Render phase
            self.world_renderer.maybe_refresh_zone_cache(engine)
            self.screen.fill(COLORS["bg"])
            self.world_renderer.draw(self.screen, state_data)
            self.hud_panel.draw(self.screen, state_data)
            self.side_panel.draw(self.screen, state_data)

            pygame.display.flip()

            # Cap render FPS
            self.clock.tick(int(target_fps or config.TARGET_FPS))

            frame_count += 1

            # Stop condition by tick limit
            if tick_limit > 0 and stats.tick >= tick_limit:
                running = False

        pygame.quit()
