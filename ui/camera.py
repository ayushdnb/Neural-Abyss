"""Coordinate transforms and camera state for the viewer."""

from __future__ import annotations


class Camera:
    """
    Camera system that handles:
    - Panning (moving the view across the world)
    - Zooming (scaling how large cells appear)
    - Coordinate transformation (world <-> screen)

    WHY "integer-friendly grid rendering"?
    -------------------------------------
    In grid simulations, each cell is usually drawn as a square.
    If we allow fractional pixel sizes everywhere, rendering can:
    - look blurry
    - cause jitter
    - misalign grid lines

    So we compute a final integer pixel size (cell_px) for stable visuals.
    """

    def __init__(self, cell_pixels: int, world_w: int, world_h: int):
        """
        Parameters
        ----------
        cell_pixels : int
            Base pixel size of one grid cell at zoom = 1.0.
            Example:
                cell_pixels = 8  → each grid cell is 8x8 pixels

        world_w : int
            Width of the world in grid cells (NOT pixels).

        world_h : int
            Height of the world in grid cells (NOT pixels).

        INTERNAL STATE EXPLANATION
        --------------------------
        base_cell:
            The "reference" size of a cell in pixels before zoom.

        zoom:
            A scaling multiplier.
            Final cell size = base_cell * zoom

        offset_x, offset_y:
            Camera position in WORLD SPACE (measured in CELLS, not pixels).
            This is extremely important:
                offset_x = 10.5 means camera is looking starting around cell 10.5

        Why offsets in cells and not pixels?
        ------------------------------------
        Because the simulation logic lives in grid space.
        Keeping camera in cell-space avoids unnecessary conversions.
        """
        # Ensure cell size is always at least 1 pixel (prevents invisible cells)
        self.base_cell = max(1, int(cell_pixels))

        # Zoom factor (1.0 = default scale)
        self.zoom = 1.0

        # Camera offset in WORLD CELL UNITS (not pixels)
        # These represent the top-left visible world coordinate.
        self.offset_x = 0.0  # horizontal offset (in cells)
        self.offset_y = 0.0  # vertical offset (in cells)

        # World dimensions (in grid cells)
        self.world_w = int(world_w)
        self.world_h = int(world_h)

    # DERIVED PROPERTIES (Computed Values)
    @property
    def cell_px(self) -> int:
        """
        Final pixel size of one grid cell AFTER applying zoom.

        MATHEMATICS:
        ------------
        cell_px = round(base_cell * zoom)

        Why round()?
        ------------
        - Prevents sub-pixel rendering artifacts
        - Keeps grid crisp and aligned
        - Ensures integer pixel dimensions (better for performance & visuals)

        Why max(1, ...)?
        ----------------
        Prevents cell size from becoming 0 pixels at very small zoom levels.
        A 0-pixel cell would make rendering impossible.
        """
        return max(1, int(round(self.base_cell * self.zoom)))

    # CAMERA OPERATIONS (User Interaction / Engine Control)
    def pan(self, dx_cells: float, dy_cells: float) -> None:
        """
        Move the camera across the world (panning).

        Parameters
        ----------
        dx_cells : float
            Movement in X direction (measured in CELLS).
        dy_cells : float
            Movement in Y direction (measured in CELLS).

        Example:
            pan(1, 0)  -> move camera right by 1 cell
            pan(0, -2) -> move camera up by 2 cells

        CLAMPING (CRITICAL ENGINE DETAIL)
        ---------------------------------
        We clamp the offset so the camera never goes outside the world bounds.

        Formula:
            new_offset = clamp(old_offset + delta, 0, world_size - 1)

        This prevents:
        - showing negative world coordinates
        - rendering empty/uninitialized space
        - index out-of-bounds issues in render pipelines
        """
        # Update X offset with clamping
        self.offset_x = float(
            min(
                max(self.offset_x + dx_cells, 0.0),  # lower bound = 0
                self.world_w - 1                     # upper bound = world width
            )
        )

        # Update Y offset with clamping
        self.offset_y = float(
            min(
                max(self.offset_y + dy_cells, 0.0),  # lower bound = 0
                self.world_h - 1                     # upper bound = world height
            )
        )

    def zoom_at(self, factor: float) -> None:
        """
        Zoom the camera in or out by a multiplicative factor.

        Parameters
        ----------
        factor : float
            Zoom multiplier.
            Examples:
                1.1  -> zoom in (10% larger)
                0.9  -> zoom out (10% smaller)

        MATHEMATICS:
        ------------
        zoom_new = zoom_old * factor

        CLAMP RANGE:
        ------------
        [0.25, 8.0]

        Why clamp zoom?
        ---------------
        Too small zoom (< 0.25):
            - Cells become too tiny to see
            - Precision issues

        Too large zoom (> 8.0):
            - Cells become huge
            - Performance drops (large draw calls)
            - Memory bandwidth increases in rendering
        """
        self.zoom = float(
            min(
                max(self.zoom * factor, 0.25),  # minimum zoom (25% scale)
                8.0                             # maximum zoom (800% scale)
            )
        )

    # CORE TRANSFORMATION FUNCTIONS (MOST IMPORTANT PART)
    def world_to_screen(self, x_cell: int, y_cell: int) -> tuple[int, int]:
        """
        Convert WORLD grid coordinates → SCREEN pixel coordinates.

        Parameters
        ----------
        x_cell : int
            X position in world grid (cell index).
        y_cell : int
            Y position in world grid (cell index).

        RETURNS
        -------
        (px, py) : tuple[int, int]
            Pixel coordinates on the screen where this cell should be drawn.

        CORE TRANSFORMATION EQUATION:
        -----------------------------
        screen_x = (world_x - camera_offset_x) * pixels_per_cell
        screen_y = (world_y - camera_offset_y) * pixels_per_cell

        Step-by-step intuition:
        -----------------------
        1) Subtract offset:
           This shifts the world relative to the camera position.
           If camera offset = 10, cell 10 appears at screen 0.

        2) Multiply by cell_px:
           Converts from "cells" (grid units) to "pixels" (screen units).

        Why int() cast?
        ---------------
        - Rendering APIs expect integer pixel positions
        - Prevents subpixel jitter
        - Improves visual stability in grid simulations
        """
        px = int((x_cell - self.offset_x) * self.cell_px)
        py = int((y_cell - self.offset_y) * self.cell_px)
        return px, py

    def screen_to_world(self, px: int, py: int) -> tuple[int, int]:
        """
        Convert SCREEN pixel coordinates → WORLD grid coordinates.

        This is the INVERSE transformation of world_to_screen().

        Use cases:
        ----------
        - Mouse click selection
        - Debug picking (click on agent)
        - UI interaction with the grid world

        INVERSE MATHEMATICS:
        --------------------
        world_x = offset_x + (screen_x / pixels_per_cell)
        world_y = offset_y + (screen_y / pixels_per_cell)

        Why division?
        -------------
        Because we are reversing the multiplication done in world_to_screen.

        Clamping:
        After conversion, we clamp the result to valid world bounds:
            [0, world_w - 1]
            [0, world_h - 1]

        This prevents:
        - Selecting cells outside the world
        - Index errors in simulation arrays
        """
        # Convert pixels back to cell-space using inverse scaling
        x = int(self.offset_x + px / self.cell_px)
        y = int(self.offset_y + py / self.cell_px)

        # Clamp to valid world grid boundaries
        x = max(0, min(self.world_w - 1, x))
        y = max(0, min(self.world_h - 1, y))

        return x, y
