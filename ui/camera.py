from __future__ import annotations

class Camera:
    """
    Screen <-> world transform with integer-friendly grid rendering.
    """
    def __init__(self, cell_pixels: int, world_w: int, world_h: int):
        self.base_cell = max(1, int(cell_pixels))
        self.zoom = 1.0
        self.offset_x = 0.0  # in cells
        self.offset_y = 0.0  # in cells
        self.world_w = int(world_w)
        self.world_h = int(world_h)

    # --- derived ---
    @property
    def cell_px(self) -> int:
        return max(1, int(round(self.base_cell * self.zoom)))

    # --- ops ---
    def pan(self, dx_cells: float, dy_cells: float) -> None:
        self.offset_x = float(min(max(self.offset_x + dx_cells, 0.0), self.world_w - 1))
        self.offset_y = float(min(max(self.offset_y + dy_cells, 0.0), self.world_h - 1))

    def zoom_at(self, factor: float) -> None:
        self.zoom = float(min(max(self.zoom * factor, 0.25), 8.0))

    # --- transforms ---
    def world_to_screen(self, x_cell: int, y_cell: int) -> tuple[int, int]:
        px = int((x_cell - self.offset_x) * self.cell_px)
        py = int((y_cell - self.offset_y) * self.cell_px)
        return px, py

    def screen_to_world(self, px: int, py: int) -> tuple[int, int]:
        x = int(self.offset_x + px / self.cell_px)
        y = int(self.offset_y + py / self.cell_px)
        x = max(0, min(self.world_w - 1, x))
        y = max(0, min(self.world_h - 1, y))
        return x, y
