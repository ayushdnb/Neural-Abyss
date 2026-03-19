from ui.camera import Camera


def test_set_view_clamps_zoom_and_offsets():
    cam = Camera(cell_pixels=5, world_w=100, world_h=80)
    cam.set_view(offset_x=-10.0, offset_y=500.0, zoom=99.0)

    assert cam.zoom == 8.0
    assert cam.offset_x == 0.0
    assert cam.offset_y == 79.0
