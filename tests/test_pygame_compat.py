from types import SimpleNamespace

from ui import pygame_compat


def test_resize_from_event_handles_legacy_and_window_events(monkeypatch):
    fake_pygame = SimpleNamespace(
        VIDEORESIZE=1,
        WINDOWRESIZED=2,
        WINDOWSIZECHANGED=3,
        MOUSEWHEEL=4,
        MOUSEBUTTONDOWN=5,
        display=SimpleNamespace(),
    )
    monkeypatch.setattr(pygame_compat, "pygame", fake_pygame)

    assert pygame_compat.resize_from_event(SimpleNamespace(type=1, w=900, h=700)) == (900, 700)
    assert pygame_compat.resize_from_event(SimpleNamespace(type=2, x=910, y=710)) == (910, 710)


def test_wheel_steps_prefers_mousewheel_event(monkeypatch):
    fake_pygame = SimpleNamespace(
        VIDEORESIZE=1,
        WINDOWRESIZED=2,
        WINDOWSIZECHANGED=3,
        MOUSEWHEEL=4,
        MOUSEBUTTONDOWN=5,
        display=SimpleNamespace(),
    )
    monkeypatch.setattr(pygame_compat, "pygame", fake_pygame)

    assert pygame_compat.wheel_steps_from_event(SimpleNamespace(type=4, y=2, flipped=False)) == 2
    assert pygame_compat.wheel_steps_from_event(SimpleNamespace(type=4, y=1, flipped=True)) == -1
    assert pygame_compat.wheel_steps_from_event(SimpleNamespace(type=5, button=4)) is None
