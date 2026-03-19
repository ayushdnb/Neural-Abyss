from ui.viewer_state import normalize_viewer_checkpoint_state


def test_normalize_viewer_state_supports_legacy_speed_key_and_score_keys():
    state = normalize_viewer_checkpoint_state(
        {
            "paused": "true",
            "speed_mult": "4",
            "camera": {"offset_x": "12.5", "offset_y": 7, "zoom": "2.0"},
            "agent_scores": {"12": "3.5", "bad": "x"},
            "marked": [1, "2", 1, "bad"],
        }
    )

    assert state["paused"] is True
    assert state["speed_multiplier"] == 4.0
    assert state["camera"] == {"offset_x": 12.5, "offset_y": 7.0, "zoom": 2.0}
    assert state["agent_scores"] == {12: 3.5}
    assert state["marked"] == [1, 2]
