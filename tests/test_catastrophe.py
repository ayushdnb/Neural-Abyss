from __future__ import annotations

import torch

from engine.catastrophe import HealZoneCatastropheController
from engine.mapgen import Zones
from tests._sim_helpers import CPU, make_zones


def test_zones_manual_override_and_checkpoint_roundtrip_preserve_effective_mask() -> None:
    zones = make_zones(
        grid_h=6,
        grid_w=6,
        heal_cells=(
            ((1, 1), (1, 2)),
            ((4, 4),),
        ),
    )

    assert zones.heal_zone_count == 2
    assert set(zones.active_heal_zone_ids) == {"heal_0", "heal_1"}

    zones.apply_catastrophe_suppression_state(("heal_0",), event_id="cat_1")
    assert set(zones.active_heal_zone_ids) == {"heal_1"}

    zones.enable_zone_manually("heal_0")
    zones.disable_zone_manually("heal_1", clear_catastrophe=False)

    payload = zones.to_checkpoint_payload()
    restored = Zones.from_checkpoint_payload(payload, device=CPU)

    assert restored.catastrophe_state is None
    assert restored.manual_zone_enabled == {"heal_0": True, "heal_1": False}
    assert set(restored.active_heal_zone_ids) == {"heal_0"}
    assert torch.equal(restored.heal_mask, zones.heal_mask)


def test_catastrophe_controller_adopts_existing_zone_slot_when_payload_missing() -> None:
    zones = make_zones(
        grid_h=5,
        grid_w=5,
        heal_cells=(
            ((1, 1),),
            ((2, 2),),
            ((3, 3),),
        ),
    )
    zones.apply_catastrophe_suppression_state(("heal_2",), event_id="cat_7")

    controller = HealZoneCatastropheController(None)
    controller.load_checkpoint_payload(None, zones=zones, current_tick=9)
    status = controller.ui_status_snapshot(tick_now=9)

    assert status["catastrophe_slot_active"] is True
    assert status["active_event"]["event_id"] == "cat_7"
    assert status["event_source"] == "zones_sync"
    assert set(status["active_zone_ids"]) == {"heal_0", "heal_1"}
