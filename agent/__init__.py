# war_simulation/agent/__init__.py
from .mlp_brain import (
    WhisperingAbyssBrain,
    VeilOfEchoesBrain,
    CathedralOfAshBrain,
    DreamerInBlackFogBrain,
    ObsidianPulseBrain,
    create_mlp_brain,
    brain_kind_from_module,
    brain_kind_display_name,
    brain_kind_short_label,
    describe_brain_module,
)

__all__ = [
    "WhisperingAbyssBrain",
    "VeilOfEchoesBrain",
    "CathedralOfAshBrain",
    "DreamerInBlackFogBrain",
    "ObsidianPulseBrain",
    "create_mlp_brain",
    "brain_kind_from_module",
    "brain_kind_display_name",
    "brain_kind_short_label",
    "describe_brain_module",
]
