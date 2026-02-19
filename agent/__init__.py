# war_simulation/agent/__init__.py
from .transformer_brain import TransformerBrain, scripted_transformer_brain
from .tron_brain import TronBrain
from .mirror_brain import MirrorBrain

__all__ = [
    "TransformerBrain",
    "scripted_transformer_brain",
    "TronBrain",
    "MirrorBrain",
]
