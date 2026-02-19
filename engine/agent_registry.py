from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import config

# ================================================================
# Column layout (Struct-of-Arrays for GPU efficiency)
# ================================================================
COL_ALIVE = 0       # float: 1.0 alive, 0.0 dead
COL_TEAM  = 1       # float: 2.0 red, 3.0 blue
COL_X     = 2       # float: x coordinate
COL_Y     = 3       # float: y coordinate
COL_HP    = 4       # float: current health points
COL_UNIT  = 5       # float: 1.0 = Soldier, 2.0 = Archer
COL_HP_MAX = 6      # float: maximum health points for this agent
COL_VISION = 7      # float: vision range in cells for this agent
COL_ATK    = 8      # float: attack power for this agent
COL_AGENT_ID = 9    # float: permanent unique ID for this agent
NUM_COLS = 10

# Ensure config matches this new layout if it's used elsewhere
if hasattr(config, 'AGENT_FEATURES'):
    config.AGENT_FEATURES = NUM_COLS

TEAM_RED_ID  = 2.0
TEAM_BLUE_ID = 3.0

UNIT_SOLDIER = float(getattr(config, "UNIT_SOLDIER", 1.0))
UNIT_ARCHER  = float(getattr(config, "UNIT_ARCHER", 2.0))

# ================================================================
# Buckets: allow grouping agents with same NN architecture
# ================================================================
@dataclass
class Bucket:
    signature: str
    indices: torch.Tensor          # LongTensor [K] of agent indices
    models: List[nn.Module]        # length K (same order as indices)

class AgentsRegistry:
    """
    Stores all agents in the simulation as a big tensor (SoA layout).
    Mirrors: grid occupancy channel -> registry rows.
    """

    def __init__(self, grid: torch.Tensor) -> None:
        self.grid = grid
        self.device = grid.device
        self.capacity = int(config.MAX_AGENTS)

        self._next_agent_id: int = 0

        # Main agent tensor (SoA layout)
        self.agent_data = torch.zeros(
            (self.capacity, NUM_COLS),
            dtype=config.TORCH_DTYPE,
            device=config.TORCH_DEVICE
        )
        self.agent_data[:, COL_ALIVE] = 0.0
        # Permanent unique IDs must NOT live in agent_data (which may be float16 under AMP).
        # Store them in an int64 side-tensor to avoid overflow/precision loss on long runs.
        self.agent_uids = torch.full(
            (self.capacity,),
            -1,
            dtype=torch.int64,
            device=config.TORCH_DEVICE,
        )
        # Brains are stored separately
        self.brains: List[Optional[nn.Module]] = [None] * self.capacity
        self.generations: List[int] = [0] * self.capacity

        # Add column constants as instance attributes for easy access
        self.COL_ALIVE, self.COL_TEAM, self.COL_X, self.COL_Y, self.COL_HP, self.COL_UNIT = 0, 1, 2, 3, 4, 5
        self.COL_HP_MAX, self.COL_VISION, self.COL_ATK, self.COL_AGENT_ID = 6, 7, 8, 9


    def clear(self) -> None:
        """Reset all agents (keeps capacity)."""
        self.agent_data.zero_()
        self.agent_data[:, COL_ALIVE] = 0.0
        self.agent_uids.fill_(-1)
        self.brains = [None] * self.capacity
        self.generations = [0] * self.capacity
        self._next_agent_id = 0

    def get_next_id(self) -> int:
        """Returns the next available unique agent ID and increments the counter."""
        agent_id = self._next_agent_id
        self._next_agent_id += 1
        return agent_id

    def register(
        self,
        slot: int,
        *,
        agent_id: int, 
        team_is_red: bool,
        x: int,
        y: int,
        hp: float,
        atk: float,
        brain: nn.Module,
        unit: float | int,
        hp_max: float,
        vision_range: int,
        generation: int = 0,
    ) -> None:
        """
        Put an agent into a fixed slot with all its attributes.
        """
        assert 0 <= slot < self.capacity
        d = self.agent_data
        d[slot, COL_ALIVE] = 1.0
        d[slot, COL_TEAM]  = TEAM_RED_ID if team_is_red else TEAM_BLUE_ID
        d[slot, COL_X]     = float(x)
        d[slot, COL_Y]     = float(y)
        d[slot, COL_HP]    = float(hp)
        d[slot, COL_ATK]   = float(atk)
        d[slot, COL_UNIT]  = float(unit)
        d[slot, COL_HP_MAX] = float(hp_max)
        d[slot, COL_VISION]= float(vision_range)
        # Store true UID in int64 tensor; keep float column as display-only (clamped).
        self.agent_uids[slot] = int(agent_id)
        try:
            max_f = torch.finfo(d.dtype).max
            d[slot, COL_AGENT_ID] = float(min(float(agent_id), float(max_f)))
        except Exception:
            d[slot, COL_AGENT_ID] = float(agent_id)
        

        self.brains[slot]  = brain.to(self.device)
        self.generations[slot] = int(generation)

    def kill(self, slots: torch.Tensor) -> None:
        """Mark agents dead (grid clearing happens in TickEngine)."""
        if slots is None or slots.numel() == 0:
            return
        self.agent_data[slots, COL_ALIVE] = 0.0

    def positions_xy(self, indices: torch.Tensor) -> torch.Tensor:
        """Return LongTensor [(N,2)] of XY positions for given indices."""
        x = self.agent_data[indices, COL_X].to(torch.long)
        y = self.agent_data[indices, COL_Y].to(torch.long)
        return torch.stack((x, y), dim=1)

    @staticmethod
    def _signature(model: nn.Module) -> str:
        """Create a cheap architecture fingerprint (MLP-friendly)."""
        sig_parts: List[str] = [model.__class__.__name__]
        try:
            for _, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    sig_parts.append(f"L({m.in_features},{m.out_features})")
        except Exception:
            pass
        return "|".join(sig_parts)

    def build_buckets(self, alive_idx: torch.Tensor) -> List[Bucket]:
        """Group alive agents by model signature for batched inference."""
        buckets_dict: Dict[str, List[int]] = {}
        for i in alive_idx.tolist():
            brain = self.brains[i]
            if brain is None:
                self.agent_data[i, COL_ALIVE] = 0.0
                continue
            key = self._signature(brain)
            buckets_dict.setdefault(key, []).append(i)

        out: List[Bucket] = []
        for key, lst in buckets_dict.items():
            idx = torch.tensor(lst, dtype=torch.long, device=self.device)
            models = [self.brains[j] for j in lst if j < len(self.brains) and self.brains[j] is not None]
            if models:
                out.append(Bucket(signature=key, indices=idx, models=models))
        return out