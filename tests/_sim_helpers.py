from __future__ import annotations

from typing import Iterable, Sequence

import torch

import config
from agent.mlp_brain import create_mlp_brain
from engine.agent_registry import AgentsRegistry, TEAM_BLUE_ID, TEAM_RED_ID
from engine.grid import make_grid
from engine.mapgen import HealZone, Zones
from engine.tick import TickEngine
from simulation.stats import SimulationStats


CPU = torch.device("cpu")


def configure_cpu_runtime(
    monkeypatch,
    *,
    grid_w: int = 7,
    grid_h: int = 7,
    max_agents: int = 8,
    ppo_enabled: bool = False,
    respawn_enabled: bool = False,
    metabolism_enabled: bool = False,
) -> None:
    monkeypatch.setattr(config, "TORCH_DEVICE", CPU)
    monkeypatch.setattr(config, "DEVICE", CPU, raising=False)
    monkeypatch.setattr(config, "TORCH_DTYPE", torch.float32)
    monkeypatch.setattr(config, "GRID_WIDTH", int(grid_w))
    monkeypatch.setattr(config, "GRID_HEIGHT", int(grid_h))
    monkeypatch.setattr(config, "MAX_AGENTS", int(max_agents))
    monkeypatch.setattr(config, "PPO_ENABLED", bool(ppo_enabled))
    monkeypatch.setattr(config, "RESPAWN_ENABLED", bool(respawn_enabled))
    monkeypatch.setattr(config, "METABOLISM_ENABLED", bool(metabolism_enabled))
    monkeypatch.setattr(config, "RESP_FLOOR_PER_TEAM", 0)
    monkeypatch.setattr(config, "RESP_MAX_PER_TICK", 0)
    monkeypatch.setattr(config, "RESP_PERIOD_BUDGET", 0)
    monkeypatch.setattr(config, "RESPAWN_PROB_PER_DEAD", 0.0)
    monkeypatch.setattr(config, "RESPAWN_BATCH_PER_TEAM", 0)
    monkeypatch.setattr(config, "RESPAWN_COOLDOWN_TICKS", 0)
    monkeypatch.setattr(config, "RANDOM_WALLS", 0)
    monkeypatch.setattr(config, "HEAL_RATE", 0.1)
    monkeypatch.setattr(config, "CP_REWARD_PER_TICK", 1.0)
    monkeypatch.setattr(config, "PPO_REWARD_HEALING_RECOVERY", 0.0, raising=False)
    monkeypatch.setattr(config, "PPO_REWARD_CONTESTED_CP", 0.0)
    monkeypatch.setattr(config, "TEAM_KILL_REWARD", 1.0)
    monkeypatch.setattr(config, "TEAM_DEATH_PENALTY", -0.5)
    monkeypatch.setattr(config, "TEAM_DMG_DEALT_REWARD", 0.0)
    monkeypatch.setattr(config, "TEAM_DMG_TAKEN_PENALTY", 0.0)


def make_test_engine(
    monkeypatch,
    *,
    grid_w: int = 7,
    grid_h: int = 7,
    max_agents: int = 8,
    zones: Zones | None = None,
    ppo_enabled: bool = False,
    respawn_enabled: bool = False,
    metabolism_enabled: bool = False,
) -> tuple[TickEngine, AgentsRegistry, torch.Tensor, SimulationStats]:
    configure_cpu_runtime(
        monkeypatch,
        grid_w=grid_w,
        grid_h=grid_h,
        max_agents=max_agents,
        ppo_enabled=ppo_enabled,
        respawn_enabled=respawn_enabled,
        metabolism_enabled=metabolism_enabled,
    )
    grid = make_grid(CPU)
    registry = AgentsRegistry(grid)
    stats = SimulationStats()
    engine = TickEngine(registry, grid, stats, zones=zones)
    return engine, registry, grid, stats


def _mask_from_cells(grid_h: int, grid_w: int, cells: Iterable[tuple[int, int]]) -> torch.Tensor:
    mask = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=CPU)
    for x, y in cells:
        mask[int(y), int(x)] = True
    return mask


def make_zones(
    *,
    grid_h: int,
    grid_w: int,
    heal_cells: Sequence[Iterable[tuple[int, int]]] = (),
    cp_cells: Sequence[Iterable[tuple[int, int]]] = (),
) -> Zones:
    heal_zones = []
    for idx, cells in enumerate(heal_cells):
        mask = _mask_from_cells(grid_h, grid_w, cells)
        ys, xs = mask.nonzero(as_tuple=True)
        bounds = (
            int(ys.min().item()),
            int(ys.max().item()) + 1,
            int(xs.min().item()),
            int(xs.max().item()) + 1,
        )
        heal_zones.append(HealZone(zone_id=f"heal_{idx}", mask=mask, bounds=bounds))

    cp_masks = [_mask_from_cells(grid_h, grid_w, cells) for cells in cp_cells]
    return Zones(heal_zones=heal_zones, cp_masks=cp_masks)


def register_agent(
    registry: AgentsRegistry,
    grid: torch.Tensor,
    slot: int,
    *,
    team_is_red: bool,
    x: int,
    y: int,
    hp: float = 1.0,
    atk: float = 0.1,
    unit: int | float = 1,
    hp_max: float | None = None,
    vision_range: int = 4,
    generation: int = 1,
    brain_kind: str = "throne_of_ashen_dreams",
) -> int:
    agent_id = registry.get_next_id()
    brain = create_mlp_brain(brain_kind, int(config.OBS_DIM), int(config.NUM_ACTIONS))
    hp_cap = float(hp if hp_max is None else hp_max)

    registry.register(
        int(slot),
        agent_id=int(agent_id),
        team_is_red=bool(team_is_red),
        x=int(x),
        y=int(y),
        hp=float(hp),
        atk=float(atk),
        brain=brain,
        unit=float(unit),
        hp_max=float(hp_cap),
        vision_range=int(vision_range),
        generation=int(generation),
    )

    grid[0, int(y), int(x)] = TEAM_RED_ID if team_is_red else TEAM_BLUE_ID
    grid[1, int(y), int(x)] = float(hp)
    grid[2, int(y), int(x)] = float(slot)
    return int(agent_id)
