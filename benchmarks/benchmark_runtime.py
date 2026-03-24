from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _env_flag(name: str, enabled: bool) -> None:
    os.environ[name] = "1" if enabled else "0"


def _apply_args_to_env(args: argparse.Namespace) -> None:
    os.environ["FWS_PROFILE"] = str(args.profile)
    os.environ["FWS_UI"] = "0"
    os.environ["FWS_TELEMETRY"] = "1" if args.telemetry else "0"
    os.environ["FWS_HEADLESS_PRINT_EVERY_TICKS"] = "0"
    os.environ["FWS_CHECKPOINT_EVERY_TICKS"] = "0"
    os.environ["FWS_CHECKPOINT_ON_EXIT"] = "0"
    os.environ["FWS_GRID_W"] = str(int(args.grid_w))
    os.environ["FWS_GRID_H"] = str(int(args.grid_h))
    os.environ["FWS_START_PER_TEAM"] = str(int(args.start_per_team))
    os.environ["FWS_MAX_AGENTS"] = str(int(args.max_agents))
    os.environ["FWS_RESPAWN"] = "1" if args.respawn else "0"
    os.environ["FWS_PPO_TICKS"] = str(int(args.ppo_ticks))
    os.environ["FWS_PPO_EPOCHS"] = str(int(args.ppo_epochs))
    os.environ["FWS_PPO_MINIB"] = str(int(args.ppo_minibatches))

    if args.cuda != "auto":
        _env_flag("FWS_CUDA", args.cuda == "on")
    if args.use_vmap != "auto":
        _env_flag("FWS_USE_VMAP", args.use_vmap == "on")


def _build_runtime():
    import config
    from main import seed_everything
    from engine.agent_registry import AgentsRegistry, COL_ALIVE
    from engine.grid import make_grid
    from engine.mapgen import add_random_walls, make_zones
    from engine.spawn import spawn_uniform_random
    from engine.tick import TickEngine
    from simulation.stats import SimulationStats

    seed_everything(int(getattr(config, "RNG_SEED", getattr(config, "SEED", 42))))

    grid = make_grid(config.TORCH_DEVICE)
    registry = AgentsRegistry(grid)
    stats = SimulationStats()
    add_random_walls(grid)
    zones = make_zones(config.GRID_HEIGHT, config.GRID_WIDTH, device=config.TORCH_DEVICE)
    spawn_uniform_random(registry, grid, per_team=config.START_AGENTS_PER_TEAM)
    engine = TickEngine(registry, grid, stats, zones=zones)
    return config, engine, registry, stats, COL_ALIVE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible direct TickEngine throughput benchmark.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--ticks", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--grid-w", type=int, default=64)
    parser.add_argument("--grid-h", type=int, default=64)
    parser.add_argument("--start-per-team", type=int, default=32)
    parser.add_argument("--max-agents", type=int, default=128)
    parser.add_argument("--use-vmap", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cuda", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--respawn", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--ppo-ticks", type=int, default=192)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatches", type=int, default=8)
    args = parser.parse_args()

    _apply_args_to_env(args)
    config, engine, registry, stats, col_alive = _build_runtime()

    for _ in range(int(args.warmup)):
        engine.run_tick()

    start = time.perf_counter()
    for _ in range(int(args.ticks)):
        engine.run_tick()
    elapsed = time.perf_counter() - start

    alive = int((registry.agent_data[:, col_alive] > 0.5).sum().item())
    result = {
        "device": str(config.TORCH_DEVICE),
        "profile": str(config.PROFILE),
        "grid": [int(config.GRID_WIDTH), int(config.GRID_HEIGHT)],
        "start_per_team": int(config.START_AGENTS_PER_TEAM),
        "max_agents": int(config.MAX_AGENTS),
        "use_vmap": bool(config.USE_VMAP),
        "respawn_enabled": bool(config.RESPAWN_ENABLED),
        "telemetry_enabled": bool(config.TELEMETRY_ENABLED),
        "ppo_window_ticks": int(config.PPO_WINDOW_TICKS),
        "ppo_epochs": int(config.PPO_EPOCHS),
        "ppo_minibatches": int(config.PPO_MINIBATCHES),
        "ticks": int(args.ticks),
        "warmup": int(args.warmup),
        "elapsed_s": elapsed,
        "ticks_per_s": (float(args.ticks) / elapsed) if elapsed > 0.0 else float("inf"),
        "final_tick": int(stats.tick),
        "alive_after": alive,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
