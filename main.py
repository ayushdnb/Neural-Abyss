from __future__ import annotations

import os
import json
import time
import signal
import traceback
from pathlib import Path
from typing import Optional

import torch
import numpy as np
try:
    import cv2 #type:ignore # optional; recording becomes no-op if not installed
except Exception:
    cv2 = None

# Local modules
import config
from simulation.stats import SimulationStats
from engine.agent_registry import AgentsRegistry
from engine.tick import TickEngine
from engine.grid import make_grid
from engine.spawn import spawn_symmetric, spawn_uniform_random
from engine.mapgen import add_random_walls, make_zones
from utils.persistence import ResultsWriter
from utils.telemetry import TelemetrySession
from ui.viewer import Viewer


from utils.checkpointing import CheckpointManager

def seed_everything(seed: int):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ------------------------------- helpers -------------------------------

def _config_snapshot() -> dict:
    """Light, serializable snapshot of current config for run metadata."""
    return {
        "summary": config.summary_str(),
        "GRID_W": config.GRID_WIDTH,
        "GRID_H": config.GRID_HEIGHT,
        "START_PER_TEAM": config.START_AGENTS_PER_TEAM,
        "MAX_AGENTS": getattr(config, "MAX_AGENTS", None),
        "OBS_DIM": config.OBS_DIM,
        "NUM_ACTIONS": config.NUM_ACTIONS,
        "MAX_HP": config.MAX_HP,
        "BASE_ATK": config.BASE_ATK,
        "AMP": config.amp_enabled() if hasattr(config, "amp_enabled") else False,
        "PPO": {
            "UPDATE_TICKS": getattr(config, "PPO_UPDATE_TICKS", 5),
            "LR": getattr(config, "PPO_LR", 3e-4),
            "EPOCHS": getattr(config, "PPO_EPOCHS", 3),
            "CLIP": getattr(config, "PPO_CLIP", 0.2),
            "ENTROPY": getattr(config, "PPO_ENTROPY_BONUS", 0.01),
            "VCOEF": getattr(config, "PPO_VALUE_COEF", 0.5),
            "MAX_GN": getattr(config, "PPO_MAX_GRAD_NORM", 1.0),
        },
        "REWARDS": {
            "KILL": config.TEAM_KILL_REWARD,
            "DMG_DEALT": config.TEAM_DMG_DEALT_REWARD,
            "DEATH": config.TEAM_DEATH_PENALTY,
            "DMG_TAKEN": config.TEAM_DMG_TAKEN_PENALTY,
            "CAPTURE_TICK": getattr(config, "CP_REWARD_PER_TICK", None),
        },
        "UI": {
            "ENABLE_UI": config.ENABLE_UI,
            "CELL_SIZE": config.CELL_SIZE,
            "TARGET_FPS": config.TARGET_FPS,
        },
        "SPAWN": {
            "SPAWN_ARCHER_RATIO": float(getattr(config, "SPAWN_ARCHER_RATIO", 0.4)),
        },
        "ARCHER_LOS_BLOCKS_WALLS": bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False)),
    }


def _seed_all_from_env() -> Optional[int]:
    raw = os.getenv("FWS_SEED", "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        return seed
    except (ValueError, TypeError):
        return None


def _mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_json_dump(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --------------------------- simple video recorder ---------------------------

class _SimpleRecorder:
    def __init__(self, run_dir: Path, grid: torch.Tensor, fps: int, scale: int):
        self.enabled = False
        self.writer = None
        self.path = None
        self.size = None
        self.grid = grid
        if not getattr(config, "RECORD_VIDEO", False) or cv2 is None:
            return

        h, w = int(grid.size(1)), int(grid.size(2))
        self.size = (w * scale, h * scale)
        self.path = run_dir / "simulation_raw.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), self.size)

        if self.writer is not None and self.writer.isOpened():
            self.enabled = True
            print(f"[video] recording → {self.path}")
            # index 0 channel assumed occupancy; palette can be expanded later
            self.palette = np.array([
                [30, 30, 30],   # 0
                [80, 80, 80],   # 1
                [220, 80, 80],  # 2
                [80, 120, 240], # 3
            ], dtype=np.uint8)
        else:
            print(f"[video] ERROR: could not open writer for {self.path}.")

    def write(self) -> None:
        if not self.enabled:
            return
        occ = self.grid[0].detach().contiguous().to("cpu").numpy().astype(np.uint8)
        frame = self.palette[occ % len(self.palette)]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_NEAREST)
        self.writer.write(frame_resized)

    def close(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.release()
            print(f"[video] saved → {self.path}")


# ------------------------------ run loops ------------------------------

def _headless_loop(engine: TickEngine, stats: SimulationStats, reg: AgentsRegistry,
                   grid: torch.Tensor, rw: ResultsWriter, limit: int) -> None:
    # Imported lazily to avoid any circular import pitfalls
    from utils.profiler import torch_profiler_ctx, nvidia_smi_summary
    from utils.sanitize import runtime_sanity_check

    with torch_profiler_ctx() as prof:
        try:
            while limit == 0 or stats.tick < limit:
                engine.run_tick()
                rw.write_tick(stats.as_row())
                deaths = stats.drain_dead_log()
                if deaths:
                    rw.write_deaths(deaths)

                if (stats.tick % 500) == 0:
                    runtime_sanity_check(grid, reg.agent_data)

                if prof is not None:
                    prof.step()

                if (stats.tick % 100) == 0:
                    gpu = nvidia_smi_summary() or "-"
                    print(
                        f"Tick {stats.tick:7d} | "
                        f"Red {stats.red.score:8.2f} | Blue {stats.blue.score:8.2f} | "
                        f"Elapsed {stats.elapsed_seconds:7.2f}s | GPU {gpu}"
                    )
        except KeyboardInterrupt:
            print("\n[main] Interrupted — shutting down gracefully.")


# --------------------------------- main --------------------------------

def main() -> None:
    torch.set_float32_matmul_precision("high")
    seed = _seed_all_from_env()
    if seed is not None:
        print(f"[main] Using deterministic seed: {seed}")

    # This prints your banner like: [final_war_sim] dev=... grid=... etc.
    print(config.summary_str())

    # ------------------------------------------------------------------
    # Checkpoint handling: if a checkpoint path is given, load it first.
    # ------------------------------------------------------------------
    ckpt = None  # will hold loaded checkpoint data if any
    checkpoint_path = getattr(config, "CHECKPOINT_PATH", "")

    if checkpoint_path:
        print(f"[main] Resuming from checkpoint: {checkpoint_path}")
        # Load checkpoint onto CPU initially to avoid GPU memory fragmentation
        ckpt = CheckpointManager.load(checkpoint_path, map_location="cpu")

        # Extract world components (grid and zones) from checkpoint
        # The grid tensor must be moved to the target device
        grid = ckpt["world"]["grid"].to(config.TORCH_DEVICE)
        zones = CheckpointManager.zones_from_checkpoint(
            ckpt["world"], device=torch.device(config.TORCH_DEVICE)
        )

        # Create fresh registry and stats objects; they will be populated
        # from checkpoint data after the engine is created.
        registry = AgentsRegistry(grid)
        stats = SimulationStats()

        # No need to add walls or spawn agents now – they come from checkpoint.
        print("[main] Checkpoint loaded – world restored, will restore runtime next.")
    else:
        # No checkpoint: build a brand new world.
        # Create grid (occupancy and feature channels)
        grid = make_grid(config.TORCH_DEVICE)
        registry = AgentsRegistry(grid)
        stats = SimulationStats()

        # Add random walls to the grid
        add_random_walls(grid)

        # Create capture zones (control points)
        zones = make_zones(config.GRID_HEIGHT, config.GRID_WIDTH, device=config.TORCH_DEVICE)

        # Spawn initial agents according to environment variable or default
        spawn_mode = os.getenv("FWS_SPAWN_MODE", "uniform").lower()
        if spawn_mode == "symmetric":
            spawn_symmetric(registry, grid, per_team=config.START_AGENTS_PER_TEAM)
            print('[SYMMETRIC_SPAWNING]')
        else:
            spawn_uniform_random(registry, grid, per_team=config.START_AGENTS_PER_TEAM)
            print('[UNIFORM_RANDOM_SPAWNING]')

    # ------------------------------------------------------------------
    # Create the tick engine (handles all simulation logic)
    # ------------------------------------------------------------------
    print('[INITIATING_TICK_ENGINE]')
    engine = TickEngine(registry, grid, stats, zones=zones)

    # ------------------------------------------------------------------
    # If we loaded a checkpoint, apply all runtime state to the engine,
    # registry, and stats (brains, PPO buffers, scores, respawn controller, RNG).
    # ------------------------------------------------------------------
    if ckpt is not None:
        CheckpointManager.apply_loaded_checkpoint(
            ckpt,
            engine=engine,
            registry=registry,
            stats=stats,
            device=torch.device(config.TORCH_DEVICE),
        )
        print("[main] Runtime state restored from checkpoint.")

    # ------------------------------------------------------------------
    # Results directory and logging setup
    # ------------------------------------------------------------------
    rw = ResultsWriter()
    # The `start` method creates a new run directory and returns its path.
    # We pass a snapshot of the config to be saved as metadata.
    run_dir = Path(rw.start(config_obj=_config_snapshot()))  # ensure created by ResultsWriter
    _mkdir_p(run_dir)  # defensive: guarantee directory exists
    print(f"[main] Results → {run_dir}")

    # ------------------------------------------------------------------
    # Minimal scientific telemetry (AgentLife + LineageEdges + core events)
    # ------------------------------------------------------------------
    telemetry = None
    try:
        telemetry = TelemetrySession(run_dir)
        if telemetry.enabled:
            engine.telemetry = telemetry
            telemetry.attach_context(registry=registry, stats=stats)
            telemetry.write_run_meta({
                "schema_version": getattr(config, "TELEMETRY_SCHEMA_VERSION", "v2"),
                "config": _config_snapshot(),
                "seed": getattr(config, "SEED", None),
                "device": str(grid.device),
                "grid_h": int(grid.shape[1]),
                "grid_w": int(grid.shape[2]),
                "start_tick": int(stats.tick),
                "resume": bool(ckpt is not None),
                "resume_from": (checkpoint_path if ckpt is not None else None),
            })
            if ckpt is not None:
                telemetry.record_resume(tick=int(stats.tick), checkpoint_path=str(checkpoint_path))
            # Seed births for initial population present at run start
            telemetry.bootstrap_from_registry(registry, tick=int(stats.tick), note="bootstrap_run_start")
    except Exception as e:
        # Telemetry must never crash the sim unless TELEMETRY_ABORT_ON_ANOMALY is set inside telemetry itself.
        print(f"[main] Telemetry init failed: {e}")

    # ------------------------------------------------------------------
    # Optional video recorder (depends on grid)
    # ------------------------------------------------------------------
    recorder = _SimpleRecorder(
        run_dir, grid,
        fps=getattr(config, "VIDEO_FPS", 30),
        scale=getattr(config, "VIDEO_SCALE", 4),
    )

    # Wrap the engine's run_tick to also record frames if enabled
    _orig_run_tick = engine.run_tick

    def _run_tick_with_recording():
        _orig_run_tick()
        if recorder.enabled and (stats.tick % getattr(config, "VIDEO_EVERY_TICKS", 1) == 0):
            recorder.write()

    engine.run_tick = _run_tick_with_recording

    # ------------------------------------------------------------------
    # Graceful signal handling
    # ------------------------------------------------------------------
    shutdown_requested = {"flag": False}

    def _signal_handler(signum, frame):
        shutdown_requested["flag"] = True
        print(f"\n[main] Signal {signum} received — will finish current tick and shut down.")

    # Register common signals (SIGTERM present on Windows too)
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    start_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()
    status = "ok"
    error_msg = None
    crash_trace = None

    try:
        if config.ENABLE_UI:
            viewer = Viewer(grid, cell_size=config.CELL_SIZE)
            # Pass run_dir to viewer so manual checkpoints can be saved inside the run's folder
            viewer.run(engine, registry, stats,
                       tick_limit=config.TICK_LIMIT,
                       target_fps=config.TARGET_FPS,
                       run_dir=run_dir)   # <-- added run_dir argument
        else:
            _headless_loop(engine, stats, registry, grid, rw, limit=config.TICK_LIMIT)
    except KeyboardInterrupt:
        print("\n[main] Interrupted — flushing logs…")
        status = "interrupted"
    except Exception as e:
        status = "crash"
        error_msg = str(e)
        crash_trace = "".join(traceback.format_exc())
        _mkdir_p(run_dir)
        (run_dir / "crash_trace.txt").write_text(crash_trace, encoding="utf-8")
        raise
    finally:
        # Persist final artifacts even on crash/interrupt
        try:
            deaths = stats.drain_dead_log()
            if deaths:
                rw.write_deaths(deaths)
        except Exception:
            pass

        # Final summary (atomic)
        try:
            summary = {
                "status": status,
                "started_at": start_ts,
                "duration_sec": round(time.time() - start_time, 3),
                "final_tick": int(stats.tick),
                "elapsed_seconds": float(stats.elapsed_seconds),
                "scores": {"red": float(stats.red.score), "blue": float(stats.blue.score)},
                "error": error_msg,
            }
            _mkdir_p(run_dir)
            _atomic_json_dump(summary, run_dir / "summary.json")
        except Exception as e:
            try:
                (run_dir / "summary_fallback.txt").write_text(
                    f"FAILED TO WRITE JSON SUMMARY: {e}\n{summary!r}", encoding="utf-8"
                )
            except Exception:
                pass

        # Flush telemetry first (so it can use final ticks and won't race writer shutdown)
        try:
            if telemetry is not None:
                telemetry.close()
        except Exception:
            pass

        # Close writer & recorder
        try:
            rw.close()
        except Exception:
            pass
        try:
            recorder.close()
        except Exception:
            pass

        print("[main] Shutdown complete.")

if __name__ == "__main__":
    main()