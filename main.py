"""Top-level runtime orchestration for Neural Abyss."""

from __future__ import annotations

import os

# Keep Ctrl+C mapped to KeyboardInterrupt on Windows before importing numerical stacks.
if os.name == "nt":
    if os.getenv("FWS_WIN_FORRTL_MITIGATE", "1").strip().lower() in ("1", "true", "yes", "y", "on", "t"):
        os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "TRUE")

import json
import time
import signal
import traceback
from pathlib import Path
from typing import Optional

import torch
import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import config
except ModuleNotFoundError:
    import sys as _sys
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import config
from simulation.stats import SimulationStats
from engine.agent_registry import (
    AgentsRegistry,
    COL_ALIVE,
    COL_HP,
    COL_TEAM,
    TEAM_BLUE_ID,
    TEAM_RED_ID,
)
from engine.tick import TickEngine
from engine.grid import make_grid
from engine.spawn import spawn_symmetric, spawn_uniform_random
from engine.mapgen import add_random_walls, make_zones
from utils.persistence import ResultsWriter
from utils.telemetry import TelemetrySession
from utils.checkpointing import CheckpointManager, resolve_checkpoint_path


# Reproducibility utilities

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNG state."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _config_snapshot() -> dict:
    """Return the JSON-serializable runtime configuration snapshot."""
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
            "SPAWN_MODE": str(getattr(config, "SPAWN_MODE", "uniform")),
            "SPAWN_ARCHER_RATIO": float(getattr(config, "SPAWN_ARCHER_RATIO", 0.4)),
            "RESPAWN_CHILD_UNIT_MODE": str(getattr(config, "RESPAWN_CHILD_UNIT_MODE", "inherit_parent_on_clone")),
            "RESPAWN_PARENT_SELECTION_MODE": str(getattr(config, "RESPAWN_PARENT_SELECTION_MODE", "random")),
        },
        "ARCHER_LOS_BLOCKS_WALLS": bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False)),
    }


def _telemetry_schema_manifest(grid: torch.Tensor, zones) -> dict:
    return {
        "schema_version": str(getattr(config, "TELEMETRY_SCHEMA_VERSION", "v2")),
        "lineage_fields": [
            "tick",
            "parent_id",
            "child_id",
            "parent_unit_type",
            "child_unit_type",
            "parent_team",
            "spawn_reason",
            "parent_generation",
            "child_generation",
            "mutation_flag",
            "rare_mutation",
            "mutation_delta_hp",
            "mutation_delta_atk",
            "mutation_delta_vis",
        ],
        "reward_fields": [
            "reward_total",
            "reward_individual_total",
            "reward_team_total",
            "reward_hp",
            "reward_kill_individual",
            "reward_damage_dealt_individual",
            "reward_damage_taken_penalty",
            "reward_contested_cp_individual",
            "reward_team_kill",
            "reward_team_death",
            "reward_team_cp",
            "reward_healing_recovered",
        ],
        "death_causes": ["combat", "metabolism", "environmental", "collision", "unknown"],
        "mechanics": {
            "archer_los_blocks_walls": bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False)),
            "metabolism_enabled": bool(getattr(config, "METABOLISM_ENABLED", True)),
            "heal_zones_enabled": bool(zones is not None and getattr(zones, "heal_mask", None) is not None),
            "cp_zone_count": int(len(getattr(zones, "cp_masks", []) or [])) if zones is not None else 0,
            "respawn_child_unit_mode": str(getattr(config, "RESPAWN_CHILD_UNIT_MODE", "inherit_parent_on_clone")),
            "respawn_parent_selection_mode": str(getattr(config, "RESPAWN_PARENT_SELECTION_MODE", "random")),
        },
    }


# Small helpers

def _mkdir_p(p: Path) -> None:
    """
    Create a directory tree when it does not already exist.

    This is used to ensure run directories exist even after crashes.

    Uses ``parents=True`` and ``exist_ok=True``.
    """
    p.mkdir(parents=True, exist_ok=True)


def _atomic_json_dump(obj: dict, path: Path) -> None:
    """
    Write JSON atomically to prevent partial/corrupt files.

    Atomic write pattern:
    1) Write to a temporary file (same directory)
    2) Rename/replace temp file -> final file

    Why do this?
    ------------
    If the program crashes mid-write (power loss, forced kill, exception),
    you might end up with half-written JSON (invalid file).
    The temp+replace pattern ensures the final file is either:
    - old version, or
    - fully complete new version
    but never a corrupted half version.

    `os.replace` is atomic on POSIX and "effectively atomic" on Windows in practice.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# Optional video recording (simple occupancy visualization)

class _SimpleRecorder:
    """
    A minimal video recorder that writes the occupancy grid as an AVI file.

    It uses OpenCV (`cv2`) if available. If OpenCV is missing or config disables
    recording, this recorder becomes a no-op (enabled=False).

    Important: this recorder is intentionally "simple":
    - it records a coarse grid view, not a fancy UI
    - it uses a small fixed palette
    - it scales pixels using nearest neighbor so cells stay crisp

    Performance note
    ----------------
    Writing video frames is I/O-heavy. This implementation writes from the main process,
    which can slow down simulation if you record every tick.
    The code therefore supports recording every N ticks.
    """

    def __init__(self, run_dir: Path, grid: torch.Tensor, fps: int, scale: int):
        self.enabled = False
        self.writer = None
        self.path = None
        self.size = None
        self.grid = grid

        # Enable only if config allows and OpenCV is installed.
        if not getattr(config, "RECORD_VIDEO", False) or cv2 is None:
            return

        h, w = int(grid.size(1)), int(grid.size(2))

        # Output video frame size after scaling
        self.size = (w * scale, h * scale)

        self.path = run_dir / "simulation_raw.avi"

        # MJPG is widely supported and easy to encode.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), self.size)

        # Confirm writer is usable
        if self.writer is not None and self.writer.isOpened():
            self.enabled = True
            print(f"[video] recording -> {self.path}")

            # Palette maps occupancy IDs -> RGB colors
            # You can extend this palette if occupancy encoding changes.
            self.palette = np.array(
                [
                    [30, 30, 30],    # 0 = empty
                    [80, 80, 80],    # 1 = wall
                    [220, 80, 80],   # 2 = red team
                    [80, 120, 240],  # 3 = blue team
                ],
                dtype=np.uint8,
            )
        else:
            print(f"[video] ERROR: could not open writer for {self.path}.")

    def write(self) -> None:
        """
        Capture the current grid and write one frame.

        Steps
        -----
        1) Pull occupancy channel from GPU to CPU (numpy needs CPU memory)
        2) Map occupancy values to colors using palette indexing
        3) Convert RGB -> BGR (OpenCV convention)
        4) Scale up using nearest neighbor
        5) Write frame to video

        Technical detail: `.detach()` prevents autograd tracking; we are logging, not training.

        Advanced note (why `.contiguous()`):
        -----------------------------------
        Some tensor views are non-contiguous (strided). Converting those to numpy
        can fail or be slow. `.contiguous()` ensures a compact memory layout.
        """
        if not self.enabled:
            return

        # occupancy channel assumed to be grid[0]
        occ = self.grid[0].detach().contiguous().to("cpu").numpy().astype(np.uint8)

        # Map occupancy IDs to RGB colors
        frame_rgb = self.palette[occ % len(self.palette)]

        # OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Scale image while preserving pixelated look
        frame_resized = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_NEAREST)

        # Write to file
        self.writer.write(frame_resized)

    def close(self) -> None:
        """Release the video writer cleanly."""
        if self.enabled and self.writer is not None:
            self.writer.release()
            print(f"[video] saved -> {self.path}")


class _NoopRecorder:
    """No-op recorder used for explicit inspector/no-output mode."""

    enabled = False

    def write(self) -> None:
        return

    def close(self) -> None:
        return


def _inspector_no_output_mode_active() -> bool:
    """True when explicit inspector viewer mode should avoid all run output creation."""
    mode = str(getattr(config, "INSPECTOR_MODE", "off")).strip().lower()
    if mode in ("ui_no_output", "inspect", "inspector", "no_output", "viewer_no_output"):
        return True
    return bool(getattr(config, "INSPECTOR_UI_NO_OUTPUT", False))


def _infer_resume_run_dir_from_checkpoint_path(checkpoint_path: str) -> Path:
    """Infer original run_dir from checkpoint path (supports file/dir/checkpoints-root inputs)."""
    ckpt_dir, _ = resolve_checkpoint_path(str(checkpoint_path))
    ckpt_base = ckpt_dir.parent
    if ckpt_base.name.lower() != "checkpoints":
        raise RuntimeError(
            f"[main] Cannot infer run_dir from checkpoint path (expected parent folder 'checkpoints'): {ckpt_dir}"
        )
    run_dir = ckpt_base.parent
    if (not run_dir.exists()) or (not run_dir.is_dir()):
        raise RuntimeError(f"[main] Inferred run_dir does not exist: {run_dir}")
    return run_dir


# Headless loop (no UI)

def _headless_loop(
    engine: TickEngine,
    stats: SimulationStats,
    registry: AgentsRegistry,
    grid: torch.Tensor,
    results_writer: ResultsWriter,
    limit: int,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> None:
    """Run the simulation loop without the UI."""
    from utils.profiler import torch_profiler_ctx, nvidia_smi_summary
    from utils.sanitize import runtime_sanity_check

    # Optional torch profiler context manager (enabled via config)
    with torch_profiler_ctx() as profiler:
        try:
            # - replace ONLY while header + first lines
            # - add shutdown check at bottom of loop
            while limit == 0 or stats.tick < limit:
                # If a signal requested shutdown, do not start a new tick.
                # `engine.shutdown_requested` is attached in main() after signal registration.
                if getattr(engine, "shutdown_requested", {}).get("flag", False):
                    break

                # 1) Advance the simulation by exactly one tick
                tick_metrics = engine.run_tick()


                # 2) Log tick summary (non-blocking)
                results_writer.write_tick(stats.as_row())

                # 3) Log deaths/kill events (batch)
                deaths = stats.drain_dead_log()
                if deaths:
                    results_writer.write_deaths(deaths)
                # 3b) Additive headless telemetry sidecar (decoupled from print mode)
                gpu_probe_line = None
                telemetry = getattr(engine, "telemetry", None)
                if telemetry is not None and getattr(telemetry, "enabled", False):
                    try:
                        # Poll GPU only on summary ticks (independent of console prints).
                        want_gpu = bool(getattr(telemetry, "headless_summary_include_gpu", False))
                        every = int(getattr(telemetry, "tick_summary_every", 0))
                        if want_gpu and every > 0 and (int(stats.tick) % every) == 0:
                            gpu_probe_line = nvidia_smi_summary()

                        if hasattr(telemetry, "on_headless_tick"):
                            telemetry.on_headless_tick(
                                tick=int(stats.tick),
                                tick_metrics=(tick_metrics if isinstance(tick_metrics, dict) else None),
                                gpu_line=gpu_probe_line,
                            )
                    except Exception:
                        # Fail-safe: telemetry must not crash headless runs.
                        pass
                # After completing this tick’s output work, exit if shutdown requested.
                if getattr(engine, "shutdown_requested", {}).get("flag", False):
                    break


                # 4) Periodic sanity check
                # A sanity check is a defensive validation routine to catch:
                # - NaNs/infs
                # - impossible values (negative HP, out-of-bounds positions)
                # - corrupted tensors
                # Running it every tick would be expensive, so we do it periodically.
                if (stats.tick % 500) == 0:
                    runtime_sanity_check(grid, registry.agent_data)

                # 5) Profiler bookkeeping
                # If profiler is enabled, advance its internal step counter.
                if profiler is not None:
                    profiler.step()

                # 6) Checkpointing
                if checkpoint_manager is not None:
                    # Manual checkpoint trigger file:
                    # create "checkpoint.now" (or configured filename) in run_dir to force save.
                    trigger_path = Path(results_writer.run_dir) / str(getattr(config, "CHECKPOINT_TRIGGER_FILE", "checkpoint.now"))

                    checkpoint_manager.maybe_save_trigger_file(
                        trigger_path=trigger_path,
                        engine=engine,
                        registry=registry,
                        stats=stats,
                        default_pin=bool(getattr(config, "CHECKPOINT_PIN_ON_MANUAL", True)),
                        pin_tag=str(getattr(config, "CHECKPOINT_PIN_TAG", "manual")),
                        keep_last_n=int(getattr(config, "CHECKPOINT_KEEP_LAST_N", 1)),
                    )

                    # Periodic checkpoints (every N ticks)
                    checkpoint_manager.maybe_save_periodic(
                        engine=engine,
                        registry=registry,
                        stats=stats,
                        every_ticks=int(getattr(config, "CHECKPOINT_EVERY_TICKS", 0)),
                        keep_last_n=int(getattr(config, "CHECKPOINT_KEEP_LAST_N", 1)),
                    )

                # 7) Periodic status print
                pe = int(getattr(config, "HEADLESS_PRINT_EVERY_TICKS", 100))
                if pe > 0 and (stats.tick % pe) == 0:
                    # Reuse GPU probe if already collected for telemetry on this tick.
                    gpu = (gpu_probe_line if gpu_probe_line is not None else nvidia_smi_summary()) or "-"
                    lvl = int(getattr(config, "HEADLESS_PRINT_LEVEL", 1))
                    show_gpu = bool(getattr(config, "HEADLESS_PRINT_GPU", True))

                    # Always show core tick info
                    msg = (
                        f"Tick {stats.tick:7d} | "
                        f"Score R {stats.red.score:7.2f} B {stats.blue.score:7.2f} | "
                        f"Elapsed {stats.elapsed_seconds:7.2f}s"
                    )

                    if lvl >= 1:
                        agent_data = registry.agent_data
                        alive = agent_data[:, COL_ALIVE] > 0.5
                        red_alive = int((alive & (agent_data[:, COL_TEAM] == TEAM_RED_ID)).sum().item())
                        blue_alive = int((alive & (agent_data[:, COL_TEAM] == TEAM_BLUE_ID)).sum().item())

                        msg += f" | Alive R {red_alive:4d} B {blue_alive:4d}"
                        msg += f" | K/D R {stats.red.kills}/{stats.red.deaths} B {stats.blue.kills}/{stats.blue.deaths}"
                        msg += f" | CP R {stats.red.cp_points:.1f} B {stats.blue.cp_points:.1f}"

                    if lvl >= 2:
                        agent_data = registry.agent_data
                        alive = agent_data[:, COL_ALIVE] > 0.5
                        red_hp = agent_data[:, COL_HP][alive & (agent_data[:, COL_TEAM] == TEAM_RED_ID)]
                        blue_hp = agent_data[:, COL_HP][alive & (agent_data[:, COL_TEAM] == TEAM_BLUE_ID)]

                        rmean = float(red_hp.mean().item()) if red_hp.numel() else 0.0
                        bmean = float(blue_hp.mean().item()) if blue_hp.numel() else 0.0

                        msg += f" | AvgHP R {rmean:5.1f} B {bmean:5.1f}"
                        msg += f" | DMG+ R {stats.red.dmg_dealt:.1f} B {stats.blue.dmg_dealt:.1f}"

                    if show_gpu:
                        msg += f" | GPU {gpu}"

                    print(msg)

        except KeyboardInterrupt:
            # Let outer main() handle final flush/shutdown
            print("\n[main] Interrupted; shutting down gracefully.")


# Main entry point

def main() -> None:
    """Initialize the runtime and execute the selected run loop."""
    # PyTorch can use different matmul kernels; "high" may improve performance on some GPUs.
    torch.set_float32_matmul_precision("high")

    # Seed from resolved config value (which already honors FWS_SEED overrides).
    # This ensures deterministic startup seeding even when the env var is absent and
    # only the config default (RNG_SEED) is in effect.
    seed = int(getattr(config, "RNG_SEED", getattr(config, "SEED", 42)))
    seed_everything(seed)
    print(f"[main] Using deterministic seed: {seed}")

    # Print a one-line banner summarizing config (useful for logs)
    print(config.summary_str())

    inspector_no_output_mode = _inspector_no_output_mode_active()
    if inspector_no_output_mode:
        print("[main] Inspector UI no-output mode enabled (no results/telemetry/checkpoints/files).")

    # 1) Checkpoint loading or fresh world creation
    checkpoint_data = None
    checkpoint_path = getattr(config, "CHECKPOINT_PATH", "")

    if checkpoint_path:
        # Resume mode
        print(f"[main] Resuming from checkpoint: {checkpoint_path}")

        # Load on CPU first (reduces GPU fragmentation / spikes)
        checkpoint_data = CheckpointManager.load(checkpoint_path, map_location="cpu")

        # Restore world grid and zones
        grid = checkpoint_data["world"]["grid"].to(config.TORCH_DEVICE)

        zones = CheckpointManager.zones_from_checkpoint(
            checkpoint_data["world"],
            device=torch.device(config.TORCH_DEVICE),
        )

        # Create empty containers; they will be populated by apply_loaded_checkpoint()
        registry = AgentsRegistry(grid)
        stats = SimulationStats()

        print("[main] Checkpoint loaded - world restored, will restore runtime next.")

    else:
        # Fresh start mode
        grid = make_grid(config.TORCH_DEVICE)
        registry = AgentsRegistry(grid)
        stats = SimulationStats()

        # Map generation (walls)
        add_random_walls(grid)

        # Capture zones / control points
        zones = make_zones(config.GRID_HEIGHT, config.GRID_WIDTH, device=config.TORCH_DEVICE)

        # Spawn initial agents
        spawn_mode = str(getattr(config, "SPAWN_MODE", "uniform")).strip().lower()

        if spawn_mode == "symmetric":
            spawn_symmetric(registry, grid, per_team=config.START_AGENTS_PER_TEAM)
            print("[SYMMETRIC_SPAWNING]")
        else:
            spawn_uniform_random(registry, grid, per_team=config.START_AGENTS_PER_TEAM)
            print("[UNIFORM_RANDOM_SPAWNING]")

    # 2) Create the tick engine (core simulation logic)
    print("[INITIATING_TICK_ENGINE]")
    engine = TickEngine(registry, grid, stats, zones=zones)

    # 3) If resuming, apply runtime state (brains, RNG, PPO buffers, etc.)
    if checkpoint_data is not None:
        CheckpointManager.apply_loaded_checkpoint(
            checkpoint_data,
            engine=engine,
            registry=registry,
            stats=stats,
            device=torch.device(config.TORCH_DEVICE),
        )
        checkpoint_tick = int(checkpoint_data.get("meta", {}).get("tick", stats.tick))
        if int(stats.tick) != checkpoint_tick:
            raise RuntimeError(
                f"[main] checkpoint tick mismatch after restore: stats.tick={int(stats.tick)} manifest_tick={checkpoint_tick}"
            )
        print("[main] Runtime state restored from checkpoint.")

    # 4) Results directory & background logging process
    resume_continuity_active = False
    results_writer = None
    run_dir: Optional[Path] = None
    checkpoint_manager = None
    telemetry = None

    if inspector_no_output_mode:
        # Explicit no-output inspector UI mode: no results folder, no writer, no telemetry, no checkpoints.
        recorder = _NoopRecorder()
    else:
        resume_requested = bool(checkpoint_data is not None)
        continuity_cfg = bool(getattr(config, "RESUME_OUTPUT_CONTINUITY", True))
        force_new_on_resume = bool(getattr(config, "RESUME_FORCE_NEW_RUN", False))
        resume_continuity_active = bool(resume_requested and continuity_cfg and (not force_new_on_resume))

        results_writer = ResultsWriter()

        if resume_continuity_active:
            inferred_run_dir = _infer_resume_run_dir_from_checkpoint_path(checkpoint_path)
            run_dir = Path(
                results_writer.start(
                    config_obj=_config_snapshot(),
                    run_dir=str(inferred_run_dir),
                    append_existing=True,
                    strict_csv_schema=bool(getattr(config, "RESUME_APPEND_STRICT_CSV_SCHEMA", True)),
                )
            )
            print(f"[main] Results -> {run_dir} (resume-in-place append)")
        else:
            # start() creates a fresh run directory and writes config.json there
            run_dir = Path(results_writer.start(config_obj=_config_snapshot()))
            # Defensive: ensure directory exists (should already, but safe)
            _mkdir_p(run_dir)
            print(f"[main] Results -> {run_dir}")

        # 5) Checkpoint manager (saves into run_dir/checkpoints/)
        checkpoint_manager = CheckpointManager(run_dir)

        # 6) Telemetry (optional, must never crash sim)
        try:
            telemetry = TelemetrySession(run_dir)

            if telemetry.enabled:
                # Attach to engine so it can emit events
                engine.telemetry = telemetry

                # Give telemetry access to registry/stats for context
                telemetry.attach_context(
                    registry=registry,
                    stats=stats,
                    ppo_runtime=getattr(engine, "_ppo", None),
                )

                telemetry_manifest = _telemetry_schema_manifest(grid, zones)
                if resume_continuity_active and checkpoint_data is not None:
                    telemetry.validate_schema_manifest_compat(telemetry_manifest)
                else:
                    telemetry.write_schema_manifest(telemetry_manifest)

                # Resume-in-place: preserve original run_meta.json by default (avoid overwrite).
                if not (resume_continuity_active and checkpoint_data is not None):
                    telemetry.write_run_meta(
                        {
                            "schema_version": getattr(config, "TELEMETRY_SCHEMA_VERSION", "v2"),
                            "config": _config_snapshot(),
                            "seed": int(getattr(config, "RNG_SEED", getattr(config, "SEED", 42))),
                            "device": str(grid.device),
                            "grid_h": int(grid.shape[1]),
                            "grid_w": int(grid.shape[2]),
                            "start_tick": int(stats.tick),
                            "resume": bool(checkpoint_data is not None),
                            "resume_from": (checkpoint_path if checkpoint_data is not None else None),
                        }
                    )

                if checkpoint_data is not None:
                    telemetry.record_resume(tick=int(stats.tick), checkpoint_path=str(checkpoint_path))

                # Bootstrap initial population as "birth events" so lineage tracking is consistent.
                # Skip on resume-in-place to avoid duplicate bootstrap birth rows in the same lineage files.
                if not (resume_continuity_active and checkpoint_data is not None):
                    telemetry.bootstrap_from_registry(registry, tick=int(stats.tick), note="bootstrap_run_start")

        except Exception as e:
            print(f"[main] Telemetry init failed: {e}")

        # 7) Optional video recorder
        recorder = _SimpleRecorder(
            run_dir,
            grid,
            fps=getattr(config, "VIDEO_FPS", 30),
            scale=getattr(config, "VIDEO_SCALE", 4),
        )

    _orig_run_tick = engine.run_tick

    def _run_tick_with_recording():
        """Record frames around the original tick callback when enabled."""
        out = _orig_run_tick()

        if recorder.enabled and (stats.tick % getattr(config, "VIDEO_EVERY_TICKS", 1) == 0):
            recorder.write()
        return out
    engine.run_tick = _run_tick_with_recording

    shutdown_requested = {"flag": False}

    def _signal_handler(signum, _frame) -> None:
        """Request shutdown after the current tick completes."""
        shutdown_requested["flag"] = True
        print(f"\n[main] Signal {signum} received; finishing current tick before shutdown.")

    # Register SIGINT and SIGTERM when available
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            # Some environments (e.g., notebooks) restrict signal registration
            pass

    # - after signal registration loop in main(), add engine attachment
    # Expose shutdown flag to engine (headless loop + UI loop can poll it)
    try:
        engine.shutdown_requested = shutdown_requested
    except Exception:
        pass

    # 9) Run loop (UI or headless)
    start_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()

    status = "ok"
    error_msg = None
    crash_trace = None
    viewer = None

    try:
        if config.ENABLE_UI or inspector_no_output_mode:
            # Import the viewer lazily so pure headless runs do not require the
            # optional pygame-ce dependency at module import time.
            from ui.viewer import Viewer

            viewer = Viewer(grid, cell_size=config.CELL_SIZE)
            if checkpoint_data is not None:
                viewer.apply_checkpoint_state(checkpoint_data.get("viewer", {}))

            # Pass run_dir so UI can trigger manual checkpoints inside the run folder
            viewer.run(
                engine,
                registry,
                stats,
                tick_limit=config.TICK_LIMIT,
                target_fps=config.TARGET_FPS,
                run_dir=(None if inspector_no_output_mode else run_dir),
            )
        else:
            # Headless mode: run as fast as possible (or as configured)
            _headless_loop(
                engine,
                stats,
                registry,
                grid,
                results_writer,
                limit=config.TICK_LIMIT,
                checkpoint_manager=checkpoint_manager,
            )

        # - after UI/headless run completes inside the try: block,
        #   add conversion of flag to KeyboardInterrupt to reuse shutdown path
        # If a signal requested shutdown, translate it into the existing KeyboardInterrupt path.
        if shutdown_requested.get("flag", False):
            raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[main] Interrupted; flushing logs...")
        status = "interrupted"

    except Exception as e:
        # Crash path: store trace and re-raise
        status = "crash"
        error_msg = str(e)
        crash_trace = "".join(traceback.format_exc())

        if run_dir is not None:
            _mkdir_p(run_dir)
            (run_dir / "crash_trace.txt").write_text(crash_trace, encoding="utf-8")

        raise

    finally:
        # 10) Persist final death logs
        try:
            deaths = stats.drain_dead_log()
            if deaths and results_writer is not None:
                results_writer.write_deaths(deaths)
        except Exception:
            pass

        # 11) Close telemetry first so final snapshots/events reach the same horizon
        #     before checkpoint + summary are frozen.
        try:
            if telemetry is not None:
                telemetry.close()
        except Exception:
            pass

        # 12) On-exit checkpoint
        if checkpoint_manager is not None and bool(getattr(config, "CHECKPOINT_ON_EXIT", True)):
            try:
                viewer_state = viewer.capture_state() if viewer is not None else None
                out = checkpoint_manager.save_atomic(
                    engine=engine,
                    registry=registry,
                    stats=stats,
                    viewer_state=viewer_state,
                    notes="on_exit",
                )
                checkpoint_manager.prune_keep_last_n(int(getattr(config, "CHECKPOINT_KEEP_LAST_N", 1)))
                print("[checkpoint] on-exit saved:", out.name)
            except Exception as ex:
                print("[checkpoint] on-exit FAILED:", type(ex).__name__, ex)

        # 13) Final summary (atomic JSON)
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
            if run_dir is not None:
                _mkdir_p(run_dir)
                _atomic_json_dump(summary, run_dir / "summary.json")
        except Exception as e:
            # Worst-case fallback: plain text summary
            try:
                if run_dir is not None:
                    (run_dir / "summary_fallback.txt").write_text(
                        f"FAILED TO WRITE JSON SUMMARY: {e}\n{summary!r}",
                        encoding="utf-8",
                    )
            except Exception:
                pass

        # 14) Shutdown background writer and recorder
        try:
            if results_writer is not None:
                results_writer.close()
        except Exception:
            pass

        try:
            recorder.close()
        except Exception:
            pass

        print("[main] Shutdown complete.")


if __name__ == "__main__":
    main()
