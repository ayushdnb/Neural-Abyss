from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List   # added List (B1 style)

import numpy as np
import torch

import config


class CheckpointError(RuntimeError):
    """Custom exception for checkpoint-related errors."""
    pass


def _now_stamp() -> str:
    """Return current timestamp formatted as YYYY-MM-DD_HH-MM-SS."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write text to a file.
    
    Args:
        path: Path object where file should be written
        text: String content to write
    """
    # Create temporary file path by adding .tmp suffix
    tmp = path.with_suffix(path.suffix + ".tmp")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file with proper flushing and fsync for durability
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    
    # Atomic replace - this is atomic on most filesystems
    os.replace(tmp, path)


def _atomic_json_dump(path: Path, obj: Any) -> None:
    """
    Atomically write JSON-serialized object to file.
    
    Args:
        path: Path object where JSON file should be written
        obj: Python object to serialize to JSON
    """
    # Create temporary file path by adding .tmp suffix
    tmp = path.with_suffix(path.suffix + ".tmp")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON to temporary file with proper formatting
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    
    # Atomic replace
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, obj: Any) -> None:
    """
    Atomically save PyTorch object to file.
    
    Args:
        path: Path object where PyTorch file should be saved
        obj: PyTorch object to save
    """
    # Create temporary file path by adding .tmp suffix
    tmp = path.with_suffix(path.suffix + ".tmp")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch object to temporary file
    torch.save(obj, tmp)
    
    # Atomic replace
    os.replace(tmp, path)


def _cpuize(x: Any) -> Any:
    """
    Recursively move torch tensors to CPU for portable checkpoints.
    
    This ensures checkpoints can be loaded on systems with different GPU setups.
    
    Args:
        x: Any Python object potentially containing torch tensors
        
    Returns:
        Same structure with all tensors moved to CPU and detached
    """
    # If input is a tensor, detach and move to CPU
    if torch.is_tensor(x):
        return x.detach().to("cpu")
    
    # If input is a dictionary, recursively process each value
    if isinstance(x, dict):
        return {k: _cpuize(v) for k, v in x.items()}
    
    # If input is a list or tuple, recursively process each element
    if isinstance(x, (list, tuple)):
        t = [_cpuize(v) for v in x]
        # Preserve tuple type if input was tuple
        return type(x)(t) if isinstance(x, tuple) else t
    
    # Return non-tensor objects unchanged
    return x


def _try_git_commit() -> Optional[str]:
    """
    Try to get current git commit hash.
    
    Returns:
        Git commit hash as string, or None if git command fails
    """
    try:
        # Run git rev-parse HEAD to get current commit
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        # Return None if any error occurs (git not installed, not in git repo, etc.)
        return None


def _infer_brain_kind(brain: torch.nn.Module) -> str:
    """
    Determine the type of brain neural network.
    
    Args:
        brain: PyTorch module representing the brain
        
    Returns:
        String identifier for the brain type
    """
    # Import brain classes locally to avoid import cycles at module import time
    from agent.tron_brain import TronBrain
    from agent.mirror_brain import MirrorBrain
    from agent.transformer_brain import TransformerBrain

    # Check brain type and return appropriate identifier
    if isinstance(brain, TronBrain):
        return "tron"
    if isinstance(brain, MirrorBrain):
        return "mirror"
    if isinstance(brain, TransformerBrain):
        return "transformer"
    
    # Fallback: store class name; load will error loudly if unknown
    return brain.__class__.__name__


def _make_brain(kind: str, device: torch.device) -> torch.nn.Module:
    """
    Create a brain instance of the specified kind.
    
    Args:
        kind: String identifier for brain type
        device: PyTorch device to place the brain on
        
    Returns:
        Initialized brain module on the specified device
        
    Raises:
        CheckpointError: If brain kind is unknown
    """
    # Import brain classes locally to avoid import cycles
    from agent.tron_brain import TronBrain
    from agent.mirror_brain import MirrorBrain
    from agent.transformer_brain import TransformerBrain

    # Get observation and action dimensions from config
    obs_dim = int(getattr(config, "OBS_DIM"))
    act_dim = int(getattr(config, "NUM_ACTIONS"))

    # Create appropriate brain based on kind
    if kind == "tron":
        return TronBrain(obs_dim, act_dim).to(device)
    if kind == "mirror":
        return MirrorBrain(obs_dim, act_dim).to(device)
    if kind == "transformer":
        return TransformerBrain(obs_dim, act_dim).to(device)
    
    # Raise error for unknown brain type
    raise CheckpointError(f"Unknown brain kind in checkpoint: {kind}")


def _get_rng_state() -> Dict[str, Any]:
    """
    Capture current random number generator states from all sources.
    
    Returns:
        Dictionary containing RNG states for Python random, NumPy, and PyTorch (CPU and CUDA)
    """
    state: Dict[str, Any] = {
        "python_random": random.getstate(),           # Python's random module state
        "numpy_random": np.random.get_state(),        # NumPy's random state
        "torch_cpu": torch.random.get_rng_state(),    # PyTorch CPU RNG state
        "torch_cuda": None,                           # Initialize CUDA state as None
    }
    
    # Try to capture CUDA RNG states if CUDA is available
    if torch.cuda.is_available():
        try:
            # Get RNG state for all CUDA devices
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            # If capturing fails, leave as None
            state["torch_cuda"] = None
    
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore random number generator states.
    
    Important: This should be called LAST in resume flow to avoid constructor
    RNG consumption changing post-resume randomness.
    
    Args:
        state: Dictionary containing RNG states from _get_rng_state()
    """
    # Restore Python random state
    random.setstate(state["python_random"])
    
    # Restore NumPy random state
    np.random.set_state(state["numpy_random"])
    
    # Restore PyTorch CPU RNG state
    torch.random.set_rng_state(state["torch_cpu"])
    
    # Restore PyTorch CUDA RNG states if available
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            # If device count differs, keep best-effort behavior (just skip)
            pass


def _extract_stats(stats: Any) -> Dict[str, Any]:
    """
    Extract serializable state from stats object.
    
    Args:
        stats: Statistics object (typically a dataclass)
        
    Returns:
        Dictionary representation of stats
        
    Raises:
        CheckpointError: If stats object is not supported
    """
    # If stats is a dataclass, convert to dictionary
    if is_dataclass(stats):
        return asdict(stats)
    
    # If stats has __dict__ attribute, use it for serialization
    if hasattr(stats, "__dict__"):
        # Make a shallow copy; nested dataclasses get handled above
        return dict(stats.__dict__)
    
    # Raise error for unsupported stats objects
    raise CheckpointError("Unsupported stats object for checkpointing")


def _apply_stats(stats_obj: Any, payload: Dict[str, Any]) -> None:
    """
    Apply serialized stats to a stats object.
    
    Args:
        stats_obj: Target stats object to update
        payload: Dictionary of stats values to apply
    """
    # Iterate through payload items and set attributes best-effort
    for k, v in payload.items():
        try:
            setattr(stats_obj, k, v)
        except Exception:
            # Silently skip attributes that can't be set
            pass


def resolve_checkpoint_path(p: str) -> Tuple[Path, Path]:
    """
    Resolve a checkpoint path to directory and file.
    
    Accepts either:
      - a directory containing DONE + checkpoint.pt
      - a direct path to checkpoint.pt
    
    Args:
        p: Path string to checkpoint directory or file
        
    Returns:
        Tuple of (checkpoint_directory, checkpoint_pt_path)
        
    Raises:
        CheckpointError: If path is not found or invalid
    """
    # Expand user directory (~) and convert to Path
    path = Path(p).expanduser()
    
    # Case 1: Direct path to checkpoint.pt file
    if path.is_file() and path.name == "checkpoint.pt":
        ckpt_dir = path.parent
        return ckpt_dir, path
    
    # Case 2: Directory containing checkpoint
    if path.is_dir():
        return path, path / "checkpoint.pt"
    
    # Invalid path
    raise CheckpointError(f"Checkpoint path not found: {p}")


class CheckpointManager:
    """Manages checkpoint creation, saving, and loading for the simulation."""
    
    # Version identifier for checkpoint format compatibility
    checkpoint_version: int = 1

    def __init__(self, run_dir: Path) -> None:
        """
        Initialize checkpoint manager.
        
        Args:
            run_dir: Base directory for the run (checkpoints will be in run_dir/checkpoints)
        """
        self.run_dir = Path(run_dir)
        self.ckpt_base = self.run_dir / "checkpoints"

    def save_atomic(
        self,
        *,
        engine: Any,
        registry: Any,
        stats: Any,
        viewer_state: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> Path:
        """
        Atomically save a checkpoint.
        
        This method creates a new checkpoint directory with timestamp, writes
        all data atomically, and updates the latest pointer.
        
        Args:
            engine: Simulation engine object
            registry: Agent registry object
            stats: Statistics object
            viewer_state: Optional viewer state dictionary
            notes: Optional notes about this checkpoint
            
        Returns:
            Path to the created checkpoint directory
            
        Raises:
            CheckpointError: If checkpoint directory already exists
        """
        # Get current tick from stats
        tick = int(getattr(stats, "tick"))
        
        # Generate timestamp for this checkpoint
        stamp = _now_stamp()
        
        # Create checkpoint directory name
        name = f"ckpt_t{tick}_{stamp}"
        final_dir = self.ckpt_base / name
        tmp_dir = self.ckpt_base / (name + "__tmp")

        # Clean stale tmp directory (crash recovery)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        
        # Create temporary directory
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # --- Build checkpoint dict (store tensors on CPU for portability) ---
        
        # Extract grid and zones from engine
        grid = getattr(engine, "grid")
        zones = getattr(engine, "zones", None)

        # Extract brains per slot from registry
        brains_payload = []
        for b in getattr(registry, "brains"):
            if b is None:
                # Empty slot
                brains_payload.append(None)
                continue
            
            # Get brain type and state dict (moved to CPU)
            kind = _infer_brain_kind(b)
            sd = _cpuize(b.state_dict())
            brains_payload.append({"kind": kind, "state_dict": sd})

        # Main checkpoint dictionary
        ckpt: Dict[str, Any] = {
            "checkpoint_version": self.checkpoint_version,
            "meta": {
                "tick": tick,
                "timestamp": stamp,
                "notes": notes,
                "saved_device": str(getattr(grid, "device", "unknown")),
                "runtime_device": str(getattr(config, "TORCH_DEVICE", "unknown")),
                "git_commit": _try_git_commit(),
            },
            "world": {
                "grid": _cpuize(grid),
                "zones": None if zones is None else {
                    "heal_mask": _cpuize(getattr(zones, "heal_mask")),
                    "cp_masks": _cpuize(getattr(zones, "cp_masks")),
                },
            },
            "registry": {
                "agent_data": _cpuize(getattr(registry, "agent_data")),
                "agent_uids": _cpuize(getattr(registry, "agent_uids", None)),
                "generations": list(getattr(registry, "generations")),
                "next_agent_id": int(getattr(registry, "_next_agent_id")),
                "brains": brains_payload,
            },
            "engine": {
                "agent_scores": dict(getattr(engine, "agent_scores", {})),
                "respawn_controller": self._extract_respawn_state(getattr(engine, "respawner", None)),
            },
            "ppo": self._extract_ppo_state(engine),
            "stats": _extract_stats(stats),
            "viewer": viewer_state or {},
            "rng": _get_rng_state(),
        }

        # --- Write files atomically inside tmp_dir ---
        ckpt_pt = tmp_dir / "checkpoint.pt"
        manifest_json = tmp_dir / "manifest.json"

        # Save checkpoint file
        _atomic_torch_save(ckpt_pt, ckpt)
        
        # Create and save manifest
        manifest = {
            "version": self.checkpoint_version,
            "tick": tick,
            "timestamp": stamp,
            "notes": notes,
            "git_commit": ckpt["meta"]["git_commit"],
            "file_list": ["manifest.json", "checkpoint.pt", "DONE"],
        }
        _atomic_json_dump(manifest_json, manifest)
        
        # Write DONE marker (indicates checkpoint is complete)
        _atomic_write_text(tmp_dir / "DONE", "OK\n")

        # Rename tmp folder -> final folder (atomic on same filesystem)
        if final_dir.exists():
            raise CheckpointError(f"Checkpoint dir already exists: {final_dir}")
        os.replace(tmp_dir, final_dir)

        # Update latest pointer (points to most recent checkpoint)
        _atomic_write_text(self.ckpt_base / "latest.txt", str(final_dir.name) + "\n")
        
        return final_dir

    @staticmethod
    def load(path: str, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
        """
        Load a checkpoint from disk.
        
        Args:
            path: Path to checkpoint directory or checkpoint.pt file
            map_location: Device to map tensors to when loading
            
        Returns:
            Loaded checkpoint dictionary
            
        Raises:
            CheckpointError: If checkpoint is invalid or missing DONE marker
        """
        # Resolve path to directory and checkpoint file
        ckpt_dir, ckpt_pt = resolve_checkpoint_path(path)
        
        # Check for DONE marker (ensures checkpoint is complete)
        done = ckpt_dir / "DONE"
        if not done.exists():
            raise CheckpointError(f"Refusing to load checkpoint without DONE marker: {ckpt_dir}")
        
        # PyTorch 2.6+ changed `torch.load` default to `weights_only=True`,
        # which blocks unpickling custom Python objects (non-tensor state).
        # Our simulation checkpoints intentionally store full runtime state
        # and are produced by *this* codebase, so we load them in full.
        try:
            obj = torch.load(ckpt_pt, map_location=map_location, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have the `weights_only` kwarg.
            obj = torch.load(ckpt_pt, map_location=map_location)
        
        # Validate checkpoint format
        if not isinstance(obj, dict) or "checkpoint_version" not in obj:
            raise CheckpointError(f"Invalid checkpoint format: {ckpt_pt}")
        
        return obj

    @staticmethod
    def apply_loaded_checkpoint(
        ckpt: Dict[str, Any],
        *,
        engine: Any,
        registry: Any,
        stats: Any,
        device: torch.device,
    ) -> None:
        """
        Apply a loaded checkpoint to simulation objects.
        
        Args:
            ckpt: Loaded checkpoint dictionary
            engine: Simulation engine object to update
            registry: Agent registry object to update
            stats: Statistics object to update
            device: PyTorch device to place restored tensors on
            
        Raises:
            CheckpointError: If PPO state exists but engine doesn't have PPO
        """
        # --- Restore world/registry ---
        reg = ckpt["registry"]
        
        # Restore agent data (move to specified device)
        registry.agent_data = reg["agent_data"].to(device)
        # Restore permanent unique IDs (int64). Older checkpoints may not have this.
        if reg.get("agent_uids") is not None:
            registry.agent_uids = reg["agent_uids"].to(device)
        else:
            # Best-effort reconstruction (may be lossy if older runs overflowed float16 IDs).
            try:
                from engine.agent_registry import COL_AGENT_ID
                legacy = registry.agent_data[:, COL_AGENT_ID]
                legacy = torch.where(torch.isfinite(legacy), legacy, torch.full_like(legacy, -1.0))
                registry.agent_uids = legacy.to(torch.int64)
            except Exception:
                pass
        # Restore generations
        registry.generations = list(reg["generations"])
        
        # Restore next agent ID
        registry._next_agent_id = int(reg["next_agent_id"])

        # Restore brains per slot
        brains_payload = reg["brains"]
        for i, payload in enumerate(brains_payload):
            if payload is None:
                # Empty slot
                registry.brains[i] = None
                continue
            
            # Create brain of appropriate kind and load state
            b = _make_brain(payload["kind"], device)
            b.load_state_dict(payload["state_dict"])
            registry.brains[i] = b

        # --- Restore engine internal state ---
        eng = ckpt.get("engine", {})
        
        # Restore agent scores
        if hasattr(engine, "agent_scores"):
            engine.agent_scores.clear()
            engine.agent_scores.update(eng.get("agent_scores", {}))

        # Restore respawn controller state
        CheckpointManager._apply_respawn_state(
            getattr(engine, "respawner", None), 
            eng.get("respawn_controller")
        )

        # Restore statistics
        _apply_stats(stats, ckpt.get("stats", {}))

        # Restore PPO runtime state if present
        ppo = ckpt.get("ppo", {"enabled": False})
        if ppo.get("enabled", False):
            if not hasattr(engine, "_ppo") or engine._ppo is None:
                raise CheckpointError("Checkpoint has PPO state but engine._ppo is None (config mismatch)")
            engine._ppo.load_checkpoint_state(ppo["state"], registry=registry, device=device)

        # Restore RNG LAST (critical for reproducible simulation)
        _set_rng_state(ckpt["rng"])

    @staticmethod
    def zones_from_checkpoint(world_payload: Dict[str, Any], *, device: torch.device) -> Optional[Any]:
        """
        Reconstruct Zones object from checkpoint world payload.
        
        Args:
            world_payload: World section from checkpoint
            device: PyTorch device to place zone tensors on
            
        Returns:
            Reconstructed Zones object or None if no zones
        """
        z = world_payload.get("zones", None)
        if z is None:
            return None
        
        # Import lazily to avoid import cycles
        from engine.mapgen import Zones
        
        # Move zone tensors to specified device
        heal_mask = z["heal_mask"].to(device)
        cp_masks = [t.to(device) for t in z["cp_masks"]]
        
        # Create new Zones object
        return Zones(heal_mask=heal_mask, cp_masks=cp_masks)

    @staticmethod
    def _extract_respawn_state(respawner: Any) -> Dict[str, Any]:
        """
        Extract serializable state from respawn controller.
        
        Args:
            respawner: RespawnController object
            
        Returns:
            Dictionary of respawner state
        """
        if respawner is None:
            return {}
        
        # Keys to extract from respawner (private ints)
        keys = ["_cooldown_red_until", "_cooldown_blue_until", "_last_period_tick"]
        out: Dict[str, Any] = {}
        
        # Extract each key if present
        for k in keys:
            if hasattr(respawner, k):
                out[k] = int(getattr(respawner, k))
        
        return out

    @staticmethod
    def _apply_respawn_state(respawner: Any, payload: Optional[Dict[str, Any]]) -> None:
        """
        Apply serialized state to respawn controller.
        
        Args:
            respawner: RespawnController object to update
            payload: Dictionary of respawner state
        """
        if respawner is None or not payload:
            return
        
        # Set each attribute if present
        for k, v in payload.items():
            if hasattr(respawner, k):
                setattr(respawner, k, int(v))

    @staticmethod
    def _extract_ppo_state(engine: Any) -> Dict[str, Any]:
        """
        Extract PPO trainer state from engine.
        
        Args:
            engine: Simulation engine object
            
        Returns:
            Dictionary with PPO enabled flag and state
        """
        ppo = getattr(engine, "_ppo", None)
        if ppo is None:
            return {"enabled": False, "state": {}}
        
        return {"enabled": True, "state": ppo.get_checkpoint_state()}
        # Telemetry safety: flush buffered telemetry chunks BEFORE freezing the checkpoint.
        # This does not affect determinism; it is best-effort and failure-safe.
        telemetry = getattr(engine, "telemetry", None)
        if telemetry is not None:
            try:
                telemetry.flush(reason="checkpoint_save")
            except Exception:
                # Never let telemetry break checkpointing.
                pass