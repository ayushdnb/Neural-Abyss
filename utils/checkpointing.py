from __future__ import annotations

"""
Checkpointing module (save/load/resume) for a PyTorch + NumPy simulation.

This file is designed to:
1) Save the *entire* simulation runtime state to disk in a crash-safe manner.
2) Restore it later so the run can continue deterministically (as much as possible).

Key ideas used here (high-level):
- **Atomic writes**: write to a temporary file first, then use `os.replace(...)`
  which is (typically) atomic on the same filesystem. This prevents partially
  written/corrupted files if the process crashes mid-write.
- **Portability**: tensors are moved to **CPU** before writing, so checkpoints can
  be loaded on machines without the same GPU setup.
- **Reproducibility**: RNG states from Python, NumPy, and PyTorch (CPU + CUDA) are
  saved and restored.
- **Versioning**: `checkpoint_version` supports forward compatibility if you
  evolve the checkpoint format later.
"""

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


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class CheckpointError(RuntimeError):
    """
    Custom exception for checkpoint-related errors.

    Why create a custom exception type?
    - It allows calling code to catch *checkpoint problems specifically*,
      without accidentally catching unrelated RuntimeError exceptions.
    """
    pass


# -----------------------------------------------------------------------------
# Small utilities (timestamps, atomic file operations)
# -----------------------------------------------------------------------------

def _now_stamp() -> str:
    """
    Return current timestamp formatted as YYYY-MM-DD_HH-MM-SS.

    This is used in checkpoint folder naming, giving:
    - human readability
    - chronological sorting by name
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write text to a file.

    Atomic write pattern:
    1) Write content to a temporary file in the same directory.
    2) Flush and fsync so bytes reach disk (best effort).
    3) Replace the target path with the temp file via `os.replace`
       (atomic on most OS/filesystems when staying on same filesystem).

    Why this matters:
    - If the program crashes mid-write, you won't end up with a half-written file.
    - Readers will see either the old complete file or the new complete file.

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
        # fsync asks the OS to push buffered data to disk.
        # This reduces risk of data loss on sudden power loss, but can't guarantee it.
        os.fsync(f.fileno())

    # Atomic replace - this is atomic on most filesystems
    os.replace(tmp, path)


def _atomic_json_dump(path: Path, obj: Any) -> None:
    """
    Atomically write JSON-serialized object to file.

    Same atomic pattern as `_atomic_write_text`, but JSON-encodes `obj`.

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
        # indent=2 makes diffs readable in git
        # sort_keys=True makes output deterministic (stable ordering)
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, obj: Any) -> None:
    """
    Atomically save PyTorch object to file.

    `torch.save(...)` can write large binary blobs. We use temp + replace
    to avoid corrupt checkpoint files if interrupted.

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


# -----------------------------------------------------------------------------
# Tensor/device portability helpers
# -----------------------------------------------------------------------------

def _cpuize(x: Any) -> Any:
    """
    Recursively move torch tensors to CPU for portable checkpoints.

    Key PyTorch concepts:
    - A tensor can live on CPU or GPU (CUDA).
    - Checkpoints that store CUDA tensors may fail to load if:
      - CUDA isn't available
      - device IDs differ
      - driver/runtime differs
    - So we store tensors on CPU by detaching them and moving to "cpu".

    Terminology:
    - detach(): removes the tensor from autograd graph (no gradient history).
      This avoids serializing gradient graph metadata and makes the tensor a
      pure value snapshot.

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


# -----------------------------------------------------------------------------
# Git metadata helper (optional)
# -----------------------------------------------------------------------------

def _try_git_commit() -> Optional[str]:
    """
    Try to get current git commit hash.

    This is useful for:
    - tracking exactly which code produced a checkpoint
    - debugging mismatches between checkpoint data and code

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


# -----------------------------------------------------------------------------
# Brain (model) type inference + reconstruction
# -----------------------------------------------------------------------------

def _infer_brain_kind(brain: torch.nn.Module) -> str:
    """
    Determine the type of brain neural network.

    Why do this?
    - Your registry may contain multiple possible model architectures.
    - When saving, we store:
        - which architecture it was ("kind")
        - its state_dict (weights)
    - When loading, we recreate that architecture and load weights into it.

    Args:
        brain: PyTorch module representing the brain

    Returns:
        String identifier for the brain type
    """
    from agent.mlp_brain import brain_kind_from_module

    kind = brain_kind_from_module(brain)
    if kind:
        return kind
    return brain.__class__.__name__


def _make_brain(kind: str, device: torch.device) -> torch.nn.Module:
    """
    Create a brain instance of the specified kind.

    This is the mirror of `_infer_brain_kind`.

    Args:
        kind: String identifier for brain type
        device: PyTorch device to place the brain on

    Returns:
        Initialized brain module on the specified device

    Raises:
        CheckpointError: If brain kind is unknown
    """
    from agent.mlp_brain import create_mlp_brain

    # Get observation and action dimensions from config
    obs_dim = int(getattr(config, "OBS_DIM"))
    act_dim = int(getattr(config, "NUM_ACTIONS"))

    if kind in {
        "whispering_abyss",
        "veil_of_echoes",
        "cathedral_of_ash",
        "dreamer_in_black_fog",
        "obsidian_pulse",
    }:

        return create_mlp_brain(kind, obs_dim, act_dim).to(device)  

    raise CheckpointError(f"Unknown brain kind in checkpoint: {kind}")


# -----------------------------------------------------------------------------
# RNG state capture/restore (determinism)
# -----------------------------------------------------------------------------

def _get_rng_state() -> Dict[str, Any]:
    """
    Capture current random number generator states from all sources.

    Why capture RNG state?
    - Deterministic resume: If you restore RNG states, random decisions after
      resume can match what would have happened without interruption.

    Sources of randomness here:
    - Python's `random` module
    - NumPy RNG
    - PyTorch CPU RNG
    - PyTorch CUDA RNG (for each GPU) if available

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

    Explanation:
    - If you set RNG state early, then creating objects / initializing tensors
      might consume random numbers and shift the sequence.
    - So you restore everything else first, then finally restore RNG.

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


# -----------------------------------------------------------------------------
# Stats extraction/apply (generic)
# -----------------------------------------------------------------------------

def _extract_stats(stats: Any) -> Dict[str, Any]:
    """
    Extract serializable state from stats object.

    Supports:
    - dataclasses (preferred): `asdict(stats)`
    - generic objects with __dict__

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

    Best-effort behavior:
    - sets known attributes
    - skips attributes that cannot be set (read-only, property, etc.)

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


# -----------------------------------------------------------------------------
# Path resolution for checkpoint inputs
# -----------------------------------------------------------------------------

def resolve_checkpoint_path(p: str) -> Tuple[Path, Path]:
    """
    Resolve a checkpoint path to directory and file.

    Accepts either:
      - a directory containing DONE + checkpoint.pt
      - a direct path to checkpoint.pt

    Also supports a "checkpoints root dir" UX via latest.txt:
    - If user passes ".../checkpoints", and latest.txt exists, then we interpret
      it as pointer to latest checkpoint subdir.

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

    # Case 2: Directory containing checkpoint OR directory containing latest.txt pointer
    if path.is_dir():
        # If this directory directly contains checkpoint.pt, treat as checkpoint dir (backward-compatible).
        if (path / "checkpoint.pt").exists():
            return path, path / "checkpoint.pt"

        # Optional UX: if user passes checkpoints root dir, resolve via latest.txt
        latest = path / "latest.txt"
        if latest.exists():
            name = latest.read_text(encoding="utf-8", errors="ignore").strip()
            if name:
                cand = path / name
                if cand.is_dir():
                    return cand, cand / "checkpoint.pt"

        # Fallback: preserve previous behavior (may fail later if invalid)
        return path, path / "checkpoint.pt"

    # Invalid path
    raise CheckpointError(f"Checkpoint path not found: {p}")


# -----------------------------------------------------------------------------
# Main class: CheckpointManager
# -----------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages checkpoint creation, saving, and loading for the simulation.

    Conceptually:
    - A checkpoint is stored as a directory containing:
        - checkpoint.pt      (torch-saved dict of state)
        - manifest.json      (human-readable summary metadata)
        - DONE              (marker file that signals checkpoint is complete)
        - PINNED (optional) (marker file for retention policy)
    - The root "checkpoints" directory also contains:
        - latest.txt         (points to the latest checkpoint dir name)
    """

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
        pinned: bool = False,
        pin_tag: str = "",
    ) -> Path:
        """
        Atomically save a checkpoint.

        This method creates a new checkpoint directory with timestamp, writes
        all data atomically, and updates the latest pointer.

        High-level algorithm:
        1) Create a temporary checkpoint directory: <name>__tmp
        2) Write checkpoint.pt and manifest.json into it (atomic file writes)
        3) Write DONE marker
        4) Rename temp dir to final dir using os.replace (atomic directory swap)
        5) Update latest.txt

        Args:
            engine: Simulation engine object
            registry: Agent registry object
            stats: Statistics object
            viewer_state: Optional viewer state dictionary
            notes: Optional notes about this checkpoint
            pinned: Whether this checkpoint should be marked as pinned (protected from pruning)
            pin_tag: Optional tag for pinned checkpoint

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
            "pinned": bool(pinned),
            "pin_tag": str(pin_tag) if pinned else "",
        }
        _atomic_json_dump(manifest_json, manifest)

        # Optional pinned marker (for retention policy). Safe: loader ignores extra files.
        if pinned:
            _atomic_write_text(tmp_dir / "PINNED", (str(pin_tag) + "\n") if pin_tag else "manual\n")
            manifest["file_list"].append("PINNED")

        # Write DONE marker (indicates checkpoint is complete)
        _atomic_write_text(tmp_dir / "DONE", "OK\n")

        # ---- finalize: rename tmp -> final (atomic on same filesystem) ----
        self.ckpt_base.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise CheckpointError(f"Checkpoint dir already exists: {final_dir}")
        os.replace(tmp_dir, final_dir)

        # Update latest pointer
        _atomic_write_text(self.ckpt_base / "latest.txt", final_dir.name + "\n")

        return final_dir

    def _is_complete_dir(self, d: Path) -> bool:
        """
        Check if a directory contains a complete checkpoint.

        A checkpoint is considered complete only if:
        - DONE exists (written last)
        - checkpoint.pt exists

        Args:
            d: Directory path to check

        Returns:
            True if DONE and checkpoint.pt exist, else False
        """
        return (d / "DONE").exists() and (d / "checkpoint.pt").exists()

    def _is_pinned_dir(self, d: Path) -> bool:
        """
        Check if a checkpoint directory is pinned (protected from pruning).

        Args:
            d: Directory path to check

        Returns:
            True if PINNED file exists, else False
        """
        return (d / "PINNED").exists()

    def prune_keep_last_n(self, keep_last_n: int) -> None:
        """
        Keep only the newest N completed checkpoints, never deleting pinned.

        Practical reasons:
        - Checkpoints can be huge (tensors, model weights).
        - Without pruning, disk usage can explode during long runs.

        Rules:
        - Only checkpoint dirs matching "ckpt_t*" are considered.
        - tmp dirs and incomplete dirs are ignored.
        - pinned checkpoints are never deleted.
        - The checkpoint referenced by latest.txt is never deleted.

        Args:
            keep_last_n: Number of latest checkpoints to retain (non-pinned)
        """
        if keep_last_n is None or int(keep_last_n) <= 0:
            return
        self.ckpt_base.mkdir(parents=True, exist_ok=True)

        # Gather completed checkpoint dirs (ignore tmp dirs)
        dirs = []
        for p in self.ckpt_base.iterdir():
            if not p.is_dir():
                continue
            if p.name.endswith("__tmp") or p.name.startswith(".tmp"):
                continue
            if not p.name.startswith("ckpt_t"):
                continue
            if self._is_complete_dir(p):
                dirs.append(p)

        if len(dirs) <= keep_last_n:
            return

        # Sort by name (tick is embedded; names are stable)
        dirs.sort(key=lambda x: x.name)
        keep = set(dirs[-keep_last_n:])

        # Do not delete the one referenced by latest.txt (belt+suspenders)
        latest = self.ckpt_base / "latest.txt"
        if latest.exists():
            name = latest.read_text(encoding="utf-8", errors="ignore").strip()
            if name:
                keep.add(self.ckpt_base / name)

        for d in dirs:
            if d in keep:
                continue
            if self._is_pinned_dir(d):
                continue
            # extra safety: only delete completed checkpoints
            if not self._is_complete_dir(d):
                continue
            shutil.rmtree(d, ignore_errors=True)

    def maybe_save_periodic(
        self,
        *,
        engine: Any,
        registry: Any,
        stats: Any,
        every_ticks: int,
        keep_last_n: int,
    ) -> Optional[Path]:
        """
        Save a checkpoint periodically if the tick matches the interval.

        Typical use:
        - called once per simulation tick (or once per main loop iteration)
        - saves every N ticks
        - prunes after saving

        Args:
            engine: Simulation engine object
            registry: Agent registry object
            stats: Statistics object
            every_ticks: Save checkpoint every this many ticks (if >0)
            keep_last_n: Number of latest checkpoints to retain after saving

        Returns:
            Path to saved checkpoint directory, or None if not saved
        """
        if every_ticks is None or int(every_ticks) <= 0:
            return None
        tick = int(getattr(stats, "tick", 0))
        if tick <= 0 or (tick % int(every_ticks)) != 0:
            return None
        last = getattr(self, "_last_periodic_tick", None)
        if last == tick:
            return None
        out = self.save_atomic(engine=engine, registry=registry, stats=stats, notes=f"auto_every_{every_ticks}")
        setattr(self, "_last_periodic_tick", tick)
        self.prune_keep_last_n(keep_last_n)
        return out

    def maybe_save_trigger_file(
        self,
        *,
        trigger_path: Path,
        engine: Any,
        registry: Any,
        stats: Any,
        default_pin: bool,
        pin_tag: str,
        keep_last_n: int,
    ) -> Optional[Path]:
        """
        Save a checkpoint if a trigger file exists, then delete the trigger.

        This enables an external "manual save" mechanism:
        - user creates a file (like `SAVE_NOW.txt`)
        - the simulation sees it and saves a checkpoint
        - then it deletes the trigger file

        Pinned behavior:
        - If default_pin is True OR file contains words "pin" or "keep",
          the checkpoint becomes pinned.

        Args:
            trigger_path: Path to trigger file (if it exists, save)
            engine: Simulation engine object
            registry: Agent registry object
            stats: Statistics object
            default_pin: Default pinned value if trigger content doesn't specify
            pin_tag: Tag to use if pinned
            keep_last_n: Number of latest checkpoints to retain after saving

        Returns:
            Path to saved checkpoint directory, or None if trigger didn't exist
        """
        if trigger_path is None or not trigger_path.exists():
            return None
        raw = ""
        try:
            raw = trigger_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            raw = ""
        lower = raw.lower()
        pinned = bool(default_pin) or ("pin" in lower) or ("keep" in lower)
        notes = raw if raw else "manual_trigger"
        out = self.save_atomic(
            engine=engine,
            registry=registry,
            stats=stats,
            notes=notes,
            pinned=pinned,
            pin_tag=pin_tag if pinned else "",
        )
        # Delete trigger only after success
        try:
            trigger_path.unlink()
        except Exception:
            pass
        self.prune_keep_last_n(keep_last_n)
        return out

    @staticmethod
    def load(path: str, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
        """
        Load a checkpoint from disk.

        Safety rules:
        - Refuse to load if DONE marker missing (likely incomplete checkpoint).

        PyTorch note:
        - Newer PyTorch versions changed defaults around `weights_only`.
          This code explicitly loads full checkpoint payload (weights_only=False)
          because you store runtime state (not just tensors).

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

        This mutates in-place:
        - registry (agent data, uids, generations, brains, etc.)
        - engine internal runtime state
        - stats object
        - PPO trainer state (if enabled in checkpoint)
        - RNG states (restored LAST)

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
        from engine.agent_registry import NUM_COLS

        ckpt_agent_data = reg["agent_data"]
        if int(ckpt_agent_data.dim()) != 2:
            raise CheckpointError(
                f"checkpoint registry.agent_data must be rank-2, got shape={tuple(ckpt_agent_data.shape)}"
            )
        if int(ckpt_agent_data.shape[0]) != int(registry.capacity):
            raise CheckpointError(
                f"checkpoint capacity mismatch: ckpt_slots={int(ckpt_agent_data.shape[0])} registry.capacity={int(registry.capacity)}"
            )
        if int(ckpt_agent_data.shape[1]) != int(NUM_COLS):
            raise CheckpointError(
                f"checkpoint NUM_COLS mismatch: ckpt_cols={int(ckpt_agent_data.shape[1])} expected={int(NUM_COLS)}"
            )
        if len(reg["brains"]) != int(registry.capacity):
            raise CheckpointError(
                f"checkpoint brains length mismatch: got={len(reg['brains'])} expected={int(registry.capacity)}"
            )
        if len(reg["generations"]) != int(registry.capacity):
            raise CheckpointError(
                f"checkpoint generations length mismatch: got={len(reg['generations'])} expected={int(registry.capacity)}"
            )
        if reg.get("agent_uids") is not None and int(reg["agent_uids"].numel()) != int(registry.capacity):
            raise CheckpointError(
                f"checkpoint agent_uids length mismatch: got={int(reg['agent_uids'].numel())} expected={int(registry.capacity)}"
            )

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
        generations = [int(g) for g in list(reg["generations"])]
        if any(int(g) < 0 for g in generations):
            raise CheckpointError("checkpoint contains negative generation values")
        registry.generations = generations

        # Restore next agent ID
        registry._next_agent_id = int(reg["next_agent_id"])

        # Restore brains per slot
        brains_payload = reg["brains"]
        for i, payload in enumerate(brains_payload):
            if payload is None:
                # Empty slot
                registry.set_brain(i, None)
                continue

            # Create brain of appropriate kind and load state
            b = _make_brain(payload["kind"], device)
            b.load_state_dict(payload["state_dict"])
            registry.set_brain(i, b)

        # Rebuild architecture metadata from the restored brains so checkpoint
        # loads remain backward-compatible without storing extra bucket state.
        if hasattr(registry, "rebuild_arch_metadata"):
            registry.rebuild_arch_metadata()
        if hasattr(registry, "agent_uids"):
            valid_uid_mask = (registry.agent_uids >= 0)
            if bool(valid_uid_mask.any().item()):
                max_uid = int(registry.agent_uids[valid_uid_mask].max().item())
                if int(registry._next_agent_id) <= max_uid:
                    raise CheckpointError(
                        f"checkpoint next_agent_id is stale: next_agent_id={int(registry._next_agent_id)} max_uid={max_uid}"
                    )
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

        Why this exists:
        - Some code paths want to reconstruct `engine.zones` from checkpoint,
          but you might not want to restore the whole engine yet.

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

        We store only a small subset of internal fields, typically counters/ticks,
        because those determine future respawn logic.

        Args:
            respawner: RespawnController object

        Returns:
            Dictionary of respawner state
        """
        if respawner is None:
            return {}

        # Keys to extract from respawner (private ints)
        keys = [
            "_cooldown_red_until",
            "_cooldown_blue_until",
            "_last_period_tick",
            "_rare_mutation_pending_ticket",
            "_rare_mutation_last_window_idx",
            "_legacy_respawn_counter",
        ]
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

    # REPLACE THIS FUNCTION (exact name/signature) =================================
    @staticmethod
    def _extract_ppo_state(engine: Any) -> Dict[str, Any]:
        """
        Extract PPO trainer state from engine.

        NOTE (bug fix):
        The previous version returned before flushing telemetry, making that code unreachable.
        """
        ppo = getattr(engine, "_ppo", None)
        if ppo is None:
            return {"enabled": False, "state": {}}

        state = ppo.get_checkpoint_state()

        # Telemetry safety: best-effort flush BEFORE freezing the checkpoint.
        # This does not affect determinism; it just improves durability of logs.
        telemetry = getattr(engine, "telemetry", None)
        if telemetry is not None and hasattr(telemetry, "flush"):
            try:
                telemetry.flush(reason="checkpoint_save")
            except Exception:
                pass

        return {"enabled": True, "state": state}
    # =============================================================================