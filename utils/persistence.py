"""
A **Windows-friendly**, **multiprocessing-based** background writer that records simulation/training
results to disk *without* stalling the main simulation loop.

Why this file exists (high-level)
---------------------------------
In simulations, reinforcement learning training loops, or real-time game-like engines, the main loop
often runs at high frequency (e.g., tens of ticks per second or more). Writing to disk (CSV/JSON)
can be unexpectedly expensive because:

1) Disk I/O is slow relative to CPU/GPU compute.
2) Python file writes can block.
3) CSV writers and frequent flushes can introduce latency spikes.
4) If you log a lot, the main loop may jitter (uneven tick time), causing unstable training or a laggy sim.

So the common engineering solution is:
- Put disk I/O into a **separate process**
- Send "write requests" (messages) from the main process to the writer process via a **Queue**
- The main process continues running almost uninterrupted.

This is exactly what this module implements.

Key architecture
----------------
Main Process (simulation/training loop)
    |
    |  (Queue message: "write this tick row")
    V
Writer Process (background)
    |
    |  writes CSV/JSON to disk
    V
Filesystem (results/sim_YYYY-MM-DD_HH-MM-SS/...)

Important multiprocessing context (esp. on Windows)
---------------------------------------------------
On Windows, `multiprocessing` uses "spawn" (new Python process starts fresh) rather than "fork".
That means:
- The child process does NOT automatically inherit open file handles safely.
- You must open files inside the child process, not in the parent.
- Functions and objects used as process targets must be importable at top level.

This code follows those rules.

Performance philosophy
----------------------
- The main loop should never block on logging.
- If the queue is full, it's better to drop logs than to freeze the simulation.
  (This is a conscious choice: "best-effort" telemetry.)

No math-heavy content here
--------------------------
There is no advanced math in this module. The core concepts are systems/programming concepts:
- Producer/consumer queue
- Inter-process communication (IPC)
- Event/message-driven I/O
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, Any, Optional, List

import os
import json
import csv
import datetime
import queue  # IMPORTANT: this is the *standard library* "queue" module (for queue.Empty/queue.Full)


# =============================================================================
# Message Protocol (Typed "commands" sent to the writer process)
# =============================================================================
#
# We define a small set of message types. The main process will put these message
# objects into a multiprocessing.Queue. The writer process will read them and act.
#
# This pattern is often called:
# - "message passing"
# - "command pattern"
# - "actor model" (loosely)
#
# The purpose is to avoid sending raw tuples like ("write_tick", row) everywhere,
# and instead have explicit, typed message classes.
# This improves readability, correctness, and makes refactors safer.


@dataclass
class _MsgInit:
    """
    Initialization message.

    Sent exactly once at startup so the writer process knows:
    - where to create the run directory
    - what config to save to config.json

    Fields
    ------
    run_dir:
        Directory where outputs will be written.
    config_obj:
        Serializable configuration dictionary (must be JSON-serializable).
    """
    run_dir: str
    config_obj: Dict[str, Any]


@dataclass
class _MsgTickRow:
    """
    One row of per-tick statistics (to be written into stats.csv).

    `row` is typically a dictionary like:
        {
          "tick": 1234,
          "score_red": 10.5,
          "score_blue": 9.7,
          "alive_red": 200,
          ...
        }

    NOTE:
    - CSV writing expects a stable set of columns (fieldnames).
    - This code uses the keys from the first row as the header.
    """
    row: Dict[str, float]


@dataclass
class _MsgDeaths:
    """
    Batch of death/kill log rows (to be written into dead_agents_log.csv).

    We accept a list because deaths may occur in bursts and batching is more efficient:
    - fewer queue messages
    - fewer disk writes

    Each row is a dictionary (CSV columns are derived from first row).
    """
    rows: List[Dict[str, Any]]


@dataclass
class _MsgSaveModel:
    """
    Message to save a "model metadata" file describing a state_dict.

    This module intentionally does NOT import torch in the writer process:
    - keeps child process lightweight
    - avoids GPU context issues
    - avoids large imports in the spawned process

    We therefore store "meta" only:
    - keys of tensors
    - shapes if available (via .size())
    - otherwise fallback string markers

    The actual .pth model save (torch.save) should be done by the caller in the main process
    if desired.
    """
    label: str
    state_dict: Dict[str, Any]


class _MsgClose:
    """
    Sentinel message to tell the writer loop to shut down cleanly.
    """
    pass


# =============================================================================
# Background writer loop (runs inside the child process)
# =============================================================================

def _writer_loop(q: Queue) -> None:
    """
    Child process entry point.

    This function runs forever (until it receives _MsgClose), consuming messages from `q`
    and writing files.

    Design notes
    ------------
    1) We use `q.get(timeout=0.2)` instead of blocking forever so that:
       - the loop can periodically wake up
       - if you later add heartbeat logic or graceful shutdown checks, it's easy
       - it avoids hanging "forever" in certain edge cases

    2) Files are opened ONLY after _MsgInit is received.
       This ensures the run directory exists and the writer knows where to write.

    3) We flush after each write batch/row to reduce data loss if the program crashes.
       Trade-off: flushing frequently is slower. For critical logging, it's worth it.
    """
    run_dir = None

    # File handles (opened after init)
    stats_fp = None
    deaths_fp = None

    # CSV DictWriter objects (created lazily when first row arrives)
    stats_writer = None
    deaths_writer = None

    # --- NEW: Ignore Ctrl+C in this subprocess; main process coordinates shutdown. ---
    # Why:
    # - On Windows, Ctrl+C can be delivered to *all* processes in the console process group.
    # - The main process is responsible for coordinated shutdown and for sending _MsgClose.
    # - If this subprocess reacted to SIGINT by raising KeyboardInterrupt and exiting, you could
    #   lose buffered logs and/or leave the main process confused about writer state.
    try:
        import signal as _signal
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    except Exception:
        # If setting signal handler fails (platform quirks), we fall back to runtime handling below.
        pass

    try:
        while True:
            # -----------------------------------------------------------------
            # 1) Receive message (non-busy waiting)
            # -----------------------------------------------------------------
            try:
                # Wait up to 0.2 seconds for a message.
                # If nothing arrives, we loop again.
                msg = q.get(timeout=0.2)
            except KeyboardInterrupt:
                # On Windows, Ctrl+C can be delivered to the whole process group.
                # This subprocess must not die noisily; it should keep serving until _MsgClose.
                continue
            except queue.Empty:
                # No message available right now.
                continue

            # -----------------------------------------------------------------
            # 2) Handle init message
            # -----------------------------------------------------------------
            if isinstance(msg, _MsgInit):
                run_dir = msg.run_dir

                # Make sure the output directory exists.
                # exist_ok=True means: do not error if it already exists.
                os.makedirs(run_dir, exist_ok=True)

                # Save config.json (human-readable with indent=2)
                with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                    json.dump(msg.config_obj, f, indent=2)

                # Open CSV files for writing (overwrite mode).
                #
                # newline="" is important for CSV correctness on Windows:
                # it prevents extra blank lines.
                stats_fp = open(os.path.join(run_dir, "stats.csv"), "w", newline="", encoding="utf-8")
                deaths_fp = open(os.path.join(run_dir, "dead_agents_log.csv"), "w", newline="", encoding="utf-8")

                # We don't yet know CSV header columns until first row arrives.
                stats_writer = None
                deaths_writer = None

            # -----------------------------------------------------------------
            # 3) Handle tick stats row
            # -----------------------------------------------------------------
            elif isinstance(msg, _MsgTickRow):
                # Lazily create CSV writer on first row to determine the columns.
                if stats_writer is None:
                    # Header is derived from keys of the first row dictionary.
                    # NOTE: CSV column order is the order in this fieldnames list.
                    stats_writer = csv.DictWriter(stats_fp, fieldnames=list(msg.row.keys()))
                    stats_writer.writeheader()

                # Write one row.
                stats_writer.writerow(msg.row)

                # Flush to ensure row hits disk.
                stats_fp.flush()

            # -----------------------------------------------------------------
            # 4) Handle deaths batch
            # -----------------------------------------------------------------
            elif isinstance(msg, _MsgDeaths):
                # If list is empty, nothing to write.
                if not msg.rows:
                    continue

                if deaths_writer is None:
                    # Columns come from the first row's keys.
                    deaths_writer = csv.DictWriter(deaths_fp, fieldnames=list(msg.rows[0].keys()))
                    deaths_writer.writeheader()

                # Write many rows efficiently.
                deaths_writer.writerows(msg.rows)
                deaths_fp.flush()

            # -----------------------------------------------------------------
            # 5) Handle model metadata save
            # -----------------------------------------------------------------
            elif isinstance(msg, _MsgSaveModel):
                # "Torch-agnostic" metadata:
                # - If an object has .size() (like a torch tensor), record its shape as list(...)
                # - Else store a generic marker
                #
                # This avoids importing torch in this subprocess.
                meta = {
                    k: (list(v.size()) if hasattr(v, "size") else "tensor")
                    for k, v in msg.state_dict.items()
                }

                # Requires run_dir to exist (init should have happened).
                # If someone calls save_model_meta before start(), run_dir may be None.
                # This code assumes correct API usage; the public API enforces that by design.
                with open(os.path.join(run_dir, f"{msg.label}.state_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            # -----------------------------------------------------------------
            # 6) Handle shutdown
            # -----------------------------------------------------------------
            elif isinstance(msg, _MsgClose):
                break

            # Any unknown message type is silently ignored by design here.
            # In strict systems you'd log/raise, but for a telemetry sidecar,
            # "fail-safe" behavior is common.

    finally:
        # Always close file handles, even if an exception occurs.
        # We close each handle independently so one close failure doesn't block the other.
        try:
            if stats_fp is not None:
                stats_fp.close()
        except Exception:
            pass
        try:
            if deaths_fp is not None:
                deaths_fp.close()
        except Exception:
            pass


# =============================================================================
# Public API (used by the main process)
# =============================================================================

class ResultsWriter:
    """
    ResultsWriter
    =============

    A simple, non-blocking telemetry writer for simulations/training loops.

    The writer is implemented as a separate process that writes:
    - config.json
    - stats.csv (one row per tick)
    - dead_agents_log.csv (many rows per batch)
    - <label>.state_meta.json (model state metadata)

    Primary goals
    -------------
    1) Keep the main simulation loop fast (avoid blocking on disk I/O).
    2) Be Windows-friendly (spawn-safe).
    3) Be resilient: if queue is full, drop logs rather than freeze the sim.

    Usage pattern
    -------------
    writer = ResultsWriter()
    run_dir = writer.start(config_obj)

    for tick in ...:
        writer.write_tick({...})
        writer.write_deaths([...])  # optional

    writer.save_model_meta("policy", state_dict)  # optional
    writer.close()

    Notes
    -----
    - This class does NOT guarantee that every message is written.
      If the queue is full, it drops messages intentionally.
    - If you need strict durability (never lose a log), do NOT drop on queue.Full;
      instead block (but that may harm real-time performance).
    """

    def __init__(self) -> None:
        # Queue used for IPC (inter-process communication).
        #
        # maxsize=1024:
        # - Limits memory usage.
        # - Provides backpressure signal (queue.Full).
        # - 1024 is a reasonable default; adjust depending on logging rate.
        self.q: Queue = Queue(maxsize=1024)

        # Background process handle (None until started).
        self.p: Optional[Process] = None

        # Run directory path (set at start()).
        self.run_dir: Optional[str] = None

    @staticmethod
    def _timestamp_dir(base: str = "results") -> str:
        """
        Create a default run directory name using local timestamp.

        Format:
            results/sim_YYYY-MM-DD_HH-MM-SS

        This keeps runs separated and sortable.

        base:
            The parent folder to contain runs.
        """
        ts = datetime.datetime.now().strftime("sim_%Y-%m-%d_%H-%M-%S")
        return os.path.join(base, ts)

    def start(self, config_obj: Dict[str, Any], run_dir: Optional[str] = None) -> str:
        """
        Start the writer process and initialize output files.

        Parameters
        ----------
        config_obj:
            Configuration dictionary to write to config.json.
            Must be JSON serializable (otherwise json.dump will fail).
        run_dir:
            Optional explicit directory path. If None, a timestamp directory is created.

        Returns
        -------
        str:
            The chosen run directory path.

        Important behavioral detail
        ---------------------------
        We spawn a daemon process:
        - daemon=True means it will not prevent the program from exiting.
        - However, relying purely on daemon behavior risks losing buffered logs at exit.
          Always call close() for clean shutdown when possible.
        """
        self.run_dir = run_dir or self._timestamp_dir()

        # Create the background process.
        #
        # target=_writer_loop: function executed in child process
        # args=(self.q,): pass the Queue object to the child
        self.p = Process(target=_writer_loop, args=(self.q,), daemon=True)

        # Start the process (it begins executing _writer_loop).
        self.p.start()

        # Send init message so the child can:
        # - create directory
        # - write config.json
        # - open stats.csv and dead_agents_log.csv
        self.q.put(_MsgInit(run_dir=self.run_dir, config_obj=config_obj))

        return self.run_dir

    def write_tick(self, row: Dict[str, float]) -> None:
        """
        Enqueue a single per-tick statistics row.

        Non-blocking by design:
        - If writer is not started, does nothing.
        - If queue is full, silently drops the message.

        row:
            Dict of numeric metrics for this tick.
            Example: {"tick": 100, "score_r": 1.2, "alive_r": 200, ...}
        """
        if self.p is None:
            return

        try:
            self.q.put_nowait(_MsgTickRow(row=row))
        except queue.Full:
            # Drop instead of blocking the simulation.
            pass

    def write_deaths(self, rows: List[Dict[str, Any]]) -> None:
        """
        Enqueue a batch of death log rows.

        rows:
            A list of dictionaries. If empty, nothing is sent.

        Non-blocking:
        - If queue is full, drops the message.
        """
        if self.p is None or not rows:
            return

        try:
            self.q.put_nowait(_MsgDeaths(rows=rows))
        except queue.Full:
            pass

    def save_model_meta(self, label: str, state_dict: Dict[str, Any]) -> None:
        """
        Enqueue a request to write model metadata.

        label:
            Used to name the output file: <label>.state_meta.json
        state_dict:
            Dictionary of model parameters (often torch tensors).
            This process stores only torch-agnostic metadata (keys and shapes).

        Non-blocking:
        - If queue is full, drops the message.
        """
        if self.p is None:
            return

        try:
            self.q.put_nowait(_MsgSaveModel(label=label, state_dict=state_dict))
        except queue.Full:
            pass

    def close(self) -> None:
        """
        Shut down the writer process.

        Steps
        -----
        1) Send _MsgClose sentinel (blocking put).
        2) Join for up to 2 seconds.
        3) If still alive, terminate forcefully.

        Why force terminate?
        --------------------
        In production systems, logging should never prevent the main app from exiting.
        If the writer is stuck (e.g., disk issues), we prefer to exit rather than hang.
        """
        if self.p is None:
            return

        try:
            # Blocking put is acceptable here because we're shutting down.
            self.q.put(_MsgClose())

            # Wait a bit for clean exit.
            self.p.join(timeout=2.0)
        finally:
            # If it didn't exit in time, kill it.
            if self.p.is_alive():
                self.p.terminate()

            # Reset handle so the object can be garbage-collected safely.
            self.p = None