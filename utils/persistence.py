"""Asynchronous run-output writer."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, Any, Optional, List

import os
import json
import csv
import datetime
import queue
import threading


_DEATHS_FIELDNAMES_CURRENT = ["tick", "agent_id", "team", "x", "y", "killer_team"]
_DEATHS_FIELDNAMES_LEGACY_V1 = ["tick", "agent_id", "team", "x", "y"]

@dataclass
class _MsgInit:
    """Initialize the writer process."""
    run_dir: str
    config_obj: Dict[str, Any]
    append_existing: bool = False
    strict_csv_schema: bool = False


@dataclass
class _MsgTickRow:
    """Write one row to ``stats.csv``."""
    row: Dict[str, float]


@dataclass
class _MsgDeaths:
    """Write a batch of rows to ``dead_agents_log.csv``."""
    rows: List[Dict[str, Any]]


@dataclass
class _MsgSaveModel:
    """Write a lightweight metadata view of a model state dict."""
    label: str
    state_dict: Dict[str, Any]


class _MsgClose:
    """Stop the writer loop."""
    pass


class _ThreadProcess:
    """Thread-backed stand-in when multiprocessing pipes are unavailable."""

    def __init__(self, target, args=(), daemon: bool = True):
        self._target = target
        self._args = args
        self.exitcode = None
        self._thread = threading.Thread(target=self._run, daemon=daemon)

    def _run(self) -> None:
        try:
            self._target(*self._args)
        except Exception:
            self.exitcode = 1
        else:
            self.exitcode = 0

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout=None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def terminate(self) -> None:
        self.exitcode = -15


def _read_csv_header(path: str) -> Optional[List[str]]:
    """Return the header row for an existing CSV, or None when unavailable."""
    if (not os.path.exists(path)) or os.path.getsize(path) <= 0:
        return None
    try:
        with open(path, "r", newline="", encoding="utf-8") as rf:
            return next(csv.reader(rf), None)
    except Exception:
        return None


def _migrate_legacy_dead_agents_log(path: str) -> bool:
    """
    Upgrade historical root death ledgers that predate the ``killer_team`` column.

    Migration contract:
    - only the exact known legacy header is auto-migrated
    - existing rows are preserved in order
    - migrated legacy rows get an empty ``killer_team`` value
    """
    header = _read_csv_header(path)
    if header != _DEATHS_FIELDNAMES_LEGACY_V1:
        return False

    rows: List[List[str]] = []
    with open(path, "r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        next(reader, None)  # header
        for idx, row in enumerate(reader, start=2):
            if len(row) != len(_DEATHS_FIELDNAMES_LEGACY_V1):
                raise RuntimeError(
                    f"[ResultsWriter] cannot migrate legacy dead_agents_log.csv row {idx}: "
                    f"expected {len(_DEATHS_FIELDNAMES_LEGACY_V1)} columns, got {len(row)}"
                )
            rows.append(list(row) + [""])

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(_DEATHS_FIELDNAMES_CURRENT)
        writer.writerows(rows)
    os.replace(tmp_path, path)
    return True


def _prepare_append_run_dir(run_dir: str, *, strict_csv_schema: bool) -> None:
    """
    Preflight append-mode files before the worker process starts.

    This keeps historical known-good migrations deterministic and pushes
    irreconcilable schema drift to startup time instead of a later worker crash.
    """
    deaths_path = os.path.join(run_dir, "dead_agents_log.csv")
    migrated = _migrate_legacy_dead_agents_log(deaths_path)
    header = _read_csv_header(deaths_path)

    if header is None:
        return

    if header == _DEATHS_FIELDNAMES_CURRENT:
        return

    if migrated:
        return

    _ = strict_csv_schema
    raise RuntimeError(
        "[ResultsWriter] dead_agents_log.csv header is incompatible with append-mode resume. "
        f"expected={_DEATHS_FIELDNAMES_CURRENT} legacy_supported={_DEATHS_FIELDNAMES_LEGACY_V1} "
        f"found={header}"
    )

def _writer_loop(q: Queue, err_q: Optional[Queue] = None) -> None:
    """Child-process entry point for background writes."""
    run_dir = None
    stats_fp = None
    deaths_fp = None
    stats_writer = None
    deaths_writer = None
    stats_fieldnames_expected = None
    deaths_fieldnames_expected = None
    strict_csv_schema = False

    try:
        import signal as _signal
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    except Exception:
        pass

    try:
        while True:
            try:
                msg = q.get(timeout=0.2)
            except KeyboardInterrupt:
                continue
            except queue.Empty:
                continue

            if isinstance(msg, _MsgInit):
                run_dir = msg.run_dir
                append_existing = bool(getattr(msg, "append_existing", False))
                strict_csv_schema = bool(getattr(msg, "strict_csv_schema", False))

                os.makedirs(run_dir, exist_ok=True)

                config_path = os.path.join(run_dir, "config.json")
                if append_existing and os.path.exists(config_path):
                    pass
                else:
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(msg.config_obj, f, indent=2)

                stats_path = os.path.join(run_dir, "stats.csv")
                deaths_path = os.path.join(run_dir, "dead_agents_log.csv")

                stats_mode = "a" if append_existing else "w"
                deaths_mode = "a" if append_existing else "w"
                stats_fp = open(stats_path, stats_mode, newline="", encoding="utf-8")
                deaths_fp = open(deaths_path, deaths_mode, newline="", encoding="utf-8")

                stats_writer = None
                deaths_writer = None

                stats_fieldnames_expected = _read_csv_header(stats_path) if append_existing else None
                deaths_fieldnames_expected = _read_csv_header(deaths_path) if append_existing else None

            elif isinstance(msg, _MsgTickRow):
                if stats_writer is None:
                    fieldnames = list(msg.row.keys())
                    if strict_csv_schema and stats_fieldnames_expected is not None and list(stats_fieldnames_expected) != fieldnames:
                        raise RuntimeError(
                            f"[ResultsWriter] stats.csv header mismatch on append. "
                            f"existing={stats_fieldnames_expected} new={fieldnames}"
                        )
                    stats_writer = csv.DictWriter(stats_fp, fieldnames=fieldnames)
                    if not stats_fieldnames_expected:
                        stats_writer.writeheader()

                stats_writer.writerow(msg.row)
                stats_fp.flush()

            elif isinstance(msg, _MsgDeaths):
                if not msg.rows:
                    continue

                if deaths_writer is None:
                    fieldnames = list(msg.rows[0].keys())
                    if strict_csv_schema and deaths_fieldnames_expected is not None and list(deaths_fieldnames_expected) != fieldnames:
                        raise RuntimeError(
                            f"[ResultsWriter] dead_agents_log.csv header mismatch on append. "
                            f"existing={deaths_fieldnames_expected} new={fieldnames}"
                        )
                    deaths_writer = csv.DictWriter(deaths_fp, fieldnames=fieldnames)
                    if not deaths_fieldnames_expected:
                        deaths_writer.writeheader()

                deaths_writer.writerows(msg.rows)
                deaths_fp.flush()

            elif isinstance(msg, _MsgSaveModel):
                meta = {
                    k: (list(v.size()) if hasattr(v, "size") else "tensor")
                    for k, v in msg.state_dict.items()
                }

                with open(os.path.join(run_dir, f"{msg.label}.state_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            elif isinstance(msg, _MsgClose):
                break
    except Exception as e:
        if err_q is not None:
            try:
                err_q.put_nowait(f"{type(e).__name__}: {e}")
            except Exception:
                pass
        raise

    finally:
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

class ResultsWriter:
    """Non-blocking writer for run metadata and CSV outputs."""

    def __init__(self) -> None:
        self.q = None
        self.err_q = None
        self.p: Optional[Process] = None
        self.run_dir: Optional[str] = None
        self._worker_error: Optional[str] = None
        self.worker_backend: str = "unstarted"

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

    @staticmethod
    def _resolve_run_dir(run_dir: Optional[str], *, base_dir: Optional[str] = None) -> str:
        """Resolve the output directory for a fresh or resumed run."""
        if run_dir is not None and base_dir is not None:
            raise ValueError("base_dir cannot be combined with an explicit run_dir")
        if run_dir is not None:
            return run_dir

        base = str(base_dir or "results").strip() or "results"
        return ResultsWriter._timestamp_dir(base=base)

    def _raise_worker_error(self) -> None:
        """Raise if the background writer has already failed."""
        if self._worker_error is None:
            try:
                if self.err_q is not None:
                    self._worker_error = self.err_q.get_nowait()
            except queue.Empty:
                pass

        if self._worker_error is not None:
            raise RuntimeError(f"[ResultsWriter] worker failed: {self._worker_error}")

        if self.p is not None and (not self.p.is_alive()):
            exitcode = getattr(self.p, "exitcode", None)
            if exitcode not in (None, 0):
                raise RuntimeError(
                    f"[ResultsWriter] worker exited unexpectedly with exitcode={exitcode}"
                )

    def _start_worker(self) -> None:
        """Start a background writer, falling back to a thread when MP pipes fail."""
        mp_error: Optional[BaseException] = None

        try:
            q = Queue(maxsize=1024)
            err_q = Queue(maxsize=32)
            p = Process(target=_writer_loop, args=(q, err_q), daemon=True)
            p.start()
            self.q = q
            self.err_q = err_q
            self.p = p
            self.worker_backend = "process"
            return
        except (PermissionError, OSError) as exc:
            mp_error = exc

        self.q = queue.Queue(maxsize=1024)
        self.err_q = queue.Queue(maxsize=32)
        self.p = _ThreadProcess(target=_writer_loop, args=(self.q, self.err_q), daemon=True)
        self.p.start()
        self.worker_backend = "thread"
        print(
            "[ResultsWriter] multiprocessing unavailable; "
            f"falling back to thread writer ({type(mp_error).__name__}: {mp_error})"
        )

    def start(
        self,
        config_obj: Dict[str, Any],
        run_dir: Optional[str] = None,
        *,
        base_dir: Optional[str] = None,
        append_existing: bool = False,
        strict_csv_schema: bool = False,
    ) -> str:
        """Start the writer process and return the chosen run directory."""
        self.run_dir = self._resolve_run_dir(run_dir, base_dir=base_dir)

        if append_existing and run_dir is None:
            raise ValueError("append_existing=True requires an explicit run_dir")
        if append_existing and (not os.path.isdir(self.run_dir)):
            raise FileNotFoundError(f"append_existing run_dir does not exist: {self.run_dir}")
        if append_existing:
            _prepare_append_run_dir(
                self.run_dir,
                strict_csv_schema=bool(strict_csv_schema),
            )

        self._worker_error = None
        self._start_worker()
        assert self.q is not None
        self.q.put(
            _MsgInit(
                run_dir=self.run_dir,
                config_obj=config_obj,
                append_existing=bool(append_existing),
                strict_csv_schema=bool(strict_csv_schema),
            )
        )

        return self.run_dir

    def write_tick(self, row: Dict[str, float]) -> None:
        """Queue one per-tick statistics row."""
        if self.p is None:
            return
        self._raise_worker_error()

        try:
            self.q.put_nowait(_MsgTickRow(row=row))
        except queue.Full:
            pass

    def write_deaths(self, rows: List[Dict[str, Any]]) -> None:
        """Queue a batch of death log rows."""
        if self.p is None or not rows:
            return
        self._raise_worker_error()

        try:
            self.q.put_nowait(_MsgDeaths(rows=rows))
        except queue.Full:
            pass

    def save_model_meta(self, label: str, state_dict: Dict[str, Any]) -> None:
        """Queue a request to write model metadata."""
        if self.p is None:
            return
        self._raise_worker_error()

        try:
            self.q.put_nowait(_MsgSaveModel(label=label, state_dict=state_dict))
        except queue.Full:
            pass

    def close(self) -> None:
        """Shut down the writer process."""
        if self.p is None:
            return

        err: Optional[RuntimeError] = None
        try:
            self._raise_worker_error()
        except RuntimeError as exc:
            err = exc

        try:
            if self.q is not None:
                self.q.put(_MsgClose())
        except Exception:
            pass

        try:
            self.p.join(timeout=2.0)
        finally:
            if self.p.is_alive():
                self.p.terminate()

        try:
            if err is None:
                self._raise_worker_error()
        except RuntimeError as exc:
            err = exc
        finally:
            self.p = None
            self.q = None
            self.err_q = None
            self.worker_backend = "closed"

        if err is not None:
            raise err
