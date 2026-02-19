from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, Any, Optional, List
import os, json, csv, time, datetime, queue

# ---- Messages for the writer process ----
@dataclass
class _MsgInit:
    run_dir: str
    config_obj: Dict[str, Any]

@dataclass
class _MsgTickRow:
    row: Dict[str, float]

@dataclass
class _MsgDeaths:
    rows: List[Dict[str, Any]]

@dataclass
class _MsgSaveModel:
    label: str
    state_dict: Dict[str, Any]

class _MsgClose: pass

# ---- Background writer ----
def _writer_loop(q: Queue):
    run_dir = None
    stats_fp = None
    stats_writer = None
    deaths_fp = None
    deaths_writer = None
    try:
        while True:
            try:
                msg = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(msg, _MsgInit):
                run_dir = msg.run_dir
                os.makedirs(run_dir, exist_ok=True)
                # config.json
                with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                    json.dump(msg.config_obj, f, indent=2)

                # open CSVs
                stats_fp = open(os.path.join(run_dir, "stats.csv"), "w", newline="", encoding="utf-8")
                deaths_fp = open(os.path.join(run_dir, "dead_agents_log.csv"), "w", newline="", encoding="utf-8")
                stats_writer = None
                deaths_writer = None

            elif isinstance(msg, _MsgTickRow):
                if stats_writer is None:
                    # header from keys
                    stats_writer = csv.DictWriter(stats_fp, fieldnames=list(msg.row.keys()))
                    stats_writer.writeheader()
                stats_writer.writerow(msg.row)
                stats_fp.flush()

            elif isinstance(msg, _MsgDeaths):
                if not msg.rows:
                    continue
                if deaths_writer is None:
                    deaths_writer = csv.DictWriter(deaths_fp, fieldnames=list(msg.rows[0].keys()))
                    deaths_writer.writeheader()
                deaths_writer.writerows(msg.rows)
                deaths_fp.flush()

            elif isinstance(msg, _MsgSaveModel):
                # save as torch-agnostic JSON (keys and shapes), AND raw .pth if available
                # We cannot import torch here (keep process light). Expect caller to save .pth too if needed.
                meta = {k: (list(v.size()) if hasattr(v, "size") else "tensor") for k, v in msg.state_dict.items()}
                with open(os.path.join(run_dir, f"{msg.label}.state_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            elif isinstance(msg, _MsgClose):
                break
    finally:
        for fp in (stats_fp, deaths_fp):
            try:
                if fp: fp.close()
            except Exception:
                pass

# ---- Public API ----
class ResultsWriter:
    """
    Windows-friendly background writer (multiprocessing.Process).
    Non-blocking calls: init(), write_tick(), write_deaths(), save_model_meta(), close().
    """
    def __init__(self) -> None:
        self.q: Queue = Queue(maxsize=1024)
        self.p: Optional[Process] = None
        self.run_dir: Optional[str] = None

    @staticmethod
    def _timestamp_dir(base: str = "results") -> str:
        ts = datetime.datetime.now().strftime("sim_%Y-%m-%d_%H-%M-%S")
        return os.path.join(base, ts)

    def start(self, config_obj: Dict[str, Any], run_dir: Optional[str] = None) -> str:
        self.run_dir = run_dir or self._timestamp_dir()
        self.p = Process(target=_writer_loop, args=(self.q,), daemon=True)
        self.p.start()
        self.q.put(_MsgInit(run_dir=self.run_dir, config_obj=config_obj))
        return self.run_dir

    def write_tick(self, row: Dict[str, float]) -> None:
        if self.p is None: return
        try: self.q.put_nowait(_MsgTickRow(row=row))
        except queue.Full: pass

    def write_deaths(self, rows: List[Dict[str, Any]]) -> None:
        if self.p is None or not rows: return
        try: self.q.put_nowait(_MsgDeaths(rows=rows))
        except queue.Full: pass

    def save_model_meta(self, label: str, state_dict: Dict[str, Any]) -> None:
        if self.p is None: return
        try: self.q.put_nowait(_MsgSaveModel(label=label, state_dict=state_dict))
        except queue.Full: pass

    def close(self) -> None:
        if self.p is None: return
        try:
            self.q.put(_MsgClose())
            self.p.join(timeout=2.0)
        finally:
            if self.p.is_alive():
                self.p.terminate()
            self.p = None
