from pathlib import Path
import queue
import threading
import time

import pytest

import utils.persistence as persistence
from utils.persistence import ResultsWriter


def test_resolve_run_dir_uses_custom_base_dir(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"

    run_dir = Path(ResultsWriter._resolve_run_dir(None, base_dir=str(base_dir)))

    assert run_dir.parent == base_dir
    assert run_dir.name.startswith("sim_")


def test_resolve_run_dir_rejects_explicit_run_dir_and_base_dir() -> None:
    with pytest.raises(ValueError):
        ResultsWriter._resolve_run_dir("results/sim_existing", base_dir="results_alt")


class _ThreadProcess:
    def __init__(self, target, args=(), daemon=True):
        self._target = target
        self._args = args
        self._daemon = daemon
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


def test_results_writer_surfaces_worker_schema_mismatch(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "stats.csv").write_text("tick,red_score\n0,1.0\n", encoding="utf-8")
    (run_dir / "dead_agents_log.csv").write_text("", encoding="utf-8")

    monkeypatch.setattr(persistence, "Process", _ThreadProcess)
    monkeypatch.setattr(persistence, "Queue", queue.Queue)

    writer = ResultsWriter()
    try:
        writer.start(
            config_obj={"summary": "test"},
            run_dir=str(run_dir),
            append_existing=True,
            strict_csv_schema=True,
        )
        writer.write_tick({"wrong": 1.0})

        deadline = time.time() + 2.0
        while writer.p is not None and writer.p.exitcode is None and time.time() < deadline:
            time.sleep(0.01)

        with pytest.raises(RuntimeError, match="stats.csv header mismatch"):
            writer.write_tick({"wrong": 2.0})
    finally:
        if writer.p is not None:
            try:
                writer.close()
            except RuntimeError:
                pass


def test_results_writer_migrates_legacy_dead_log_for_resume_append(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "stats.csv").write_text("tick,red_score\n0,1.0\n", encoding="utf-8")
    (run_dir / "dead_agents_log.csv").write_text(
        "tick,agent_id,team,x,y\n1,7,red,2,3\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(persistence, "Process", _ThreadProcess)
    monkeypatch.setattr(persistence, "Queue", queue.Queue)

    writer = ResultsWriter()
    try:
        writer.start(
            config_obj={"summary": "test"},
            run_dir=str(run_dir),
            append_existing=True,
            strict_csv_schema=True,
        )
        writer.write_deaths(
            [
                {
                    "tick": 2,
                    "agent_id": 8,
                    "team": "blue",
                    "x": 1,
                    "y": 1,
                    "killer_team": "red",
                }
            ]
        )
        writer.close()
    finally:
        if writer.p is not None:
            writer.close()

    lines = (run_dir / "dead_agents_log.csv").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "tick,agent_id,team,x,y,killer_team",
        "1,7,red,2,3,",
        "2,8,blue,1,1,red",
    ]


def test_results_writer_rejects_unknown_dead_log_header_before_worker_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "stats.csv").write_text("tick,red_score\n0,1.0\n", encoding="utf-8")
    (run_dir / "dead_agents_log.csv").write_text(
        "tick,agent_id,team,x,y,killer_team,extra\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(persistence, "Process", _ThreadProcess)
    monkeypatch.setattr(persistence, "Queue", queue.Queue)

    writer = ResultsWriter()
    with pytest.raises(RuntimeError, match="dead_agents_log.csv header is incompatible"):
        writer.start(
            config_obj={"summary": "test"},
            run_dir=str(run_dir),
            append_existing=True,
            strict_csv_schema=True,
        )


def test_results_writer_real_process_schema_mismatch_harness(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "stats.csv").write_text("tick,red_score\n0,1.0\n", encoding="utf-8")
    (run_dir / "dead_agents_log.csv").write_text("", encoding="utf-8")

    try:
        writer = ResultsWriter()
        writer.start(
            config_obj={"summary": "test"},
            run_dir=str(run_dir),
            append_existing=True,
            strict_csv_schema=True,
        )
    except PermissionError as exc:
        pytest.skip(f"sandbox blocks real multiprocessing pipes: {exc}")

    try:
        err = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                writer.write_tick({"wrong": 1.0})
            except RuntimeError as exc:
                err = exc
                break
            time.sleep(0.05)

        assert err is not None, "real-process schema mismatch did not surface before timeout"
        assert "stats.csv header mismatch" in str(err)
    finally:
        try:
            writer.close()
        except RuntimeError:
            pass
