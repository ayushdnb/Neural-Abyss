
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import time

import config

TEAM_RED  = "red"
TEAM_BLUE = "blue"


@dataclass
class TeamCounters:
    score: float = 0.0
    kills: int = 0
    deaths: int = 0
    dmg_dealt: float = 0.0
    dmg_taken: float = 0.0
    cp_points: float = 0.0  

    def clone(self) -> "TeamCounters":
        return TeamCounters(
            self.score, self.kills, self.deaths, self.dmg_dealt, self.dmg_taken, self.cp_points
        )


@dataclass
class Snapshot:
    red: TeamCounters
    blue: TeamCounters
    tick: int


class SimulationStats:
    """
    Team-scoped scoring + dead-agent log. Designed for per-agent PPO delta rewards.
    """
    def __init__(self) -> None:
        self.red = TeamCounters()
        self.blue = TeamCounters()
        self.tick = 0
        self._t0 = time.perf_counter()
        self._dead_log: List[Dict[str, float | int]] = []

    # ---- timing ----
    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def on_tick_advanced(self, dt: int) -> None:
        self.tick += int(dt)

    # ---- helpers ----
    def _team(self, name: str) -> TeamCounters:
        return self.red if name == TEAM_RED else self.blue

    # ---- points wiring ----
    def add_damage_dealt(self, team: str, amount: float) -> None:
        t = self._team(team)
        t.dmg_dealt += float(amount)
        t.score += float(amount) * float(config.TEAM_DMG_DEALT_REWARD)

    def add_damage_taken(self, team: str, amount: float) -> None:
        t = self._team(team)
        t.dmg_taken += float(amount)
        t.score += float(amount) * float(config.TEAM_DMG_TAKEN_PENALTY)

    def add_kill(self, team: str, count: int = 1) -> None:
        t = self._team(team)
        t.kills += int(count)
        t.score += float(count) * float(config.TEAM_KILL_REWARD)

    def add_death(self, team: str, count: int = 1) -> None:
        t = self._team(team)
        t.deaths += int(count)
        t.score += float(count) * float(config.TEAM_DEATH_PENALTY)

    def add_capture_points(self, team: str, amount: float) -> None:
        t = self._team(team)
        t.cp_points += float(amount)
        t.score += float(amount)

    # ---- structured death log ----
    def record_death_entry(self, agent_id: int, team_id_val: float, x: int, y: int, killer_team_id_val: float | int) -> None:
        self._dead_log.append({
            "tick": self.tick,
            "agent_id": int(agent_id),
            "team": "red" if float(team_id_val) == 2.0 else "blue",
            "x": int(x), "y": int(y),
            "killer_team": "red" if float(killer_team_id_val) == 2.0 else "blue",
        })

    def drain_dead_log(self) -> List[Dict[str, float | int]]:
        buf = self._dead_log
        self._dead_log = []
        return buf

    # ---- snapshots / CSV rows ----
    def snapshot(self) -> Snapshot:
        return Snapshot(self.red.clone(), self.blue.clone(), self.tick)

    def delta_since(self, snap: Snapshot) -> Dict[str, float]:
        return {
            TEAM_RED:  self.red.score  - snap.red.score,
            TEAM_BLUE: self.blue.score - snap.blue.score,
        }

    def as_row(self) -> Dict[str, float]:
        return {
            "tick": float(self.tick),
            "elapsed_s": float(self.elapsed_seconds),
            "red_score": float(self.red.score),
            "blue_score": float(self.blue.score),
            "red_kills": float(self.red.kills),
            "blue_kills": float(self.blue.kills),
            "red_deaths": float(self.red.deaths),
            "blue_deaths": float(self.blue.deaths),
            "red_dmg_dealt": float(self.red.dmg_dealt),
            "blue_dmg_dealt": float(self.blue.dmg_dealt),
            "red_dmg_taken": float(self.red.dmg_taken),
            "blue_dmg_taken": float(self.blue.dmg_taken),
            "red_cp_points": float(self.red.cp_points),
            "blue_cp_points": float(self.blue.cp_points),
        }
