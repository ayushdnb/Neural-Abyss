"""Team-scoped scoring and structured death logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import time

import config


# Team identifiers
# These constants avoid "magic strings" spread across the codebase.
# If later you change a team name, you update it once here.
TEAM_RED = "red"
TEAM_BLUE = "blue"


# Data containers (dataclasses)
@dataclass
class TeamCounters:
    """
    Cumulative counters for one team.

    "Cumulative" means these values only ever increase during a run.
    They represent totals since the start of the simulation.

    Fields
    ------
    score:
        A single scalar score used as a high-level "how well are we doing?" metric.
        In RL, this can be constructed from rewards and penalties.

    kills / deaths:
        Integer counters for how many agents the team killed / lost.

    dmg_dealt / dmg_taken:
        Damage totals. These are often more informative than kills alone because
        they provide a denser signal (happens more frequently than kills).

    cp_points:
        "Capture Points" or "Control Points" earned by occupying objectives.
        This is domain-specific (your game mechanic). It contributes to score.

    Why float for most values?
    --------------------------
    - Damage and capture points can be fractional (e.g., continuous scoring).
    - Score is a weighted sum of multiple components, so float is natural.
    - Kills and deaths are discrete events -> int.
    """
    score: float = 0.0
    kills: int = 0
    deaths: int = 0
    dmg_dealt: float = 0.0
    dmg_taken: float = 0.0
    cp_points: float = 0.0

    def clone(self) -> "TeamCounters":
        """
        Return a deep-ish copy of this counters object.

        Why we need clone():
        --------------------
        Snapshots should be immune to future updates.
        If we returned the same object reference, snapshots would "change under us"
        as the simulation continues (a classic bug for beginners).

        Since all fields are primitive numeric values, constructing a new instance
        is sufficient (no nested mutable structures here).
        """
        return TeamCounters(
            self.score,
            self.kills,
            self.deaths,
            self.dmg_dealt,
            self.dmg_taken,
            self.cp_points,
        )


@dataclass
class Snapshot:
    """
    A point-in-time record of both teams' counters and the simulation tick.

    This is primarily used for computing *deltas*:
        delta_score = current.score - snapshot.score

    In RL / logging, this pattern is common:
    - keep cumulative totals
    - periodically take snapshots
    - compute changes since last snapshot
    """
    red: TeamCounters
    blue: TeamCounters
    tick: int


# Main stats manager
class SimulationStats:
    """
    Team-scoped scoring + structured death log.

    This is designed to support:
    - console reporting (score, kills, etc.)
    - training logs (CSV/Parquet)
    - PPO-style "delta rewards" (reward between snapshots)
    - debugging events (death log)

    ABOUT PPO DELTA REWARDS (HIGH-LEVEL)
    -----------------------------------
    PPO expects reward signals over time steps. Often the simulation produces
    events (damage, kills, objectives) irregularly. A simple and stable approach
    is to:

    1) Maintain cumulative totals (e.g., total score so far)
    2) Take a snapshot at time t0
    3) At time t1, reward = total_score(t1) - total_score(t0)

    This is exactly what delta_since(snapshot) provides.
    """

    def __init__(self) -> None:
        # Two team buckets:
        self.red = TeamCounters()
        self.blue = TeamCounters()

        # Simulation tick is an integer time-step counter controlled by the engine.
        self.tick = 0

        # Start time marker used for elapsed time.
        # perf_counter() is:
        # - monotonic (won't jump backwards)
        # - high resolution
        self._t0 = time.perf_counter()

        # Structured death log: list of dictionaries.
        # Each entry records a death event with minimal useful fields.
        # Why not a dataclass for death entries?
        # - Dicts are convenient for JSON/CSV output and ad-hoc analytics.
        # - Schema is stable enough here.
        self._dead_log: List[Dict[str, float | int]] = []

    # Timing helpers
    @property
    def elapsed_seconds(self) -> float:
        """
        Total wall-clock seconds since this stats object was created.

        Wall-clock vs simulation time:
        - tick is "simulation time" (discrete steps)
        - elapsed_seconds is "real time" measured on the machine

        This is useful to estimate:
        - performance (ticks per second)
        - training duration
        - wall-time progress reporting
        """
        return time.perf_counter() - self._t0

    def on_tick_advanced(self, dt: int) -> None:
        """
        Advance the tick counter by dt.

        Parameters
        ----------
        dt:
            Number of ticks to advance by (usually 1).
            We force int() to guard against accidental float inputs.

        Why this method exists:
        -----------------------
        It keeps the "tick update" logic centralized, so the engine can call it
        once per step without directly mutating self.tick everywhere.
        """
        self.tick += int(dt)

    # Internal helpers
    def _team(self, name: str) -> TeamCounters:
        """
        Resolve a team name to the corresponding TeamCounters object.

        This is a small convenience function to keep the add_* methods clean.

        Important behavior:
        -------------------
        - If name == TEAM_RED -> return red counters
        - Else -> return blue counters

        Note:
        -----
        This assumes only two teams exist. If you later introduce more teams,
        you'd typically replace this with a dict mapping.
        """
        return self.red if name == TEAM_RED else self.blue

    # Scoring wiring (events -> counters -> score)
    def add_damage_dealt(self, team: str, amount: float) -> None:
        """
        Record damage dealt by a team and add its contribution to team score.

        Score update:
        -------------
        score += amount * TEAM_DMG_DEALT_REWARD

        Interpretation:
        ---------------
        - If TEAM_DMG_DEALT_REWARD is positive, dealing damage is rewarded.
        - If it is 0, damage is tracked but not rewarded.
        - If it is negative (rare), damage would be penalized.

        Type handling:
        -------------
        We cast to float to ensure consistent numeric types even if caller passes
        numpy scalars or ints.
        """
        t = self._team(team)
        t.dmg_dealt += float(amount)
        t.score += float(amount) * float(config.TEAM_DMG_DEALT_REWARD)

    def add_damage_taken(self, team: str, amount: float) -> None:
        """
        Record damage taken by a team and add its contribution to team score.

        Score update:
        -------------
        score += amount * TEAM_DMG_TAKEN_PENALTY

        Interpretation:
        ---------------
        Typically TEAM_DMG_TAKEN_PENALTY is negative (a penalty).
        That means taking damage reduces score.

        This is common in RL shaping:
        - reward good events (dealing damage)
        - penalize bad events (taking damage)
        """
        t = self._team(team)
        t.dmg_taken += float(amount)
        t.score += float(amount) * float(config.TEAM_DMG_TAKEN_PENALTY)

    def add_kill(self, team: str, count: int = 1) -> None:
        """
        Record kills for a team and update score.

        Score update:
        -------------
        score += count * TEAM_KILL_REWARD

        count default:
        --------------
        Default is 1 because kills usually happen one at a time,
        but batched updates are sometimes useful for vectorized engines.
        """
        t = self._team(team)
        t.kills += int(count)
        t.score += float(count) * float(config.TEAM_KILL_REWARD)

    def add_death(self, team: str, count: int = 1) -> None:
        """
        Record deaths for a team and update score.

        Score update:
        -------------
        score += count * TEAM_DEATH_PENALTY

        Typically TEAM_DEATH_PENALTY is negative.
        """
        t = self._team(team)
        t.deaths += int(count)
        t.score += float(count) * float(config.TEAM_DEATH_PENALTY)

    def add_capture_points(self, team: str, amount: float) -> None:
        """
        Record capture/control points earned by a team and update score.

        Current rule:
        -------------
        score += amount

        This means CP points contribute 1:1 into score.

        In more complex systems, you might use a config multiplier here,
        but this code intentionally keeps CP scoring direct and simple.
        """
        t = self._team(team)
        t.cp_points += float(amount)
        t.score += float(amount)

    # Structured death log
    def record_death_entry(
        self,
        agent_id: int,
        team_id_val: float,
        x: int,
        y: int,
        killer_team_id_val: float | int,
    ) -> None:
        """
        Record a single death event into the structured death log.

        Parameters
        ----------
        agent_id:
            Unique ID of the agent who died.

        team_id_val:
            Numeric team identifier used by the engine (observed as float here).
            This code assumes:
              - team_id == 2.0 -> red
              - otherwise      -> blue

            This mapping is domain-specific. It likely comes from your engine's
            representation where teams are encoded as numeric IDs in tensors.

        x, y:
            Grid coordinates where the agent died.

        killer_team_id_val:
            Numeric team identifier for the killer (or killer team).
            Same mapping rule.

        Why store strings "red"/"blue" in the log?
        ------------------------------------------
        Human readability + better analytics.
        Logs are often read by humans or simple scripts. Strings avoid confusion
        and match the constants used elsewhere.
        """
        self._dead_log.append(
            {
                "tick": self.tick,
                "agent_id": int(agent_id),
                "team": "red" if float(team_id_val) == 2.0 else "blue",
                "x": int(x),
                "y": int(y),
                "killer_team": "red" if float(killer_team_id_val) == 2.0 else "blue",
            }
        )

    def drain_dead_log(self) -> List[Dict[str, float | int]]:
        """
        Return the current death log and reset it to empty.

        Why "drain"?
        ------------
        In logging pipelines you often want:
        - collect events for a period
        - export them (write file / send to UI)
        - clear buffer
        so memory does not grow forever.

        This method supports exactly that.

        Implementation detail:
        ----------------------
        We return the old list object and replace self._dead_log with a new list.
        This avoids copying large logs and is efficient.
        """
        buf = self._dead_log
        self._dead_log = []
        return buf

    # Snapshots and delta rewards
    def snapshot(self) -> Snapshot:
        """
        Create and return a Snapshot of current counters.

        Snapshot contains cloned counters so it remains stable even when
        simulation continues.

        This is used for delta reward computation:
            delta_since(snap)
        """
        return Snapshot(self.red.clone(), self.blue.clone(), self.tick)

    def delta_since(self, snap: Snapshot) -> Dict[str, float]:
        """
        Compute score delta for each team since a snapshot.

        Returns a dict:
            {
              "red":  current_red_score  - snap.red.score,
              "blue": current_blue_score - snap.blue.score
            }

        This is the core utility for per-interval rewards.
        """
        return {
            TEAM_RED: self.red.score - snap.red.score,
            TEAM_BLUE: self.blue.score - snap.blue.score,
        }

    # Row export (CSV-friendly)
    def as_row(self) -> Dict[str, float]:
        """
        Return a flat dict representing the current stats, suitable for CSV logging.

        Why floats everywhere?
        ----------------------
        Many analytics pipelines (CSV writers, some parquet writers, quick plotting)
        work more smoothly if everything is numeric and consistent.

        Even though kills/deaths are integers, we cast them to float for uniformity.

        Typical usage:
        --------------
        - each tick (or every N ticks), call as_row()
        - append to a list
        - write to CSV / parquet later
        """
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
