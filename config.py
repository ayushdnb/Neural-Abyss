from __future__ import annotations

import os
import math
import torch

# =============================================================================
# 🛠️ ENV-PARSING UTILITIES (PRODUCTION-GRADE SECRETS)
# =============================================================================
# These helper functions safely parse environment variables. This architecture 
# allows you to run hyperparameter sweeps from bash scripts without ever 
# touching this Python file. 
# Example: FWS_MAX_AGENTS=2000 python main.py

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None: return float(default)
    try: return float(v)
    except Exception: return float(default)

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None: return int(default)
    try: return int(v)
    except Exception: return int(default)

def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None else str(v)

def _env_is_set(key: str) -> bool:
    return os.getenv(key) is not None

# =============================================================================
# 🧪 EXPERIMENT TRACKING & SEEDING
# =============================================================================
# Profiles act as master macros to override batches of settings at once.
PROFILE: str = _env_str("FWS_PROFILE", "default").strip().lower()

# Tag used by ResultsWriter to organize output directories for research logging.
EXPERIMENT_TAG: str = _env_str("FWS_EXPERIMENT_TAG", "god_level_run").strip()

# Master Random Seed. 
# 0 = Stochastic (True chaos, best for emergent behavior).
# 42 = Deterministic (Best for debugging specific crashes).
RNG_SEED: int = _env_int("FWS_SEED", 42)

RESULTS_DIR: str = _env_str("FWS_RESULTS_DIR", "results").strip()

#Optional: resume from a checkpoint (directory containing DONE+checkpoint.pt, or a direct checkpoint.pt path)
CHECKPOINT_PATH: str = _env_str("FWS_CHECKPOINT_PATH", "").strip()
# Autosave: set to >0 to autosave checkpoints while the viewer is running.
AUTOSAVE_EVERY_SEC = int(os.environ.get("FWS_AUTOSAVE_EVERY_SEC", "900"))  # default: 15 minutes
# =============================================================================
# 📡 SCIENTIFIC RECORDING / TELEMETRY (NON-INVASIVE, CONFIG-FIRST)
# =============================================================================
# Telemetry is OPTIONAL and must not change simulation outcomes.
# It writes append-only event logs + periodic snapshots for later analysis.

TELEMETRY_ENABLED: bool = _env_bool("FWS_TELEMETRY", True)

# Optional extra tag to distinguish telemetry-heavy runs from normal runs.
TELEMETRY_TAG: str = _env_str("FWS_TELEMETRY_TAG", "").strip()
TELEMETRY_SCHEMA_VERSION: str = _env_str("FWS_TELEM_SCHEMA", "2").strip()
TELEMETRY_WRITE_RUN_META: bool = _env_bool("FWS_TELEM_RUN_META", True)             # tiny + safe
TELEMETRY_WRITE_AGENT_STATIC: bool = _env_bool("FWS_TELEM_AGENT_STATIC", True)   # can grow; opt-in
TELEMETRY_TICK_SUMMARY_EVERY: int = _env_int("FWS_TELEM_TICK_SUMMARY_EVERY", 0)   # 0=off (opt-in)
# --- Frequencies (ticks) ---
TELEMETRY_TICK_METRICS_EVERY: int = _env_int("FWS_TELEM_TICK_EVERY", 200)          # per-tick summary rows
TELEMETRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_SNAPSHOT_EVERY", 100)        # grid occupancy snapshots
TELEMETRY_REGISTRY_SNAPSHOT_EVERY: int = _env_int("FWS_TELEM_REG_EVERY", 100)    # registry snapshots
TELEMETRY_VALIDATE_EVERY: int = _env_int("FWS_TELEM_VALIDATE_EVERY", 50)         # invariant checks
TELEMETRY_PERIODIC_FLUSH_EVERY: int = _env_int("FWS_TELEM_FLUSH_EVERY", 200)     # force flush (even if buffers not full)

# --- Buffers / chunk sizes (written as chunk files for atomicity) ---
TELEMETRY_EVENT_CHUNK_SIZE: int = _env_int("FWS_TELEM_EVENT_CHUNK", 50000)
TELEMETRY_TICK_CHUNK_SIZE: int = _env_int("FWS_TELEM_TICK_CHUNK", 5000)

# --- Event category toggles ---
TELEMETRY_LOG_BIRTHS: bool = _env_bool("FWS_TELEM_BIRTHS", True)
TELEMETRY_LOG_DEATHS: bool = _env_bool("FWS_TELEM_DEATHS", True)
TELEMETRY_LOG_DAMAGE: bool = _env_bool("FWS_TELEM_DAMAGE", True)
TELEMETRY_LOG_KILLS: bool = _env_bool("FWS_TELEM_KILLS", True)
TELEMETRY_LOG_MOVES: bool = _env_bool("FWS_TELEM_MOVES", False)   # can be huge
TELEMETRY_LOG_PPO: bool = _env_bool("FWS_TELEM_PPO", True)

# Damage logging mode:
# - "victim_sum": one row per victim per tick (smaller)
# - "per_hit": one row per attacker->victim (bigger)
TELEMETRY_DAMAGE_MODE: str = _env_str("FWS_TELEM_DMG_MODE", "victim_sum").strip().lower()

# Output formats (raw):
TELEMETRY_EVENTS_FORMAT: str = _env_str("FWS_TELEM_EVENTS_FMT", "jsonl").strip().lower()  # currently jsonl only
TELEMETRY_TICKS_FORMAT: str = _env_str("FWS_TELEM_TICKS_FMT", "csv").strip().lower()     # currently csv only
TELEMETRY_SNAPSHOT_FORMAT: str = _env_str("FWS_TELEM_SNAP_FMT", "npz").strip().lower()   # npz by default

# Validation strictness: 0=off | 1=basic | 2=strict
TELEMETRY_VALIDATE_LEVEL: int = _env_int("FWS_TELEM_VALIDATE", 1)
TELEMETRY_ABORT_ON_ANOMALY: bool = _env_bool("FWS_TELEM_ABORT", False)

# Reports (end-of-run)
TELEMETRY_REPORT_ENABLE: bool = _env_bool("FWS_TELEM_REPORT", True)
TELEMETRY_REPORT_EXCEL: bool = _env_bool("FWS_TELEM_EXCEL", True)
TELEMETRY_REPORT_PNG: bool = _env_bool("FWS_TELEM_PNG", True)

# =============================================================================
# 💻 HARDWARE ACCELERATION & TENSOR COMPILER
# =============================================================================
# These settings determine how the PyTorch backend talks to your silicon.

USE_CUDA = _env_bool("FWS_CUDA", True) and torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TORCH_DEVICE = DEVICE 

# Automatic Mixed Precision (AMP). 
# Uses FP16 for neural network matrix multiplications. This literally doubles 
# your throughput and halves VRAM usage. CRITICAL for 1000+ agents.
AMP_ENABLED = _env_bool("FWS_AMP", True)

def amp_enabled() -> bool: return AMP_ENABLED

TORCH_DTYPE = torch.float16 if (USE_CUDA and AMP_ENABLED) else torch.float32

# PyTorch VMAP (Vectorized Map).
USE_VMAP = _env_bool("FWS_USE_VMAP", True)
VMAP_MIN_BUCKET = _env_int("FWS_VMAP_MIN_BUCKET", 8)
VMAP_DEBUG = _env_bool("FWS_VMAP_DEBUG", False)

# =============================================================================
# 🌍 WORLD SCALE & MEMORY ALLOCATION
# =============================================================================

GRID_WIDTH  = _env_int("FWS_GRID_W", 100)
GRID_HEIGHT = _env_int("FWS_GRID_H", 100)

# START_AGENTS defines the initial drop. 500 per team = 1000 agents actively 
# computing raycasts and attention matrices simultaneously. 
START_AGENTS_PER_TEAM = _env_int("FWS_START_PER_TEAM", 200)

# MAX_AGENTS must be larger than START to accommodate the SoA (Struct of Arrays) 
# memory pre-allocation for reinforcements/respawns.
MAX_AGENTS  = _env_int("FWS_MAX_AGENTS", 700)

# 0 = run simulation math as fast as the CPU/GPU allows.
TICK_LIMIT = _env_int("FWS_TICK_LIMIT", 0)
TARGET_TPS = _env_int("FWS_TARGET_TPS", 0) 

# Strict schema width for the AgentRegistry. DO NOT CHANGE.
AGENT_FEATURES = 10 

# =============================================================================
# 🗺️ TOPOGRAPHY & STRATEGIC OBJECTIVES
# =============================================================================
# Topography defines the RL meta. Walls force pathfinding, zones force conflict.

# Increased walls to accommodate the massive 160x160 map. Creates distinct 
# "lanes" and "choke points" for tactical combat.
RANDOM_WALLS = _env_int("FWS_RAND_WALLS",9)
WALL_SEG_MIN = _env_int("FWS_WALL_SEG_MIN", 10)
WALL_SEG_MAX = _env_int("FWS_WALL_SEG_MAX", 47)
WALL_AVOID_MARGIN = _env_int("FWS_WALL_MARGIN", 3)

MAP_WALL_STRAIGHT_PROB = _env_float("FWS_MAP_WALL_STRAIGHT_PROB", 0.75)
MAP_WALL_GAP_PROB      = _env_float("FWS_MAP_WALL_GAP_PROB", 0.08)

# Heal Zones (The "Water holes"). Scaled up in count to support 1000 agents.
HEAL_ZONE_COUNT      = _env_int("FWS_HEAL_COUNT", 16)
HEAL_ZONE_SIZE_RATIO = _env_float("FWS_HEAL_SIZE_RATIO", 8/128)
HEAL_RATE            = _env_float("FWS_HEAL_RATE", 0.002)

# Capture Points (The "King of the Hill" objective).
CP_COUNT           = _env_int("FWS_CP_COUNT", 5)
CP_SIZE_RATIO      = _env_float("FWS_CP_SIZE_RATIO", 10/160)
CP_REWARD_PER_TICK = _env_float("FWS_CP_REWARD", 0.01)

# =============================================================================
# ⚔️ COMBAT BIOLOGY & CLASSES
# =============================================================================
UNIT_SOLDIER_ID = 1
UNIT_ARCHER_ID  = 2
UNIT_SOLDIER    = UNIT_SOLDIER_ID
UNIT_ARCHER     = UNIT_ARCHER_ID

# Higher HP yields longer Time-To-Kill (TTK). This is crucial for RL because it 
# allows the agent to witness the consequences of bad positioning and escape, 
# rather than instantly dying and learning nothing.
MAX_HP     = _env_float("FWS_MAX_HP", 1.0)
SOLDIER_HP = _env_float("FWS_SOLDIER_HP", 1.0)
ARCHER_HP  = _env_float("FWS_ARCHER_HP", 0.65) # Glass cannons

# Attack Values. 
BASE_ATK    = _env_float("FWS_BASE_ATK", 0.35)
SOLDIER_ATK = _env_float("FWS_SOLDIER_ATK", 0.35)
ARCHER_ATK  = _env_float("FWS_ARCHER_ATK", 0.20) 
MAX_ATK     = max(SOLDIER_ATK, ARCHER_ATK, BASE_ATK, 1e-6)

# Engagement Distances
BASE_RANGE   = _env_int("FWS_BASE_RANGE", 6)
ARCHER_RANGE = _env_int("FWS_ARCHER_RANGE", 4) # Gives archers extreme tactical advantage
ARCHER_LOS_BLOCKS_WALLS = _env_bool("FWS_ARCHER_BLOCK_LOS", True)

# =============================================================================
# 🔋 METABOLISM (Anti-Stalemate Mechanic)
# =============================================================================
# Agents bleed HP passively. If they hide in a corner doing nothing, they die.
# This forces exploration and confrontation over Heal Zones.
METABOLISM_ENABLED       = _env_bool("FWS_META_ON", True)
META_SOLDIER_HP_PER_TICK = _env_float("FWS_META_SOLDIER", 0.0009)
META_ARCHER_HP_PER_TICK  = _env_float("FWS_META_ARCHER",  0.0005)

# =============================================================================
# 👁️ SENSORS & INSTINCT (THE AI'S "EYES")
# =============================================================================
# Raycasting is the most CPU-heavy part of the simulation.
VISION_RANGE_SOLDIER = _env_int("FWS_VISION_SOLDIER", 10)
VISION_RANGE_ARCHER  = _env_int("FWS_VISION_ARCHER", 12) # Snipers need good eyes

VISION_RANGE_BY_UNIT = {
    UNIT_SOLDIER_ID: VISION_RANGE_SOLDIER,
    UNIT_ARCHER_ID:  VISION_RANGE_ARCHER,
}
RAYCAST_MAX_STEPS = max(max(VISION_RANGE_BY_UNIT.values()), 1)
RAY_MAX_STEPS     = RAYCAST_MAX_STEPS

# Instinct detects "Flanking" mathematically via local unit density.
INSTINCT_RADIUS = _env_int("FWS_INSTINCT_RADIUS", 14)

# =============================================================================
# 🧩 TENSOR OBSERVATION LAYOUT (STRICT CONTRACT)
# =============================================================================
# WARNING: Changing these changes the input layer of the Transformer.
# 32 rays is the sweet spot. 64 rays drops FPS heavily at 1000 agents.
RAY_TOKEN_COUNT = _env_int("FWS_RAY_TOKENS", 32)
RAY_FEAT_DIM    = 8
RAYS_FLAT_DIM   = RAY_TOKEN_COUNT * RAY_FEAT_DIM

RICH_BASE_DIM   = 23 
INSTINCT_DIM    = 4  
RICH_TOTAL_DIM  = RICH_BASE_DIM + INSTINCT_DIM

OBS_DIM = RAYS_FLAT_DIM + RICH_TOTAL_DIM

SEMANTIC_RICH_BASE_INDICES = {
    "own_context":    (0, 1, 2, 5, 6, 7, 8),
    "world_context":  (11, 20, 21, 22),
    "zone_context":   (9, 10),
    "team_context":   (3, 4, 12, 13, 14, 15),
    "combat_context": (16, 17, 18, 19),
}

SEMANTIC_TOKEN_ORDER = (
    "own_context", "world_context", "zone_context", "team_context", "combat_context", "instinct_context"
)

NUM_ACTIONS = _env_int("FWS_NUM_ACTIONS", 41)

# =============================================================================
# 🔄 POPULATION CONTROL (REINFORCEMENTS)
# =============================================================================
RESPAWN_ENABLED = _env_bool("FWS_RESPAWN", True)

# New Wave-Based Respawn logic.
RESP_FLOOR_PER_TEAM      = _env_int("FWS_RESP_FLOOR_PER_TEAM", 190) # Never let a team drop below 300
RESP_MAX_PER_TICK        = _env_int("FWS_RESP_MAX_PER_TICK", 15)    # Smooth out spawn-lag
RESP_PERIOD_TICKS        = _env_int("FWS_RESP_PERIOD_TICKS", 600)   # Reinforcement chopper arrives every 600 ticks
RESP_PERIOD_BUDGET       = _env_int("FWS_RESP_PERIOD_BUDGET", 40)   # Drops 40 agents 
RESP_HYST_COOLDOWN_TICKS = _env_int("FWS_RESP_HYST_COOLDOWN_TICKS", 45) 
RESP_WALL_MARGIN         = _env_int("FWS_RESP_WALL_MARGIN", 2)

SPAWN_ARCHER_RATIO       = _env_float("FWS_SPAWN_ARCHER_RATIO", 0.35) 

# Legacy fallbacks
RESPAWN_PROB_PER_DEAD        = _env_float("FWS_RESPAWN_PROB", 0.05)
RESPAWN_SPAWN_TRIES          = _env_int("FWS_RESPAWN_TRIES", 200)
RESPAWN_MUTATION_STD         = _env_float("FWS_MUT_STD", 0.05) # Increased mutation volatility for better evolution
RESPAWN_CLONE_PROB           = _env_float("FWS_CLONE_PROB", 0.70) # Highly favor cloning successful survivors
RESPAWN_USE_TEAM_ELITE       = _env_bool("FWS_TEAM_ELITE", True)
RESPAWN_RESET_OPT_ON_RESPAWN = _env_bool("FWS_RESET_OPT", True)
RESPAWN_JITTER_RADIUS        = _env_int("FWS_RESP_JITTER", 3)
RESPAWN_COOLDOWN_TICKS       = _env_int("FWS_RESPAWN_CD", 500)
RESPAWN_BATCH_PER_TEAM       = _env_int("FWS_RESPAWN_BATCH", 1)
RESPAWN_ARCHER_SHARE         = _env_float("FWS_RESPAWN_ARCHER_SHARE", 0.50)
RESPAWN_INTERIOR_BIAS        = _env_float("FWS_RESPAWN_INTERIOR_BIAS", 0.40)

# =============================================================================
# 🏆 REWARD SHAPING (RL Feedback Loop)
# =============================================================================
# This defines the "Win Condition" for the agents.

TEAM_KILL_REWARD       = _env_float("FWS_REW_KILL",       1.0)
TEAM_DMG_DEALT_REWARD  = _env_float("FWS_REW_DMG_DEALT",  0.00)
TEAM_DEATH_PENALTY     = _env_float("FWS_REW_DEATH",     -1.0) # Increased penalty for dying
TEAM_DMG_TAKEN_PENALTY = _env_float("FWS_REW_DMG_TAKEN",  0.00)

PPO_REWARD_HP_TICK         = _env_float("FWS_PPO_REW_HP_TICK", 0.3)     # Strong incentive to find healers
PPO_REWARD_KILL_INDIVIDUAL = _env_float("FWS_PPO_REW_KILL_AGENT", 8.0)  # Massive dopamine hit for executing enemies
PPO_REWARD_DEATH           = _env_float("FWS_PPO_REW_DEATH", -3.0)      # Massive punishment
PPO_REWARD_CONTESTED_CP    = _env_float("FWS_PPO_REW_CONTEST", 0.05)     # Objective play incentive

# =============================================================================
# 🧠 REINFORCEMENT LEARNING (PROXIMAL POLICY OPTIMIZATION)
# =============================================================================
PPO_ENABLED       = _env_bool("FWS_PPO_ENABLED", True)
PPO_RESET_LOG     = _env_bool("FWS_PPO_RESET_LOG", True)
PPO_WINDOW_TICKS  = _env_int("FWS_PPO_TICKS", 512) 

PPO_LR            = _env_float("FWS_PPO_LR", 3e-4) 
PPO_LR_T_MAX      = _env_int("FWS_PPO_T_MAX", 10_000_000) # Extended decay for long runs
PPO_LR_ETA_MIN    = _env_float("FWS_PPO_ETA_MIN", 1e-6) 

PPO_CLIP          = _env_float("FWS_PPO_CLIP", 0.2)
PPO_CLIP_EPS      = PPO_CLIP 
PPO_ENTROPY_COEF  = _env_float("FWS_PPO_ENTROPY", 0.02) # Boosted slightly to force 1000 agents to try diverse tactics
PPO_VALUE_COEF    = _env_float("FWS_PPO_VCOEF", 0.5)    
PPO_EPOCHS        = _env_int("FWS_PPO_EPOCHS", 4)       # Increased epochs for better sample efficiency
PPO_MINIBATCHES   = _env_int("FWS_PPO_MINIB", 8)
PPO_MAX_GRAD_NORM = _env_float("FWS_PPO_MAXGN", 0.5)    # Tighter gradient clipping for stability in massive crowds
PPO_TARGET_KL     = _env_float("FWS_PPO_TKL", 0.02)    # KL early-stop threshold (set 0 to disable)
PPO_GAMMA         = _env_float("FWS_PPO_GAMMA", 0.995)  # Make agents care slightly more about the future
PPO_LAMBDA        = _env_float("FWS_PPO_LAMBDA", 0.95)  
PPO_UPDATE_TICKS  = _env_int("FWS_PPO_UPDATE_TICKS", 5)

PPO_ENTROPY_BONUS = _env_float("FWS_PPO_ENTROPY_BONUS", PPO_ENTROPY_COEF)
if _env_is_set("FWS_PPO_ENTROPY_BONUS") and not _env_is_set("FWS_PPO_ENTROPY"):
    PPO_ENTROPY_COEF = float(PPO_ENTROPY_BONUS)

PER_AGENT_BRAINS        = _env_bool("FWS_PER_AGENT_BRAINS", True) 
MUTATION_PERIOD_TICKS   = _env_int("FWS_MUTATE_EVERY", 8000)
MUTATION_FRACTION_ALIVE = _env_float("FWS_MUTATE_FRAC", 0.02)

# =============================================================================
# 🤖 BRAIN ARCHITECTURE (THE NEURAL ENGINES)
# =============================================================================
# The experiment: Red (Tron/Reactive) vs Blue (Mirror/Reflective)
BRAIN_KIND: str = os.getenv("FWS_BRAIN", "tron").strip().lower()
TEAM_BRAIN_ASSIGNMENT: bool = _env_bool("FWS_TEAM_BRAIN_ASSIGNMENT", True)
# ------------------------------------------------------------------
# 🧠 Team brain assignment mode (keeps old behavior by default)
# ------------------------------------------------------------------
# TEAM_BRAIN_ASSIGNMENT=True controls whether we do *any* team-aware logic.
#
# TEAM_BRAIN_ASSIGNMENT_MODE:
#   - "exclusive" (default): old behavior (Red=Tron, Blue=Mirror)
#   - "mix": each team can spawn BOTH architectures (use strategy below)
TEAM_BRAIN_ASSIGNMENT_MODE: str = _env_str("FWS_TEAM_BRAIN_MODE", "mix").strip().lower()

# TEAM_BRAIN_MIX_STRATEGY:
#   - "alternate" (default): deterministic 50/50 per team (tron, mirror, tron, mirror...)
#   - "random": probabilistic mix using TEAM_BRAIN_MIX_P_TRON
TEAM_BRAIN_MIX_STRATEGY: str = _env_str("FWS_TEAM_BRAIN_MIX_STRATEGY", "alternate").strip().lower()

# Only used when TEAM_BRAIN_MIX_STRATEGY == "random"
# Probability that a newly created brain is Tron (else Mirror).
TEAM_BRAIN_MIX_P_TRON: float = _env_float("FWS_TEAM_BRAIN_MIX_P_TRON", 0.5)

# Seed for ONLY the brain-mix RNG (so it doesn't affect map spawning RNG).
# If you already have RNG_SEED in config, we reuse it; otherwise default 0.
TEAM_BRAIN_MIX_SEED: int = _env_int("FWS_TEAM_BRAIN_MIX_SEED", int(globals().get("RNG_SEED", 0)))

# --- Transformer Hyperparameters ---
# Pushed up for high "IQ". d_model=128 + 8 heads allows complex multi-modal 
# processing of spatial rays and tactical tokens.
TRON_D_MODEL       = _env_int("FWS_TRON_DMODEL", 64)  # High capacity memory
TRON_HEADS         = _env_int("FWS_TRON_HEADS", 2)     # Parallel reasoning paths
TRON_DROPOUT       = _env_float("FWS_TRON_DROPOUT", 0.05)
TRON_RAY_LAYERS    = _env_int("FWS_TRON_RAY_LAYERS", 2) 
TRON_SEM_LAYERS    = _env_int("FWS_TRON_SEM_LAYERS", 2) 
TRON_FUSION_LAYERS = _env_int("FWS_TRON_FUSION_LAYERS", 2) 
TRON_MLP_HIDDEN    = _env_int("FWS_TRON_MLP_HID", 384) # Wide feed-forward
TRON_USE_ROPE      = _env_bool("FWS_TRON_ROPE", True)  
TRON_USE_PRENORM   = _env_bool("FWS_TRON_PRENORM", True)

# =============================================================================
# 🪞 MIRROR TRANSFORMER HYPERPARAMETERS
# =============================================================================
# MirrorBrain currently exists and is used in team brain assignment.
# These knobs allow MirrorBrain to have an *independent* architecture from TronBrain.
#
# SAFETY RULE:
# - Every default mirrors the TRON_* value so nothing changes unless you override FWS_MIRROR_*.
# - Keep MIRROR_D_MODEL divisible by MIRROR_HEADS.

# Internal hidden size for MirrorBrain. Larger -> more capacity, more VRAM.
MIRROR_D_MODEL       = _env_int("FWS_MIRROR_DMODEL", int(TRON_D_MODEL))

# Number of attention heads. Higher -> more parallel attention, but d_model must be divisible by heads.
MIRROR_HEADS         = _env_int("FWS_MIRROR_HEADS", int(TRON_HEADS))

# Dropout for MirrorBrain (if/when used). Higher -> more regularization, less overfitting, slightly noisier training.
MIRROR_DROPOUT       = _env_float("FWS_MIRROR_DROPOUT", float(TRON_DROPOUT))

# Self-attention depth over ray tokens (spatial perception).
MIRROR_RAY_LAYERS    = _env_int("FWS_MIRROR_RAY_LAYERS", int(TRON_RAY_LAYERS))

# Self-attention depth over plan/semantic tokens (tactical reasoning).
MIRROR_SEM_LAYERS    = _env_int("FWS_MIRROR_SEM_LAYERS", int(TRON_SEM_LAYERS))

# Cross-attention depth fusing plan tokens with ray tokens (spatial+tactical fusion).
MIRROR_FUSION_LAYERS = _env_int("FWS_MIRROR_FUSION_LAYERS", int(TRON_FUSION_LAYERS))

# MLP size used in policy/value heads (capacity of final decision head).
MIRROR_MLP_HIDDEN    = _env_int("FWS_MIRROR_MLP_HID", int(TRON_MLP_HIDDEN))

# MirrorBrain uses the same positional/normalization strategy toggles as Tron by default.
MIRROR_USE_ROPE      = _env_bool("FWS_MIRROR_ROPE", bool(TRON_USE_ROPE))
MIRROR_USE_PRENORM   = _env_bool("FWS_MIRROR_PRENORM", bool(TRON_USE_PRENORM))

# =============================================================================
# 🖥️ UI, VIEWER & SCREEN RECORDING (SMOOTH 60FPS)
# =============================================================================
ENABLE_UI  = _env_bool("FWS_UI", True)

# --- The De-coupling Magic ---
# Extracting 1000 agents' data from the GPU to the CPU every single tick will 
# freeze PyGame. By setting REFRESH_EVERY to 3, the backend RL environment 
# races ahead at max speed, while the UI only fetches visual data every 3rd frame. 
# This guarantees smooth 60fps screen recordings while the PC is at 100% load.
VIEWER_STATE_REFRESH_EVERY = _env_int("FWS_VIEWER_STATE_REFRESH_EVERY", 3) 
VIEWER_PICK_REFRESH_EVERY  = _env_int("FWS_VIEWER_PICK_REFRESH_EVERY", 3)

UI_FONT_NAME: str = _env_str("FWS_UI_FONT", "consolas")
VIEWER_CENTER_WINDOW: bool = _env_bool("FWS_VIEWER_CENTER_WINDOW", True)

# Adjusted cell size so a 160x160 map fits beautifully on a 1080p/1440p monitor (800x800 map pixels).
CELL_SIZE  = _env_int("FWS_CELL_SIZE", 5) 
HUD_WIDTH  = _env_int("FWS_HUD_W", 340) # Slightly wider HUD for better text clarity
TARGET_FPS = _env_int("FWS_TARGET_FPS", 60) # Buttery smooth for screen capping

# Optional direct-to-disk recording
RECORD_VIDEO: bool     = _env_bool("FWS_RECORD_VIDEO", False)
VIDEO_FPS: int         = _env_int("FWS_VIDEO_FPS", 60)
VIDEO_SCALE: int       = _env_int("FWS_VIDEO_SCALE", 4)
VIDEO_EVERY_TICKS: int = _env_int("FWS_VIDEO_EVERY_TICKS", 1)

# Visual Aesthetics
UI_COLORS = {
    "bg": (15, 17, 22), "hud_bg": (10, 12, 16), "side_bg": (14, 16, 20), # Deepened blacks for contrast
    "grid": (35, 37, 42), "border": (80, 85, 95), "wall": (100, 105, 115),
    "empty": (20, 22, 28),

    # Vivid team colors for recording clarity
    "red_soldier": (240, 50, 50), "red_archer":  (255, 120, 0), "red": (240, 50, 50),
    "blue_soldier": (30, 160, 255), "blue_archer":  (0, 220, 180), "blue": (30, 160, 255),

    "archer_glyph": (255, 245, 120), "marker": (255, 255, 255),
    "text": (240, 240, 245), "text_dim": (160, 165, 175),
    "green": (50, 220, 120), "warn": (255, 170, 0),
    "bar_bg": (30, 35, 40), "bar_fg": (50, 220, 120),
    "graph_red": (240, 50, 50, 180), "graph_blue": (30, 160, 255, 180),
    "graph_grid": (50, 50, 60), "pause_text": (255, 200, 50)
}

# =============================================================================
# 🛠️ PROFILE OVERRIDE INJECTION
# =============================================================================
def _apply_profile_overrides() -> None:
    if PROFILE == "default":
        return

    presets = {
        "debug": [
            ("FWS_GRID_W", "GRID_WIDTH", 80),
            ("FWS_GRID_H", "GRID_HEIGHT", 80),
            ("FWS_START_PER_TEAM", "START_AGENTS_PER_TEAM", 30),
            ("FWS_MAX_AGENTS", "MAX_AGENTS", 160),
            ("FWS_RAND_WALLS", "RANDOM_WALLS", 6),
            ("FWS_UI", "ENABLE_UI", True),
            ("FWS_TARGET_FPS", "TARGET_FPS", 30),
            ("FWS_RECORD_VIDEO", "RECORD_VIDEO", False),
            ("FWS_USE_VMAP", "USE_VMAP", False),
        ],
        "train_fast": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_TRON_DMODEL", "TRON_D_MODEL", 96),
            ("FWS_TRON_HEADS", "TRON_HEADS", 4),
            ("FWS_TRON_RAY_LAYERS", "TRON_RAY_LAYERS", 2),
            ("FWS_TRON_SEM_LAYERS", "TRON_SEM_LAYERS", 2),
            ("FWS_TRON_FUSION_LAYERS", "TRON_FUSION_LAYERS", 1),
            ("FWS_TRON_MLP_HID", "TRON_MLP_HIDDEN", 192),
        ],
        "train_quality": [
            ("FWS_UI", "ENABLE_UI", False),
            ("FWS_USE_VMAP", "USE_VMAP", True),
            ("FWS_TRON_DMODEL", "TRON_D_MODEL", 192),
            ("FWS_TRON_HEADS", "TRON_HEADS", 6),
            ("FWS_TRON_RAY_LAYERS", "TRON_RAY_LAYERS", 4),
            ("FWS_TRON_SEM_LAYERS", "TRON_SEM_LAYERS", 4),
            ("FWS_TRON_FUSION_LAYERS", "TRON_FUSION_LAYERS", 2),
            ("FWS_TRON_MLP_HID", "TRON_MLP_HIDDEN", 384),
        ],
    }

    rows = presets.get(PROFILE)
    if not rows: return

    g = globals()
    for env_key, var_name, value in rows:
        if not _env_is_set(env_key):
            g[var_name] = value

    if int(g.get("TRON_D_MODEL", 0)) % max(int(g.get("TRON_HEADS", 1)), 1) != 0:
        raise ValueError(f"TRON_D_MODEL must be divisible by TRON_HEADS (got {g.get('TRON_D_MODEL')} / {g.get('TRON_HEADS')})")

_apply_profile_overrides()

def summary_str() -> str:
    return (
        f"[final_war_sim: GOD LEVEL] "
        f"dev={DEVICE.type} "
        f"grid={GRID_WIDTH}x{GRID_HEIGHT} "
        f"start={START_AGENTS_PER_TEAM}/team "
        f"obs={OBS_DIM} acts={NUM_ACTIONS} "
        f"AMP={'on' if AMP_ENABLED else 'off'}"
    )