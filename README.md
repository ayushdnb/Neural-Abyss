Neural Siege
Large-scale multi-agent combat simulation with neural policy optimization.
Overview
Neural Siege is a high-performance, grid-based multi-agent simulation framework designed for research in emergent behaviors, multi-agent reinforcement learning, and large-scale combat dynamics. The system simulates team-based warfare between autonomous agents controlled by neural network policies, trained via Proximal Policy Optimization (PPO).
Design Philosophy
Vectorized Execution: All core simulation logic operates on PyTorch tensors, enabling GPU acceleration and batch processing of thousands of agents.
Deterministic Reproducibility: Full checkpointing support with RNG state preservation ensures experiments can be resumed and replicated.
Modular Architecture: Swappable brain architectures, configurable map generation, and pluggable telemetry systems.
Observability: Comprehensive telemetry, interactive visualization, and real-time inspection capabilities.
Key Features
Table
Feature	Description
Tensor-Based Simulation	Grid and agent state stored as PyTorch tensors; operations vectorized across all agents
Multi-Agent PPO	Clipped surrogate objective with entropy regularization, generalized advantage estimation (GAE)
Modular Brain Architectures	TronBrain (MLP), MirrorBrain (symmetric), TransformerBrain (self-attention)
Procedural Map Generation	Random walk-based terrain with heal zones and control points
Interactive Visualization	Pygame-based viewer with camera controls, agent inspection, and overlay modes
Comprehensive Telemetry	Event logging (JSONL), life-cycle tracking, lineage graphs, movement analytics
Atomic Checkpointing	Crash-safe save/resume with CPU-portable tensor serialization
Background Persistence	Multiprocess-based results writer to avoid simulation stalls
System Architecture
plain
Copy
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SIMULATION LOOP                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Map Gen    │───▶│  Registry   │───▶│   Engine    │───▶│   Viewer    │   │
│  │  (CPU)      │    │  (GPU)      │    │  (GPU)      │    │  (CPU/GPU)  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                              │                                                │
│                              ▼                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │   PPO       │◄───│  Brains     │◄───│  Actions    │                       │
│  │  Trainer    │    │  (NN Modules)│    │  (Discrete) │                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
│                              │                                                │
│                              ▼                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │ Telemetry   │◄───│ Checkpoints │◄───│  Results    │                       │
│  │  (JSONL)    │    │  (Atomic)   │    │  (CSV)      │                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
Data Flow
Initialization: MapGenerator creates terrain; AgentRegistry allocates agent slots with unique IDs
Observation: ObservationBuilder constructs per-agent views (self + nearby allies/enemies)
Inference: Brain networks (batched via torch.vmap) output action logits and state values
Action Resolution: ActionResolver processes moves, attacks, and collisions; updates grid state
Reward Calculation: Team-based rewards with individual penalties (damage taken, idle time)
PPO Update: Trajectory buffer flushed; policy and value networks updated periodically
Telemetry: Events batched and written to disk via background process
Core Components
Engine (engine/)
Table
Module	Responsibility
tick_engine.py	Core simulation loop: observation → inference → action → reward
action_resolver.py	Discrete action processing: 8-direction movement, attack, noop
observations.py	Per-agent observation construction with spatial and entity features
agent_registry.py	Slot-based agent management with tensor storage (alive, HP, position, team)
mapgen.py	Procedural terrain generation: walls, heal zones, control points
respawn.py	Periodic agent respawning with team cooldowns and spawn-point selection
Agent Brains (agent/)
Table
Brain	Architecture	Parameters	Use Case
TronBrain	3-layer MLP	~50K	Baseline policy, fast inference
MirrorBrain	Symmetric twin networks	~100K	Red/Blue policy sharing
TransformerBrain	Multi-head self-attention	~500K	Relational reasoning, attention visualization
All brains implement:
forward(obs) -> (action_logits, state_value)
get_action_and_value(obs, action=None) -> (action, log_prob, entropy, value)
Training (training/)
Table
Module	Function
ppo_trainer.py	Multi-agent PPO with GAE, entropy bonus, gradient clipping
reward.py	Reward shaping: team rewards, individual penalties, death penalties
PPO Hyperparameters (configurable):
Learning rate: 3e-4
Gamma (discount): 0.99
GAE lambda: 0.95
Clip epsilon: 0.2
Value loss coefficient: 0.5
Entropy coefficient: 0.01
Configuration (config.py)
Environment-based configuration system supporting:
Device selection (TORCH_DEVICE)
Grid dimensions (GRID_WIDTH, GRID_HEIGHT)
Agent counts (NUM_AGENTS, MAX_AGENTS)
PPO hyperparameters
Telemetry toggles
Checkpoint intervals
Profile selection via FWS_PROFILE environment variable:
bash
Copy
FWS_PROFILE=train_fast python main.py
FWS_PROFILE=visualize python main.py
Telemetry (utils/telemetry.py)
Event-driven telemetry with configurable verbosity:
Agent Life: Birth/death timestamps, lifespan, kills, damage totals
Lineage Edges: Parent-child relationships for evolutionary analysis
Events (JSONL): Birth, death, damage, kill, move events with schema versioning
Movement Summary: Per-tick aggregates (attempted, success, blocked, conflicts)
Tick Summary: Low-frequency population and health metrics
Checkpointing (utils/checkpointing.py)
Atomic checkpoint format:
plain
Copy
checkpoints/
├── latest.txt              # Pointer to most recent checkpoint
└── ckpt_t{tick}_{timestamp}/
    ├── checkpoint.pt       # Full state (grid, registry, brains, PPO, RNG)
    ├── manifest.json       # Metadata (tick, git commit, notes)
    └── DONE                # Atomic completion marker
Checkpoint includes:
World grid and zone masks
Agent data tensor and brain state dicts
PPO trainer state (optimizer, trajectory buffer)
RNG states (Python, NumPy, PyTorch CPU/CUDA)
Respawn controller state
Viewer (ui/viewer.py)
Interactive Pygame visualization:
Camera: Pan (WASD), zoom (scroll), adaptive cell sizing
Overlays: HP bars, brain type labels, threat vision, line-of-sight rays
Inspection: Click agent for detailed stats (HP, attack, vision, brain params)
HUD: Team scores, kill/death counts, control point status, score history graph
Minimap: World overview with viewport rectangle
Controls: Pause, speed adjustment (0.25x–16x), manual checkpoint (F9)
Technical Highlights
Vectorized Observation Construction
Observations built via tensor operations (no Python loops):
Python
Copy
# Relative positions computed via broadcasting
rel_x = other_x - self_x  # shape: (num_alive, num_alive)
rel_y = other_y - self_y
Batched Neural Inference
Brain inference parallelized via torch.vmap:
Python
Copy
logits, values = torch.vmap(self.brain_forward)(obs_tensor)
GPU-Optimized Action Resolution
Collision detection and position updates on GPU:
Wall collision: boolean mask from grid occupancy
Agent collision: hash-based position deduplication
Conflict resolution: deterministic priority by agent ID
Memory-Efficient Trajectory Storage
PPO buffer stores minimal state:
Observations (float32)
Actions (int64)
Log probabilities (float32)
Rewards (float32)
Values (float32)
Dones (bool)
Deterministic Resume
RNG state captured/restored for reproducibility:
Python random module state
NumPy random state
PyTorch CPU and CUDA RNG states
Installation & Setup
Requirements
Python 3.10+
PyTorch 2.0+ (with CUDA optional)
pygame 2.5+
numpy 1.24+
Install
bash
Copy
git clone <repository>
cd Infinite_War_Simulation
pip install -r requirements.txt
Verify Environment
bash
Copy
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
Usage
Training Mode
bash
Copy
# Fast training on GPU
FWS_PROFILE=train_fast python main.py

# With custom config
TORCH_DEVICE=cuda:0 NUM_AGENTS=256 python main.py
Visualization Mode
bash
Copy
# Interactive viewer with trained policy
FWS_PROFILE=visualize python main.py

# Load from checkpoint
python main.py --resume results/sim_2025-01-15_10-30-00/checkpoints/latest.txt
Headless Mode
bash
Copy
# No viewer, telemetry only
FWS_PROFILE=train_fast VIEWER_ENABLED=false python main.py
Expected Outputs
plain
Copy
results/
└── sim_YYYY-MM-DD_HH-MM-SS/
    ├── config.json              # Runtime configuration
    ├── stats.csv                # Per-tick metrics
    ├── dead_agents_log.csv      # Death events
    ├── checkpoints/             # Periodic snapshots
    │   ├── latest.txt
    │   └── ckpt_t{tick}_{timestamp}/
    └── telemetry/               # Detailed event logs
        ├── agent_life.csv
        ├── lineage_edges.csv
        ├── move_summary.csv
        └── events/
            └── events_000001.jsonl
Configuration
Key environment variables:
Table
Variable	Default	Description
TORCH_DEVICE	cuda	Compute device (cuda, cpu, mps)
GRID_WIDTH	64	World width in cells
GRID_HEIGHT	64	World height in cells
NUM_AGENTS	128	Initial agents per team
MAX_AGENTS	512	Maximum concurrent agents
PPO_UPDATE_EVERY	2048	Steps between policy updates
TELEMETRY_ENABLED	true	Enable event logging
CHECKPOINT_EVERY_TICKS	10000	Auto-save interval
See config.py for complete parameter list.
Design Principles
Reproducibility
All randomness controlled via seedable RNGs
Checkpoint system preserves full runtime state
Git commit hash recorded in checkpoints
Modularity
Brains implement standard interface; new architectures plug in seamlessly
Observation builder composable from feature modules
Telemetry events schema-versioned for backward compatibility
Performance
GPU-resident simulation state minimizes CPU-GPU transfers
Background I/O via multiprocessing Queue
Viewer caches static terrain; dynamic elements updated per-frame
Stability
Runtime sanity checks detect NaN/Inf in grid and agent data
Gradient clipping in PPO prevents policy collapse
Checkpoint atomic writes prevent corruption on crash
Limitations
Discrete Actions Only: Action space limited to 9 discrete actions (8 directions + noop + attack)
Grid-Based Movement: Agents occupy single cells; no continuous position or fractional movement
2D Top-Down: No elevation, line-of-sight blocked only by walls
Simplified Combat: Attack is instantaneous with fixed range; no projectile physics
Homogeneous Teams: All agents on a team share the same brain architecture (though individual parameters differ)
Single-Device Training: PPO updates assume all agents on same GPU; no distributed training support
License
MIT License
Copyright (c) 2025 Neural Siege Contributors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Acknowledgments
This project was developed as a research platform for multi-agent reinforcement learning. The architecture draws inspiration from vectorized simulation frameworks and modern PPO implementations. The codebase prioritizes clarity, extensibility, and reproducibility for academic and industrial research applications.
