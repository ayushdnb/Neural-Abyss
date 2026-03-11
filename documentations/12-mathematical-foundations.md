# Mathematical Foundations

This document gives a formal view of the inspected repository from first principles.

It separates:

1. repository-implemented mathematics
2. standard background mathematics that is needed to read the code

It does **not** claim that every mathematical object below is explicitly materialized in source in exactly this notation. Where a formula is background or a compact abstraction of the code, that is stated plainly.

## Notation table

| Symbol | Meaning |
|---|---|
| \(t\) | discrete tick index |
| \(H, W\) | grid height and width |
| \(s\) | slot index in the registry |
| \(i\) | live-agent index inside an alive subset |
| \(x_s, y_s\) | slot position |
| \(h_s\) | current HP |
| \(h^{\max}_s\) | maximum HP |
| \(u_s\) | unit type |
| \(\tau_s\) | team id |
| \(o_t^{(s)}\) | observation for slot \(s\) at tick \(t\) |
| \(a_t^{(s)}\) | sampled action |
| \(\pi_{\theta_s}\) | slot-local policy for slot \(s\) |
| \(V_{\theta_s}\) | slot-local value function for slot \(s\) |
| \(r_t^{(s)}\) | PPO reward assigned to slot \(s\) for tick \(t\) |
| \(Z_{\text{base}}\) | canonical signed base-zone field |
| \(Z_{\text{override}}\) | catastrophe override field |
| \(M_{\text{apply}}\) | catastrophe apply mask |
| \(Z_{\text{eff}}\) | runtime-effective zone field |
| \(\gamma\) | PPO discount factor |
| \(\lambda\) | GAE parameter |
| \(\varepsilon\) | PPO clip coefficient |

A terminology cross-check appears in [Glossary and notation](15-glossary-and-notation.md).

## 1. State model

### Repository-implemented abstraction

The world state can be viewed as:

\[
\mathcal{W}_t = \bigl(G_t,\; R_t,\; Z_t,\; C_t,\; S_t\bigr),
\]

where:

- \(G_t\) is the grid tensor
- \(R_t\) is the registry tensor plus per-slot brain ownership
- \(Z_t\) is the zone state
- \(C_t\) is catastrophe controller state
- \(S_t\) is cumulative simulation statistics

This is a compact abstraction of several Python objects. It is not a single serialized runtime object in code, but it matches the inspected architecture.

### Grid view

The tick engine documents a grid tensor with channels:

\[
G_t \in \mathbb{R}^{3 \times H \times W}.
\]

Conceptually:

- \(G_t[0, y, x]\): occupancy / terrain marker
- \(G_t[1, y, x]\): HP channel
- \(G_t[2, y, x]\): slot-id channel

### Registry view

Let the registry capacity be \(N_{\max}\). Then the dense slot tensor can be represented as:

\[
R^{\text{data}}_t \in \mathbb{R}^{N_{\max} \times d_{\text{agent}}},
\]

with \(d_{\text{agent}} = 10\) in the inspected config.

## 2. Observation model

### Repository-implemented structure

The current observation width is:

\[
d_{\text{obs}} = 32 \cdot 8 + 23 + 4 = 283.
\]

For a live slot \(s\),

\[
o_t^{(s)} =
\begin{bmatrix}
o^{(s)}_{\text{ray}} \\
o^{(s)}_{\text{rich}} \\
o^{(s)}_{\text{instinct}}
\end{bmatrix},
\]

where:

- \(o^{(s)}_{\text{ray}} \in \mathbb{R}^{256}\)
- \(o^{(s)}_{\text{rich}} \in \mathbb{R}^{23}\)
- \(o^{(s)}_{\text{instinct}} \in \mathbb{R}^{4}\)

### Rich-base local zone semantics

Schema version 2 uses a signed local zone scalar:

\[
o^{(s)}_{\text{rich},9} \in [-1, 1],
\]

rather than the earlier boolean heal-local indicator.

That detail matters because checkpoint compatibility and policy interpretation depend on feature meaning, not only on vector width.

## 3. Brain-side preprocessing

### Repository-implemented math

The inspected MLP family does not consume the flat observation directly. It first splits the observation into:

- 32 ray vectors of width 8
- one rich vector of width 27

Let the ray vectors be \(r_1,\dots,r_{32} \in \mathbb{R}^8\), and let the rich vector be \(q \in \mathbb{R}^{27}\).

The shared preprocessing path is:

\[
e_i = W_r \,\mathrm{LN}(r_i) + b_r \in \mathbb{R}^{D},
\]

\[
e_{\text{ray}} = \frac{1}{32}\sum_{i=1}^{32} e_i,
\]

\[
e_{\text{rich}} = W_q \,\mathrm{LN}(q) + b_q \in \mathbb{R}^{D},
\]

\[
x = \mathrm{Norm}\left(
\begin{bmatrix}
e_{\text{ray}} \\
e_{\text{rich}}
\end{bmatrix}
\right) \in \mathbb{R}^{2D}.
\]

This is directly aligned with `agent/mlp_brain.py`.

### Actor-critic heads

For trunk output \(h\),

\[
\ell = W_{\pi} h + b_{\pi},
\qquad
v = W_V h + b_V.
\]

Here:

- \(\ell\) is the action-logit vector
- \(v\) is the scalar critic value

## 4. Action model

### Repository-implemented discrete action space

The default action count is:

\[
|\mathcal{A}| = 41.
\]

The action set can be partitioned as:

\[
\mathcal{A} =
\{0\}
\cup
\mathcal{A}_{\text{move}}
\cup
\mathcal{A}_{\text{attack}},
\]

with:

- idle action \(0\)
- 8 movement actions
- 32 directional-range attack actions

For attack action \(a \ge 9\),

\[
\text{range}(a) = ((a - 9) \bmod 4) + 1,
\]

\[
\text{dir}(a) = \left\lfloor \frac{a - 9}{4} \right\rfloor.
\]

That is not background theory; it is the actual decoding used in `engine/tick.py`.

## 5. Transition model

### High-level state transition

A compact formalization is:

\[
\mathcal{W}_{t+1} = T(\mathcal{W}_t, a_t),
\]

where \(T\) is the composition of the tick phases:

\[
T =
T_{\text{combat}}
\circ
T_{\text{death}}
\circ
T_{\text{move}}
\circ
T_{\text{env}}
\circ
T_{\text{ppo}}
\circ
T_{\text{respawn}}.
\]

This is an abstraction of the actual tick order and should be read as such.

### Combat-first semantics

The code implements:

\[
T_{\text{combat}} \prec T_{\text{move}}.
\]

Therefore a slot killed by combat at tick \(t\) is excluded from the movement phase of that same tick.

## 6. Signed-zone and catastrophe math

### Repository-implemented effective-zone resolution

Let:

- \(Z_{\text{base}}(y,x)\) be the canonical signed base zone value
- \(Z_{\text{override}}(y,x)\) be the catastrophe override value
- \(M_{\text{apply}}(y,x) \in \{0,1\}\) be the apply mask

Then the effective zone field is:

\[
Z_{\text{eff}}(y,x)
=
\begin{cases}
Z_{\text{override}}(y,x), & \text{if } M_{\text{apply}}(y,x)=1, \\
Z_{\text{base}}(y,x), & \text{otherwise.}
\end{cases}
\]

The controller then clamps the result into \([-1,1]\).

### Environment use of the effective field

For a live slot \(s\) at position \((x_s, y_s)\), let:

\[
z_s = Z_{\text{eff}}(y_s, x_s).
\]

The environment phase uses the sign of \(z_s\):

- if \(z_s > 0\), HP gain is proportional to `HEAL_RATE` and \(z_s\)
- if \(z_s < 0\), HP damage is proportional to `CATASTROPHE_NEGATIVE_ZONE_DAMAGE_RATE` and \(|z_s|\)

A compact expression is:

\[
\Delta h_s^{\text{zone}}
=
\begin{cases}
+\eta_{\text{heal}} z_s, & z_s > 0, \\
-\eta_{\text{harm}} |z_s|, & z_s < 0, \\
0, & z_s = 0,
\end{cases}
\]

where:

- \(\eta_{\text{heal}} = \texttt{HEAL_RATE}\)
- \(\eta_{\text{harm}} = \texttt{CATASTROPHE\_NEGATIVE\_ZONE\_DAMAGE\_RATE}\)

This is a faithful mathematical condensation of the inspected code path.

## 7. PPO reward model

### Repository-implemented composition

For a slot \(s\) that acted this tick, the engine forms a final PPO reward by summing selected components.

A compact representation is:

\[
r_t^{(s)}
=
r^{(s)}_{\text{indiv}}
+
r^{(s)}_{\text{team}}
+
r^{(s)}_{\text{hp}},
\]

where:

\[
r^{(s)}_{\text{indiv}}
=
r^{(s)}_{\text{kill}}
+
r^{(s)}_{\text{dmg+}}
+
r^{(s)}_{\text{dmg-}}
+
r^{(s)}_{\text{contest}}
+
r^{(s)}_{\text{healrec}},
\]

and

\[
r^{(s)}_{\text{team}}
=
r^{(s)}_{\text{teamkill}}
+
r^{(s)}_{\text{teamdeath}}
+
r^{(s)}_{\text{teamcp}}.
\]

This is repository-implemented reward wiring.

### HP shaping term

In raw mode:

\[
r^{(s)}_{\text{hp}} = \alpha_{\text{hp}} \, h_s.
\]

In threshold-ramp mode, the code effectively computes a clamped ramp based on HP fraction. If the configured threshold is at or above \(1\), the ramp becomes zero.

## 8. PPO background mathematics

### Clipped surrogate

Background PPO theory aligned to the implementation:

\[
L_{\text{clip}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
r_t(\theta) A_t,\,
\operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t
\right)
\right].
\]

### Value loss

The runtime also computes a critic objective. A standard compact expression is:

\[
L_V(\theta) = \mathbb{E}_t \left[(V_\theta(s_t) - R_t)^2\right].
\]

The inspected code also includes a clipped-value-loss path.

### Entropy term

\[
L_H(\theta) = - \mathbb{E}_t\left[ \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \right].
\]

### Combined PPO objective

A compact background form is:

\[
L(\theta)
=
- L_{\text{clip}}(\theta)
+
c_V L_V(\theta)
+
c_H L_H(\theta).
\]

This is background mathematics used to read the code; it is not a claim that every line is symbol-for-symbol identical to a textbook presentation.

## 9. GAE

The implementation uses GAE-like recursion.

For TD residual

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
\]

advantage can be written as

\[
A_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}.
\]

Return is then

\[
R_t = A_t + V(s_t).
\]

## 10. Telemetry and aggregate statistics

The repository records many cumulative and summary values. A generic time-series summary statistic can be written as

\[
m_t = f(\mathcal{W}_t),
\]

where \(f\) may extract alive counts, mean HP, kills, deaths, damage totals, scheduler pressure, or catastrophe flags.

The code also stores cumulative team counters in `SimulationStats`, which can be interpreted as monotone aggregates over event history rather than as direct physical state variables.

## Deeper derivation notes

<details>
<summary>Why the catastrophe overlay is mathematically simpler than a blended field</summary>

The inspected controller uses a masked replacement model rather than a continuous blending model. That means the runtime-effective field is piecewise-selected rather than mixed by interpolation weights. This is operationally simpler because:

- the selected value at a cell has one source at a time
- edit locking can follow the same mask
- checkpoint serialization can store a clean base layer plus a clean override layer

</details>

<details>
<summary>Why per-slot PPO changes the interpretation of \(\theta\)</summary>

In shared-policy PPO one often writes one global parameter vector \(\theta\). In this repository, the more faithful interpretation is a family of slot-local parameters \(\theta_s\). Grouped execution does not change that ownership model.

</details>

## Related documents

- [Agents, observations, and actions](06-agents-observations-actions.md)
- [Learning and optimization](07-learning-and-optimization.md)
- [Glossary and notation](15-glossary-and-notation.md)
