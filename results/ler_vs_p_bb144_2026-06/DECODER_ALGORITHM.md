# Adaptive-Legs Chromatic-Block-Gibbs DEM Decoder — Algorithm & Per-p Parameters

Precise specification of the `ColoredBlockGibbsDecoder`
(`src/pbit_qec/circuit_decoder.py`) used for the bb144
`[[144,12,12]]` LER campaigns, followed by the optimal parameters per physical
error rate `p`.

All equations below are faithful to the implementation (`_decode_chunk`,
`_init_state`, `_sweep`, `_update_best`) and the DEM construction (`dem.py`).

---

## 1. Problem setup and notation

A Stim circuit is compiled to an **undecomposed** detector error model (DEM),
which `dem_to_matrices` turns into sparse GF(2) matrices (hyperedges kept as full
columns; `^` separators ignored):

| symbol | meaning | bb144 value |
|---|---|---|
| $n$ | error mechanisms (columns) | 67 824 |
| $m$ | detectors (checks / rows of $H$) | 1 728 |
| $k$ | logical observables | 12 |
| $H \in \mathbb{F}_2^{m\times n}$ | check matrix (`check_matrix`) | — |
| $L \in \mathbb{F}_2^{k\times n}$ | observable matrix (`observables_matrix`) | — |
| $p_j \in (0,\tfrac12)$ | prior fault probability of mechanism $j$ | — |

Per-mechanism **log-likelihood ratio (LLR)** prior (`self.lambdas`):

$$\lambda_j \;=\; \ln\!\frac{1-p_j}{p_j} \;>\; 0 .$$

Given a shot's **syndrome** $s\in\mathbb{F}_2^m$, the decoder searches for an
error pattern $e\in\{0,1\}^n$ with

$$\text{residual}\quad r(e) \;=\; H e \oplus s \in \mathbb{F}_2^m,\qquad
e \text{ is **valid** } \iff r(e)=\mathbf{0}.$$

The prediction is the induced logical flip $\hat o = L e \bmod 2$; a shot is a
**logical failure** iff $\hat o \neq o_{\text{true}}$ (any of the $k$ bits).

---

## 2. Energy / target distribution

Two energy terms — the **prior** (data) term and the **syndrome penalty**:

$$U(e) \;=\; \sum_{j} \lambda_j\, e_j \qquad(\text{prior energy, } \propto -\log \text{likelihood among configs}),$$

$$E_\beta(e) \;=\; \gamma\, U(e) \;+\; \kappa\,\lVert r(e)\rVert_1,\qquad \kappa \;=\; \rho\,\beta .$$

The sampler targets the Gibbs measure $\pi_\beta(e)\propto e^{-E_\beta(e)}$ via
single-site updates. Two design choices matter:

- **Decoupled field:** $\gamma$ (fixed) scales the prior; $\kappa=\rho\beta$ scales
  the syndrome penalty. The anneal $\beta$ therefore heats **only** the syndrome
  term — the prior keeps a constant pull toward low-weight errors. (Recovers the
  classic single-temperature form when $\gamma=\beta$.)
- **Penalty strength** $\rho$ is set relative to the strongest prior:
  $$\rho \;=\; \texttt{rho\_scale}\cdot \max_j \lambda_j .$$

As $\beta\to\infty$ the valid manifold $\{e: r(e)=0\}$ dominates and, within it,
minimum prior energy / minimum weight is selected.

---

## 3. Single-site update (one bit)

For column $c$ let $\partial c$ be the detectors it touches, with degree
$\deg_c=|\partial c|$, and let

$$v_c \;=\; \sum_{d\in\partial c} r_d \quad(\text{violated checks incident to }c,\ \texttt{before}=\text{gather}\,@\,r),\qquad \mathrm{sign}_c \;=\; 1-2e_c .$$

Toggling $e_c$ flips every $r_d,\,d\in\partial c$, so the residual weight changes by

$$\sigma^{\text{syn}}_c \;=\; \deg_c - 2 v_c \qquad(\Delta\lVert r\rVert_1 \text{ on a flip}).$$

The **flip field** (energy change of a toggle, `flip_delta`) is

$$\boxed{\;\Delta_c \;=\; \gamma\,\mathrm{sign}_c\,\lambda_c \;+\; \kappa\,\sigma^{\text{syn}}_c\;}\qquad \kappa=\rho\beta_t .$$

**Gibbs / heat-bath (`update="gibbs"`, used in all campaigns).** With the
directed energy gap $\Delta^{1|0}_c=\mathrm{sign}_c\,\Delta_c=E_\beta(e_c{=}1)-E_\beta(e_c{=}0)$,

$$P(e_c{=}1) \;=\; \sigma\!\big(-\Delta^{1|0}_c\big) \;=\; \tfrac12\Big(1-\tanh\tfrac12\Delta^{1|0}_c\Big),
\qquad e_c \leftarrow \mathbb{1}[u<P(e_c{=}1)],\ u\sim\mathcal U(0,1).$$

(Implementation flips iff $\big(u<P(e_c{=}1)\big)\oplus e_c$.)

**Metropolis alternative (`update="metropolis"`).**

$$\text{accept toggle} \iff \Delta_c\le 0 \ \ \text{or}\ \ u < e^{-\Delta_c}.$$

On a flip, the bookkeeping is updated incrementally (exact, no re-summation):

$$e_c\mathrel{\oplus}=1,\quad U\mathrel{+}=\mathrm{sign}_c\lambda_c,\quad
\lVert r\rVert_1\mathrel{+}=\sigma^{\text{syn}}_c,\quad r_d\mathrel{\oplus}=1\ \forall d\in\partial c .$$

---

## 4. Chromatic block Gibbs (one sweep)

Exact parallel Gibbs requires conditionally-independent sites. `greedy_coloring`
partitions the columns into **check-disjoint** color classes
$\mathcal C_1,\dots,\mathcal C_q$ (no two columns in a class share a detector).
Within a class all sites are independent given the rest, so they are updated
**simultaneously and exactly**. One **sweep** = update every color once = one
full pass over $H$ ($O(\mathrm{nnz}(H))$ per chain).

```
sweep(states, r, U, ‖r‖, β):
    κ ← ρ·β
    for each color block C (check-disjoint):
        before ← gather_C @ r                 # v_c for c in C   (sparse mat-vec)
        sign   ← 1 - 2·states[C]
        synΔ   ← deg_C - 2·before
        Δ      ← γ·sign·λ_C + κ·synΔ           # flip field
        u      ← Uniform(|C| × rows)
        # gibbs:
        p1     ← ½(1 - tanh(½·sign·Δ));  flip ← (u < p1) XOR states[C]
        states[C] ^= flip
        U        += Σ(flip · sign·λ_C)
        ‖r‖      += Σ(flip · synΔ)
        r[∂C]    ^= scatter(flip)              # toggle incident detectors
```

(The GPU backend runs an equivalent fused CUDA kernel per color; residual is held
in `float32` with parity-preserving `(a+b) mod 2`.)

---

## 5. Annealing, chains, selection

- **Anneal schedule** (`_beta_schedule`): geometric ramp over $S$ sweeps
  $$\beta_t \;=\; \beta_0\,(\beta_1/\beta_0)^{t/(S-1)},\qquad t=0,\dots,S-1 .$$
- **Chains:** $C$ independent replicas per shot are advanced in lockstep
  (columns of the state matrix, `rows = shots × C`). They share no state — pure
  breadth of exploration.
- **Selection of the reported correction** per shot (`_update_best`):
  - `selection="min_weight"` (campaigns): among **valid** samples keep
    $\arg\min_e \lVert e\rVert_1$ (prior-energy tiebreak).
  - `selection="best_energy"`: keep $\arg\min_e U(e)$ among valid samples.
  - **Fallback:** if a shot is never valid, return its lowest-residual sample
    $\arg\min_e (\lVert r(e)\rVert_1,\,U(e))$ (`best_any`) and mark
    `valid_mask=False`.

---

## 6. Adaptive legs (RelayBP-style ensemble)

A **leg** is one wave of $C$ fresh chains annealed for up to $S$ sweeps. Each chain
that ever reaches a valid state contributes one ensemble **member**. A shot stops
once it has collected `target_valid` members; shots short of the target get
another leg, up to `max_legs`. The global best-valid (and best-any) per shot
survives across legs. Setting `max_legs=1, target_valid=None` recovers a single
fixed-budget pass.

```
decode_chunk(syndromes S):
    init global best_valid/best_any per shot
    active ← all shots
    for leg in 1..max_legs:
        if active empty: break
        states,r,U,‖r‖ ← init_state(S[active], C)        # e=0 ⇒ r=s
        for t in 0..sweeps:
            capture valid states → update best_valid       # selection rule
            update best_any (lowest residual)
            mark chains that have been valid this leg
            if target_valid set and every active shot has ≥target_valid members: break
            if t == sweeps: break
            sweep(states,r,U,‖r‖, β_t)
        members[shot] += #chains-ever-valid-this-leg
        legs_used[shot] += 1 ;  sweeps_used[shot] += sweeps-run-this-leg
        active ← shots with members < need               # need = target_valid (or 1)
    e* ← where(have_valid, best_valid, best_any)          # per shot
    return  ô = (L e*) mod 2,  valid_mask = have_valid,
            legs_used, sweeps_used
```

Top level (`decode_with_stats`): chunk shots by a memory budget, run
`decode_chunk` per chunk, concatenate predictions and per-shot
`valid_mask / legs_used / sweeps_used`, and report mean + max + p99 of legs and
executed sweeps.

---

## 7. Error counting (LER) — exactly as scored

Per shot, with prediction $\hat o$ and truth $o$ (RelayBP / sinter convention):

$$\text{error} \;=\; \mathbb{1}\big[\,\exists i:\ \hat o_i \neq o_i\,\big]
\quad\Longrightarrow\quad \mathrm{LER}=\frac{1}{N}\sum_{\text{shots}}\text{error}.$$

This is **end-to-end** (full $d{=}12$, $12$-round memory experiment, *not* per
cycle) and **joint over all 12 observables** (*not* per logical qubit). Shots are
decoded independently; each decode is **decomposition-free** (full DEM /
hyperedge). A diagnostic strict count $\mathbb{1}[\text{error}\ \lor\ \neg\text{valid}]$
is also reported. (See `bench_circuit_pbit.py`.)

---

## 8. Cost model

| knob | role | one unit costs |
|---|---|---|
| `chains` $C$ | parallel replicas (breadth) | $C\times$ width per sweep |
| `sweeps` $S$ | anneal steps within a leg | one $O(\mathrm{nnz}\,H)$ graph pass / chain |
| `max_legs` $L$ | adaptive restarts | up to $L\times S$ sweeps |

Per-shot work in **BP-iteration-equivalents** (one $O(\mathrm{nnz}\,H)$ pass):

$$\text{BPIE}_{\text{shot}} \;=\; C \times (\text{sweeps executed}) \;=\; C\cdot \texttt{sw\_exec},
\qquad \text{worst case } C\cdot L\cdot S .$$

At $p{=}0.005$: mean $\approx 128\times 644 \approx 8.2\times10^4$, worst case
$128\times(12\times500)=7.68\times10^5$. (One p-bit sweep $\equiv$ one BP iteration;
the $C$ factor is unique to the sampler — RelayBP has no `chains` axis.)

---

## 9. Optimal parameters per `p`

**Fixed across all `p`** (winning configuration; `rho_scale=0.3` is the dominant
lever — lowest mean weight, highest validity, lowest LER at every `p`):

```
update      = gibbs
selection   = min_weight
gamma (γ)   = 1.0
capture     = sweep
beta        = 0.5 → 8.0   (geometric anneal)
rho (ρ)     = rho_scale · max_j λ_j ,   rho_scale = 0.3
target_valid= 1
```

**Per-`p` knobs and measured LER** (bb144 `memory_X`, $d{=}12$, $12$ rounds):

| `p` | chains $C$ | sweeps $S$ | max_legs $L$ | target_valid | rho_scale | LER | ± stderr | errors / shots |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 128 | 500 | 12 | 1 | 0.3 | 1.05e-2 | 1.3e-3 | 63 / 6 000 |
| 0.004 | 128 | 500 | 12 | 1 | 0.3 | 1.08e-3 | 1.3e-4 | 67 / 62 000 |
| 0.003 | 128 | 400 | 16 | 1 | 0.3 | 7.85e-5 | 1.1e-5 | 51 / 650 000 |
| 0.002 | 128 | 500 | 12 | 1 | 0.3 | 9.43e-6 | 2.98e-6 | 10 / 1 060 000 |
| 0.001 | 128 | 500 | 12 | 1 | 0.3 | *skipped (infeasible, ~1e8 shots)* | — | — |

Notes:
- Only `sweeps`/`max_legs` vary: $p{=}0.003$ used `sweeps=400, max_legs=16`; all
  others `sweeps=500, max_legs=12`. Mean executed legs is $\approx 1.3$ (most
  shots solve in one leg), so the budget is dominated by $C\cdot S$.
- **$p{=}0.005$ failure decomposition** (same-seed re-measure with the current
  code, `errors / invalid / errors_strict = 69 / 83 / 130` over 6 000 shots):
  valid-but-wrong-coset $=47$, invalid-and-mismatched $=22$,
  invalid-but-lucky-match $=61$. The dominant failure mode is genuine
  wrong-coset valid corrections, *not* unconverged shots.

---

## 10. Reproduction (one `p`, one GPU)

```bash
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
.venv-gpu/bin/python experiments/bench_circuit_pbit.py \
  --device cuda --shots 10000 --seed 0 \
  --circuit "<circuit .stim path for p>" \
  --chains 128 --sweeps 500 --max-legs 12 --target-valid 1 --rho-scales 0.3
```

For $p{=}0.003$ use `--sweeps 400 --max-legs 16`. Sharded multi-GPU campaigns and
pooling: see `README.md` and `scripts/` in this directory.
