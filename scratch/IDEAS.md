# Idea ledger — backprop-free LUT-net optimization toward >80% val CIFAR-10

The optimizer is a **mixture of RL + evolution strategies over a discrete/binary network**,
learning connections, gate truth tables, AND architecture (sharing, strides, polarity,
capacity). No backprop, no gradients — every accept is exact on the full train set (or its
verified cascade with depth). This file is the running record of every idea tried or queued,
by research domain, so nothing lives only in chat.

Status key: ✅ SHIPPED (won its test, in code & default) · ❌ REFUTED (A/B'd, deleted from
code) · 🔬 QUEUED (designed, not yet run) · 💡 SPECULATIVE (idea, not yet designed).

Reference ceiling: conv-difflogic (Petersen et al., NeurIPS 2024, arxiv 2411.04732) reaches
86.29% CIFAR-10 with logic gates via SGD — proves the model class can exceed 80. Our job is
to get there with discrete zeroth-order search. Plain deep difflogic (2022) ≈ 62% = the "the
architecture works" bar.

---

## Multiresolution / coarse-to-fine (THE REFACTOR, 2026-07-05) — generalize everything

### v2 REFRAME (same day, supersedes the framing below where they differ): ONE SUBSTRATE,
### ONE PROTOCOL, NO LEARNING PHASES.
Predefine ONLY the total unrolled/unfolded budget of fan-in-2 gates S (as an address lattice
rank × channel × 32² so position/relative-pattern/acyclicity exist) + input bits + voting
head. Learned = exactly TWO discrete objects: (1) WIRING (2 taps/gate → any lower rank;
skips/depth-attention = the δr tap coord) and (2) the PARTITION (per-field tying along any
lattice axis incl. rank: rank-spanning = recurrence, spatial = conv). EMERGENT, not
mechanisms: fan-in (= subtree size — NO fan-in parameter, NO 2^n table ever; the gate count
allocated to a subfunction is the "degree" interpolating from coarse functions toward the
unreachable exponential family; TTs are NEVER decomposed — splitting a big TT into smaller
ones interferes; refinement unties WHO shares, never WHAT is computed), width, depth
(pass-through TTs = transparent ranks), architecture. ONE PROTOCOL: every move =
(partition node, field, discrete delta ∈ {tt flips, tap edit, sgn, cls, SPLIT, MERGE}) →
overlay → rank-ordered exact recompute → full-train hinge → accept/XOR-revert; SPLIT
neutral-free, reject halves by DESCENDING the partition. No phases: all granularities live
from round 0, coarse→fine EMERGES from bandit hinge/second economics; split depth decided by
measured cross-member correlation (disagreement D(G) = free-gain − tied-gain from the exact
ben tables), never a schedule. Coarsest init = ONE group: the net starts as a learned binary
cellular automaton and CD differentiates an architecture out of it. DELETED by the reframe:
learned-fan-in-inside-TT lever, macro-gate init as a special concept, K as semantic knob
(substrate K=2; K=6 kept only as A/B arm at matched unrolled budget), hand-tapered channel
pyramids as default.

### v3 (same day): THE HASH GATE — one gate type subsumes all function families.
STAGE 1 ✅ SHIPPED (2026-07-05, this commit): fixed M=2**K substrate — `coef` (S,K) int16
hash weights, `_cells` = unpack-mul-accumulate-mod, rewire's shifted-gather generalized
(bas = h − c_k·x_k mod M), new `cd-cf` lever (block-scores candidate weights; c'=0 = learned
fan-in down, revive = up), `--gate lut|ternary` init corners (lut = c_k=2^k, verified
bit-exact vs the classic address in scratch/test_hash.py; ternary = signed-step tables).
All levers exactness-tested at both corners (test_hash.py). No phases: cd-cf is one more
bandit arm; per gate the learnables stay exactly (internal function = tt+coef, connections,
sharing). STAGE 2a ✅ SHIPPED (same day): --tsize decouples table size M from tap budget K
(K>8 with small tables; eval O(K), storage O(M)) + --gate hash init corner (c uniform in
[1,M) → full-table occupancy). STAGE 2b 🔬 QUEUED: per-gate M growth/halving (T-duplicate
= bit-identical) and K growth (new tap at c=0), both exactly neutral.

hg1 A/B (2026-07-05, 6-layer 184k K=6, aug, r4-5 readout): ❌ TERNARY INIT REFUTED —
val 39.6/38.5 (K6/K8) vs lut-corner 47.7; cause = dead capacity (c∈{−1,0,1} reaches only
2K+1 of 2^K cells; tt accepts 1.5k/round vs 300k+). ✅ cd-cf VALIDATED as a lever from the
lut corner: 24k accepted weight moves by r3, top-half bandit arm at layers 4-5, and ctrl
reached val 47.7 by r4 (wave-2 needed ~2x the rounds). Lesson: init must occupy the full
table; structure (threshold/hash regimes) must be EARNED by CD, never imposed at init.
hg2 (running): hash init at M=64 with K=6/12/16, K=12 M=256, lut K=8 — does a bigger tap
budget with constant table size pay? reg1 (running): spatial pyramid 32,32,16,16,8,8; cutout
queued for backfill.

FINALS (2026-07-05 late): hg1_ctrl r14 = val 50.36 / test 50.33 (peak 50.48 @ r12) vs
wave-2's 50.1 — cd-cf compounds, kept. hg2 finals: hashK6/M64 ended 47.0 @ r8, gap 10.6,
still climbing (+1.5 over last 2 rounds vs ctrl's +0.6) → long-horizon retest queued on the
pyramid base (gen2_pyrhash, r30). ❌ K12/K16 at M=64 REFUTED as INIT (6.5+ pts behind at
round parity, 2x round cost — idle tap budgets don't pay; fan-in must GROW on demand via
c=0 revival, stage 2b). ❌ K12+M256 canceled untested (both parents lost).

Same-day verdicts (r3-r12 readouts, 2026-07-05 evening):
- ✅ **cd-cf compounds on the live recipe**: hg1_ctrl val 50.48 @ r12 — beats wave-2's 50.1
  (r14) with the same everything else. Hash-weight moves are a keeper lever.
- ✅ **POOLING PYRAMID WON** (reg1_pyr, --spatial 32,32,16,16,8,8): val 48.92 @ r8 in 33 min
  vs flat ctrl 49.4 @ r8 in 85 min — ~equal val at 2.6x less wall-clock, HALF the overfit
  gap (11 vs 20). New fast-iteration geometry; gen2 probes its wall (r30) + 2x width.
- ❌ **lut K=8 (M=256) REFUTED**: train 83.7 / val 43.8 @ r3, gap 40, val decelerating —
  bigger LUT tables memorize. TABLE CAPACITY is the overfit knob; tap count is the reach
  knob. Killed at r3. Reinforces small-M + learned-weights direction.
- 🔬 hash init (K6 M64): val 45.5 @ r6, ~3 pts behind lut init but gap 9.5 vs 21 and still
  climbing +1/round late — the regularized-capacity signature. K12/K16: slower starts,
  costlier rounds (16/27 min) — K16 likely too slow to justify unless its curve crosses.
- ❌ **FAN-OUT CAP REFUTED** (gen1 r8 finals, user's signal-collapse hypothesis): fo6 ended
  48.9 (below ctrl 49.4), fo16 49.5 = noise; gaps unchanged (21-22). Concentration is REAL
  (median input bit 28 readers, max 55 vs 20 uniform; hot signals ~all input bits) but
  masking over-cap sources from new wiring bought nothing at the r8 horizon. Code deleted
  same day (measurement + verdict preserved here).
- ✅ **MARGIN 2 WON** (gen1_margin2): val 51.36 @ r8 / test 51.28 — +2.0 over ctrl at round
  parity, beat ctrl's ENTIRE r14 run (50.48) in 8 rounds. Boosting-margin regularization
  delivers; compounded into base.env together with the pyramid. DOSE-RESPONSE (flat, r8):
  m=1: 49.4, m=2: 51.4, m=4: 51.7 (gap 16.8, best), m=8: 49.0 ❌ (overshoot; killed @ r6).
  On the pyramid m=2 ≥ m=4 at parity → BASE keeps margin 2 (flat 2–4 plateau = robust).
- ❌ **CUTOUT REFUTED (2nd time)** (gen1_cut8): 49.2 @ r8 vs ctrl 49.4 — null in the
  overfit regime too (first refutation was in the underfit regime). Code deleted.
- ✅ **WINNERS COMPOSE** (gen3_ctrl30 = pyramid + margin2): 51.5 @ r7 (16 min!), 54.04 @
  r20 — +3.9 over the old wall in a third of the wall-clock, gap ~12, still climbing.
- 🔬 gen4 (running): hash-full vs grow-and-earn small-start (--init-tsize 8, cd-ms lever)
  on the new BASE, r30 — is LEARNED per-gate capacity the better regularizer?
- gen2 finals: margin-1 pyramid WALLS at ~52.7 (r19-24 flat; killed r24) — margin 2 is
  worth ~+2 asymptotically, not just early. ❌ pyrhash (hash init, margin-1 pyramid)
  refuted at horizon: 48.0 @ r21, +0.15/round — killed; gen4_hashfull (margin 2) is the
  surviving hash-init probe. pyr2x/gen3_2x: width behind at parity under BOTH margins
  (~2 pts, 2.6x cost) — capacity-via-width keeps losing to capacity-via-margin/geometry.
- gen5 (queued): explore-anneal within the run (1:0.3:30 — pyrlong's late jump at r19
  suggests schedule shape matters) + r60 frontier run to locate the compounded wall.
Gate = `T[(Σ_k c_k·x_k) mod M]`: K taps, learned integer weights c_k, learned bit-table T
of size M. Corners: c=2^k, M=2^K = full LUT (today's gate, exactly); c∈{0,1}, no wrap =
symmetric; c∈{−1,0,1} + step T, no wrap = BitNet ternary threshold (the stable corner —
init here, let CD earn the hashing regime); small M = compressed hash gate. EXACT
neutrality of all growth: new tap enters with c=0 (inert, test-free — supersedes the
"add as MSB" idea: under mod, MSB placement is dead for M=2^m and not minimal-influence
otherwise); M→2M with T duplicated (T'[j]=T[j mod M]) is bit-identical (h mod 2M ∈
{h, h+M}, both read T[h]); halve-M when halves agree = coarsening. Fan-in AND expressivity
(table size) are learned per group, storage O(M) never 2^fanin, eval O(K) — breaks the
K≤8 executor barrier. Block-CD survives verbatim (buckets partition rows → scatter-add
proposes all M bits at once); rewire's shifted-gather trick survives (h_base = h − c_k·x_k)
and also block-scores candidate c values (new cheap lever). Executor change = _cells only:
unpack-OR-shift → unpack-mul-accumulate-mod. Prior art (searched 2026-07-05): Bloom
WiSARD (hash tables replace RAM-node TTs, ~6 orders memory reduction, arxiv 2203.01479 +
ESANN 2019); DWN differentiable weightless nets (ICML 2024, arxiv 2410.11112 — same model
class, SGD-trained, no learned hash/growth); HashedNets (arxiv 1504.04788 — random
collision-tying is benign); Instant-NGP multiresolution hash encoding (arxiv 2201.05989 —
collisions disambiguated ACROSS resolutions: several gates at different M reading the same
taps disambiguate each other; our voting head already averages). Novel combination: exact
discrete CD on hash weights + exactly-neutral fan-in/table growth + hierarchical tying.
Risks: wraparound destroys monotone structure (init no-wrap); small-M aliasing escapable
only via M-growth (measure accept rates early); avoid M=2^m with c=2^k-style dead bits
(prime/odd M or odd multipliers).

Original design notes (machinery below remains valid):
Design decision: **the hierarchy lives in the OPTIMIZER, not the executor.** The executable
net stays what cd.py is (materialized fan-in-K LUT slots, bitpacked, cascade, exact-hinge
accepts). New: every learnable field (tt, taps, sign, cls, step) is addressed through a
**partition tree** over slots — `leaf_value = fold(ancestor deltas)` (XOR for bits, + for
coords). What generalizes:
- **Move at any node** = coordinated flip on all descendant leaves (cd_joint's multi-gate
  accept, generalized), verified on the exact hinge. One coarse accept moves thousands of
  gates for ~one partial forward — "low harmonics first", exact. Bandit arms become
  (op × tree-level × layer): the coarse→fine schedule is LEARNED, not hand-annealed, and
  coarse coefficients stay live after refinement (late low-frequency moves remain possible).
- **Refinement** = add children with ZERO deltas → behavior-identical, test-free (the
  net2net-neutral trick, now the universal structural op).
- **TT coarse-to-fine**: a "fan-in-n" macro gate = balanced tree of n-1 fan-in-2 LUTs with
  TTs *tied* through the group tree (one 4-bit TT per tree level ≈ m·n bits, not 2^n).
  Untying = adding harmonics; endpoint (free 2-input gates) reached purely by untying.
  REJECTED alternative: coarse gates as literal LTFs / low-degree Walsh coefficients,
  decomposed into gates later — decomposition moves are lossy, breaks everything-exact;
  tied tree has the same coarse dim and is executable from round 0. Internal tree nodes are
  slots → they vote → multi-output ("many fanout") for free.
- **Connection coarse-to-fine**: taps = group pattern (source layer, base coord, per-tap
  offset table) + hierarchical leaf deltas. Coarse rewire retargets a whole group's
  receptive field or source DEPTH (learned skips at group granularity). Locality prior and
  `step` become special cases of the pattern.
- **Sharing in ALL directions**: a group = set of placements in the (layer × c × h × w)
  lattice; orbits may span depth (iterated block, ALBERT-style). Current copies+stride
  sharing = the spatial special case, to be SUBSUMED (share_move/split_move deleted).
  Depth-tied cascade recomputes in rank order (explicit-rank DAG already provides this).
- **Width/depth learned**: growth = neutral refinement (width split exists; depth split =
  insert pass-through/residual-init block, exactly neutral). Reverse op **coarsen/merge**
  (fold groups whose deltas stayed ~0) = the capacity-annealing 💡, attacks the overfit gap.
- WHY THIS ATTACKS THE WALL: trajectory finding says the 47–50 ceiling is the optimization
  (val gain/round decelerating) — per-gate moves are too high-frequency; multigrid is the
  standard cure for exactly that.

Refactor stages (exactness checks must pass at every stage):
1. 🔬 `Group` forest beside existing arrays (parent ptrs, per-field deltas); materialized
   leaves stay the execution source of truth; `--check` folds the tree and diffs vs leaves.
2. 🔬 Levers take a node, not a gid: masked delta on all descendant leaves, one exact
   accept, XOR-revert on reject (generalize cd_joint).
3. 🔬 Re-express sharing/splits as groups; delete share_move/split_move (cd.py shrinks).
4. 🔬 Depth-spanning groups + neutral depth-insert op.
5. 🔬 Macro-gate init: net starts as few fan-in-2 trees with tied TTs + patterned taps
   (replaces `--init-deg` as THE coarse init).
6. 🔬 Bandit arms (op × level); refine/coarsen become priced ops.

## Reinforcement learning
- ✅ **Bandit operator scheduler** — epsilon-greedy × probability-matching over (operator ×
  layer) arms, reward = exact hinge-decrease per second, EMA credit. This IS the "RL" core:
  it learns which lever at which depth pays, pricing in cascade cost. The e/K floor is
  load-bearing (see Thompson below).
- ✅ **Prioritized gate visiting** (`heat`, = prioritized experience replay) — per-gate EMA
  of recent hinge yield steers the search budget; floor keeps cold gates covered; decays.
  Broke a 4-round val plateau (44.5→45.9).
- ✅ **Difference rewards / counterfactual credit** (COMA-style, multi-agent RL) — the
  `rebuild` lever: each gate's exact marginal vote value (leave-one-out, computable because
  the head is linear in votes); gates whose removal would *help* are deadwood → replaced
  with fresh randoms. Validated firing 50–180/round.
- ❌ **Thompson sampling arm choice** — posterior draw per arm instead of epsilon-greedy.
  REFUTED hard: collapsed onto one cheap-reward arm, starved tt/rewire/splits entirely
  (val 28 vs 32). Deleted; the epsilon floor is what keeps a non-stationary bandit honest.
- 🔬 **Discounted-UCB / sliding-window bandit** — principled non-stationary arm values
  (landscape shifts as the net trains) without Thompson's collapse.
- 🔬 **Count-based exploration bonus** — sqrt(log t / n_visits) added to gate priority; the
  principled version of heat's coverage floor.
- 🔬 **Macro-actions / options** — composite arms ("split then RS the clones apart",
  "share-up then re-step") so the bandit prices synergies single moves can't express.
- 🔬 **REINFORCE-style learned proposal policies** — the mutation distributions (rewire
  offset, which TT cells to burst, step-edit type) become learned categoricals updated by
  accepted-improvement, per layer. The search itself gets a policy.
- 🔬 **Value function / critic for structural moves** — splits have *deferred* value the
  reward-now bandit prices at zero (this is why we batch them by hand). Train a small critic
  on observed hinge-improvement-over-next-k-rounds to pick which gates to split. Proper
  temporal credit assignment for architecture growth.
- 🔬 **Population-based training (PBT)** — automate the twin-GPU races: N runs, periodic
  exploit (losers copy winner ckpt) + explore (perturb temp, lever mix, aug). Meta-RL over
  hyperparameters. (We do this by hand now; the small-run sweeps are step one.)
- 💡 **Data-axis prioritized replay / curriculum** — weight the hinge toward
  borderline-margin images (the ones accepts can flip), anneal to full distribution.

## Evolution strategies / evolutionary computation
- ✅ **(1+1)-ES per gate** (`rs_pass`) — random binary mutations (multi-bit TT bursts, local
  re-taps), exact-fitness selection, neutral drift on plateaus.
- ✅ **Best-of-n selection** (`rewire`) — evolutionary tournament for connections.
- ✅ **net2net output-neutral cloning** (`split_move`) — capacity grows where it later pays,
  no random-refill penalty; exactly hinge-neutral so no accept test.
- ✅ **Annealed uphill acceptance** (`rs-temp`, simulated annealing) — accept bounded-worse
  moves, decays on the explore schedule. (Under ablation in wave-1 `w1_temp0`.)
- ❌ **PBIL / estimation-of-distribution refills** — fresh gates sampled from per-layer bit
  marginals of *surviving* gates instead of uniform. REFUTED: tie with uniform (val 35.5 vs
  35.25), no gain for the cost. Deleted.
- ❌ **Per-round random-batch accepts** — cheap accepts on a fresh batch each round.
  REFUTED: accepts overfit the batch, val crawled 27–28 while exact full-train jumped.
- 🔬 **Extremal optimization** (Boettcher–Percus, statistical physics) — always attack the
  *worst* components: power-law selection over per-gate exact harm. Complements heat (finds
  improving regions) by finding deadwood; rebuild already computes the harm signal.
- 💡 **Novelty search / quality-diversity** — bias neutral drift toward behavioral novelty
  (per-class score signatures), guard against the population converging to redundant gates.
- 💡 **Parallel tempering** (physics) — replicas at different rs-temp exchanging checkpoints
  when the hotter one wins; natural extension of the twin-GPU infra.

## Neuroscience / biologically-inspired
- ✅ **Signed votes / Dale's law** (`sign_pass`) — per-gate vote polarity: features can be
  negative evidence (inhibitory populations). CASCADE-FREE (outputs unchanged, only head
  reweights) → cheapest lever. Validated firing, decaying as polarity settles.
- ✅ **Residual initialization** (conv-difflogic, also skip-connection biology) — fraction
  of fresh TTs start as pass-through so signal flows through depth from round 0. Won its A/B
  with locality (36.5 vs 31.8).
- ✅ **Locality prior** (retinotopy / local receptive fields) — fresh taps start within ±R
  px of the gate's position; rewiring undoes it where long-range pays. Same A/B win.
- 💡 **Forward-Forward-style layer-local goodness** (Hinton 2022; VFF-Net 2025) — give
  neutral drift a *direction*: among hinge-equal moves prefer those increasing per-layer
  class separability. Turns aimless plateau drift into layerwise representation learning,
  still no backprop. Refs: ncbi PMC12586560, techxplore VFF-Net.
- 💡 **Homeostatic plasticity / target firing rates** — regularize each gate toward ~50%
  activation so no gate goes dead or constant; keeps capacity usable.
- 💡 **Structural plasticity / dendritic gating** — depth-wise skip taps chosen by a
  learned per-gate gate on which layer to read (biology: dendritic compartments).

## Graph-based / combinatorial optimization
- ✅ **Explicit-rank DAG** — layers + connections into strictly-lower layers make the circuit
  acyclic under any shift; enables O(1) depth-0-style updates + bounded cascades.
- 🔬 **Survey/belief propagation** (SAT-solver lineage) — the net IS a circuit; message
  passing over it could propose coordinated multi-gate changes no local search finds.
  Highest risk / highest novelty.
- 💡 **Spectral / community-structured wiring** — init or bias connections by graph
  community structure of feature co-activation instead of pure locality.
- 💡 **Min-cut / flow-based pruning** — identify and cut low-information subgraphs wholesale
  rather than per-gate rebuild.

## Architecture (learned by the search, not hand-set)
- 🔬 **Learned OUTPUT connection** (`cls_pass`, cd-cl, NEW 2026-07-05) — until now each slot's
  vote class was HARDWIRED to `channel % 10`: the head could only flip polarity (sgn), never
  choose which class a gate votes for. So a great feature computed in the "wrong" channel
  could only be credited to that channel's class — an arbitrary cap on the readout, and the
  class quota (S/10 per class) was frozen. `ocls` makes the output class per-slot LEARNABLE
  (init = channel%10, so behavior is preserved until CD moves it); `cls_pass` proposes the
  best class per gate from the ±1 tables and verifies on the exact hinge with halving.
  Cascade-FREE (only the head re-attributes; stored outputs untouched, like sign_pass).
  EXACTNESS VERIFIED (isolated + --check: state diff 0.0, hinge drift 0.0; fires ~130/300
  gates/call). This is literally the user's "learn the output connection." A/B pending.
- ✅ **Learned weight sharing** = per-gate convolution (share degree per dim).
- ✅ **Learned input strides** (`step`) — decouples input stride from output tiling →
  stride/dilation/overlap/exact-tying all reachable; also makes channel splits neutral.
- ✅ **Flat spatial (32²) default** — BEAT the CNN pooling pyramid 47.3 vs 45.9 over a 5h
  parity race (pyramid stronger per round, 2× round cost never amortized). `--spatial`
  machinery kept for the scale path (coarse layers cut slots×D memory quadratically).
- 🔬 **Depth vs width at matched slots** — current wave/next wave probe.
- 🔬 **OR-pooling layers** (conv-difflogic ingredient 3) — max-t-conorm pooling as a layer
  type; pairs with spatial downsampling for their 61M-gate scale.
  → INSIGHT (2026-07-05): NO risky new layer type needed. A LUT gate with truth table
  fixed/biased to OR over K local taps into a COARSE `--spatial` layer *is* OR-pooling:
  downsampling = coarse grid (exists, exact), locality = `--init-loc`, and CD can refine it.
  So pooling is reachable with existing exact machinery + a small OR-biased-init flag. Test =
  run deg000 WITH a `--spatial` pyramid, long (the shape race that rejected pyramids was at
  SHARED init on a wall-clock metric — never tested with deg000 for final val).
- 🔬 **Logic-gate tree kernels** (conv-difflogic) — multi-gate trees as the conv primitive.

## Regularization / generalization (the current bottleneck: train 53 / val 47, gap growing)
- ✅ **Re-rolled augmentation** (flip + crop + jitter, fresh each round) — keeps the hinge
  from freezing; the stochasticity that replaces SGD minibatch noise.
- ❌ **Cutout** — REFUTED: hurts while the model underfits (it doesn't, now — worth
  re-testing under the current overfit regime; wave-1 `w1_aug6` tests jitter strength first).
- 🔬 **Stronger jitter / crop** — `w1_aug6` (jitter 0.6) live now.
- 💡 **Slower unsharing / capacity annealing** — the gap grows as sharing→0; hold capacity
  back until val stops tracking train.
- 💡 **DropConnect on votes** — randomly zero a fraction of gate votes per round (ensemble
  regularization, discrete-native).

---

## Fast-iteration sweeps (small runs, one variable each, same seed, 184k-slot 6-layer net,
## 14 rounds ≈ 50 min). Winners compound into defaults; losers deleted. Val@14:

**Wave 1** (baseline = old defaults): deg011 **44.2** ✅ · split512 **43.0** ✅ · temp0 40.1 ·
ctrl 40.1 · expfast 39.9 · aug6 39.3 ❌ · k4 37.8 ❌ (K=6 confirmed) · deg033 37.7 ❌.
→ VERDICT: less initial sharing wins big. Adopted defaults `init-deg 0,1,1`,
`split-batch 512`. (Flips the earlier conv-init promotion — that was for an OVERFITTING
shallow net; at depth-8 heavy tiling straitjackets the search.) Annealing/expfast/aug were
neutral-or-worse but kept (temp0≈ctrl, not clearly refuted).

**Wave 2** (baseline = wave-1 winners): deg000 **50.1** ✅ (fully unshared — new deep-net
best, adopted as default) · wide 46.8 · rs75 46.1 · rew75 45.6 · ctrl 45.1 · split1k 45.0 ·
loc4 45.1 · deep10 43.2 ❌ (deeper hurt). → VERDICT: unsharing monotonically wins
(0,0,0 > 0,1,1 > 0,2,2 > 0,3,3). Conv prior at init straitjackets the deep search. Adopted
`init-deg 0,0,0`. New bottleneck: deg000 train 74 / val 50 = **24-pt overfit gap**.

**Wave 3** (baseline = deg000): regularization knobs on the overfit gap — cutout 8/12,
jitter 0.6, crop 6, cutout+jitter combo. [running, 6 jobs, r14]

### ⚠️ Trajectory finding (from cdd8 big-run curve + wave-2) — reframes the whole effort:
Val sits near a **~47–50 ceiling across wildly different regimes**: shared-1.35M-slot
underfits (train 53 / val 47, 6-pt gap, still crawling +0.15/round at r38) while
unshared-184k-slot overfits (train 74 / val 50, 24-pt gap). Higher val at 7× FEWER slots
when unshared. **So the wall is neither pure capacity nor pure overfit — it's the
representation/optimization itself.** Two consequences:
1. 14-round sweeps RANK knobs; they do NOT converge — the big run climbed val 42→47
   monotonically over r6→r38 with no plateau. We have been KILLING RUNS TOO EARLY. The real
   number needs the winning recipe run LONG (40+ rounds) and at SCALE.
2. Aug-regularization (wave 3) can only shave the gap a few points; it will NOT break 50→80.
   The ceiling-raisers are: (a) scale (does the ~50 wall rise with slots? untested at the
   deg000 recipe), (b) OR-pooling / gate-tree kernels (conv-difflogic's actual 86%
   ingredients, not yet implemented), (c) fundamentally faster optimization (val gain/round
   is decelerating hard).

Best val to date: 55.0 (depth-0 flat, historical); deep-net best 50.1 (deg000, r14, unconverged).
