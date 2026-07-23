# SOT-1899 — time-budget deep horizon (prize-race / deck-out aware): NON-PROMOTION

**Outcome: rejected by the cross-agent league-KPI screen. Champion retained; behaviour reverted.**

## Hypothesis

take is the field's strongest rule-based agent (Kaggle 575.5) but its decision time is mean
0.157 ms / p95 0.31 ms against a 250 ms search budget — three orders of magnitude of unused
headroom. Its horizon (`_horizon_adjustment` + `adaptive_search_depth`) is a deliberately shallow
setup-first nudge. The two dominant loss modes are **prize-out** (70.3 % of losses; 64.9 % of Kaggle
trace losses) and **deck-out** (27 %). The candidate spent the headroom on *state-aware* corrections
for exactly those two modes rather than blind depth:

- **behind the prize race** (`opp prizes-left ≤ 2` and not behind us) → cancel the setup-first ATTACK
  penalty and add a bounded tempo reward, so a clock-advancing swing is not deferred into a prize-out
  loss;
- **losing the deck-out race** (own `deckCount ≤ 6` and `≤ opp deckCount`) → dampen the setup horizon
  that spends more turns digging instead of closing;
- a time-guarded one-ply / wider selective extension (`max_depth 3→4`, branch window `10→14`) in
  narrow, long-horizon lines only.

All terms stay well under the 10 000 lethal band and are opt-in via `TAKE_DEEP_SEARCH`
(default off ⇒ byte-identical champion); the unit suite is green with the flag off.

## Gate — SOT-1896 cross-agent league KPI (screen)

Real `cg`-engine matches, driver `ptcg-agent-matsu/eval/battle_matsu_take_ume.py`, deck `01`
(dragapult) **mirror**, **N = 6 per pairing**, take's opponent pool = {matsu, ume, zero}, **fault 0**.
Promotion rule (`docs/ai/league-kpi-gate.md`): candidate replaces the champion only when
`candidate.pool_ci_lower > champion.pool_ci_lower` at zero faults within latency budget.

Per-pairing take win rate (take wins / 6):

| opponent | champion (flag off) | candidate (flag on) |
| --- | ---: | ---: |
| matsu | 2/6 = 0.333 | **1/6 = 0.167** |
| ume   | 6/6 = 1.000 | 6/6 = 1.000 |
| zero  | 6/6 = 1.000 | **4/6 = 0.667** |

Pool aggregate (18 decided games each), Wilson 95 % CI:

| | pool win rate | Wilson 95 % CI | faults |
| --- | ---: | :---: | ---: |
| champion  | **0.778** | [0.548, 0.910] | 0 |
| candidate | 0.611 | [0.386, 0.797] | 0 |

The champion's fresh pool KPI (0.778) reproduces the SOT-1896 baseline (0.722) within screen noise.

## Decision — NON-PROMOTION

`candidate.pool_ci_lower = 0.386 < champion.pool_ci_lower = 0.548` — the gate is failed. The candidate
did not merely fail to improve: it **regressed**, and worst exactly where it was meant to help —
against the strongest opponent **matsu (0.333 → 0.167)** and against **zero (1.000 → 0.667)**, with
**ume unchanged (1.000)**. No confirm (large-N) run is warranted: the screen's purpose is to cut
clearly-not-better candidates cheaply, and this one is worse at screen.

**Why it backfired.** take's strength is disciplined setup-first tempo. Rewarding non-lethal swings
under prize-race pressure pulls it into premature attacks that concede board and the tempo that wins
the race, and the deck-out dampening plus the extra selective ply destabilise the same setup ordering.
This is the identical pattern already recorded for fable (SOT-1836 / SOT-1864): raw depth / breadth /
tempo increases **do not convert to strength** in this engine. take runs a deterministic rule search
three orders of magnitude smaller than fable's MCTS, but the same lesson holds — the unused time budget
is *not* the binding constraint on take's playing strength.

## Action

- Behaviour **reverted** (`git revert` of the candidate commit); the promoted champion
  `take-adaptive-tempo-v1` is unchanged and remains the submission. Kaggle 575.5 stands — **no
  resubmission** (only a promoted candidate would trigger the exec-compat gate + resubmit path).
- This document is the recorded rationale. Future take work should target **behaviour quality**
  (scoring / prize-trade discipline against matsu-class opponents), not more search in the unused
  time budget.
