# Board-wipe KPI and survival candidate (SOT-1884)

Take now records the terminal board for every arena match and classifies a loss
as `board_wipe` only when the losing side has neither an Active nor a Bench
Pokémon. Candidate and champion reports include the count, rate among losses,
and avoidance rate.

The `survival` candidate adds a bounded preference to the existing scoring
policy. Its risk estimate uses:

- the Active Pokémon's remaining HP;
- the opponent's strongest next-turn energy-payable damage;
- Bench Pokémon and viable attacker/switch replacement routes.

The production `scoring` policy remains the champion. The candidate is selected
explicitly for A/B evaluation and cannot change production behavior without a
successful promotion decision.

## Small-N screen

The frozen 25-deck rotation was evaluated with one side-swapped pair per deck
(50 matches), seed `20260723`.

| Metric | Candidate | Champion |
| --- | ---: | ---: |
| Wins / losses | 20 / 29 | 29 / 20 |
| Win rate | 0.4082 | 0.5918 |
| Wilson 95% CI | [0.2822, 0.5475] | — |
| board_wipe count | 0 | 0 |
| board_wipe rate in losses | 0.0000 | 0.0000 |
| board_wipe avoidance rate | 1.0000 | 1.0000 |
| Faults | 0 combined | 0 combined |
| Throughput | 23.641 sims/sec combined | 23.641 sims/sec combined |

Evidence: `artifacts/sot-1884/screen.json`.

## Promotion decision

The screen gate requires the candidate Wilson 95% lower bound to exceed 0.5
and fault count to remain zero. Faults passed, but the lower bound was 0.2822,
so the candidate failed the screen.

Per the gated procedure, large-N confirmation was not run, the champion
`scoring` policy was not modified, and Kaggle resubmission was not performed.
