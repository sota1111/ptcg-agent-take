# SOT-1874 runtime integration evidence

The hardened `take-adaptive-tempo-v1` profile is wired into the production
`main.agent` path with explicit 600-second match, eight-hour experiment, and
two-game checkpoint contracts. `eval/runtime_league.py` exercises real sibling
submission entrypoints with committed decks and paired seat swaps.

## Results

- 1,600-game fixed-seed, seat-balanced A/B vs `35c2d89`: 816-784 (51.00%),
  Wilson 95% `[0.4855, 0.5344]`. The requested strict lower-bound gate did not
  pass.
- Current real-runtime league (20 games each vs Sol, Debate, Fable, Zero):
  35-45, average 43.75%.
- Historical runtime league at `35c2d89` under the identical fixed-seed,
  seat-balanced 80-game schedule: 34-46, average 42.50%. The current runtime
  improves the league average by 1.25 percentage points, so the alternative
  strength gate passes.
- Both league runs: faults 0, unfinished 0, illegal actions 0.
- Current Take maximum observed decision: 0.560 ms (250 ms profile search
  budget; 600 s competition ceiling).
- Adaptation remains explicit and bounded: weight 0.32, risk band 0.35–0.65,
  max depth 3, extension branching limit 10.
- `submission.tar.gz` built successfully and contains the promoted profile.

The complete 80-game current and historical match records are stored in
`runtime-league-20.json` and `old-runtime-league-20.json`; their corresponding
checkpoint files demonstrate atomic two-game resume state but are intentionally
not duplicated in the committed evidence.

The implementation, strength, safety, and packaging gates pass. The primary
old-runtime Wilson gate remains below its strict threshold, while the specified
alternative real-runtime league-improvement gate passes.
