# SOT-1874 runtime integration evidence

The hardened `take-adaptive-tempo-v1` profile is wired into the production
`main.agent` path with explicit 600-second match, eight-hour experiment, and
two-game checkpoint contracts. `eval/runtime_league.py` exercises real sibling
submission entrypoints with committed decks and paired seat swaps.

## Results

- 1,600-game fixed-seed, seat-balanced A/B vs `35c2d89`: 816-784 (51.00%),
  Wilson 95% `[0.4855, 0.5344]`. The requested strict lower-bound gate did not
  pass.
- Current real-runtime league (2 games each vs Sol, Debate, Fable, Zero): 3-5,
  average 37.5%.
- Historical runtime league at `35c2d89`: 4-4, average 50.0%. The alternative
  league-improvement gate did not pass in this smoke sample.
- Both league runs: faults 0, unfinished 0, illegal actions 0.
- Current Take maximum observed decision: 0.444 ms (250 ms profile search
  budget; 600 s competition ceiling).
- Adaptation remains explicit and bounded: weight 0.32, risk band 0.35–0.65,
  max depth 3, extension branching limit 10.
- `submission.tar.gz` built successfully and contains the promoted profile.

The implementation and safety/packaging gates pass, but strength acceptance is
`FAIL`; no PR is created until a policy/profile tuning cycle clears either
strength criterion.
