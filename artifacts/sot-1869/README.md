# SOT-1869 runtime promotion evidence

The bundled `take-adaptive-tempo-v1` profile is now the default production
runtime profile and is included in the Kaggle submission archive.

- Implementation commit: `f354f9c0aca005a8f15dbcb7695ac8ce970d2271`
- Deck SHA-256: `42068a1803902756badcfd418f6f348b7901365a281d78af0692cbf2589f0799`
- Seed / seats: `1869`, 100 games per card, paired seat swap
- Candidate vs previous runtime: 56-44, Wilson 95% `[0.4623, 0.6533]` — non-regression promotion gate PASS
- Candidate vs random strength floor: 90-10, Wilson 95% `[0.8256, 0.9448]` — hard floor PASS
- Safety: fault 0, unfinished 0, illegal action 0
- Runtime: max observed decision 0.33 ms; profile budget 250 ms; competition budget 600 s
- Submission SHA-256: `23aa551c9bb1a2a3eacb18cd2a783ce0eab296ca03942a2d4752646a5b58f3ec`

Raw card results are in `regression.json`; normalized promotion metadata is in
`promotion.json`.
