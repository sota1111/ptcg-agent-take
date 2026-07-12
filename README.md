# ptcg-agent-ume

Agent & local evaluation environment for the **Pokémon TCG AI Battle Challenge** (Kaggle).

- Competition (Simulation): https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- Competition (Strategy):   https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy

## ⚠️ License note
The battle engine (`cg/`, `libcg.so`) and card data (`data/`) are **competition-use-only and must not
be redistributed**. They are **gitignored** and never committed. Only our own code
(`main.py`, `deck.csv`, `eval/`, `scripts/`) lives in git.

## Layout
```
main.py              # submission entry: agent(obs_dict) -> list[int]  (tracked)
deck.csv             # our 60-card deck                                (tracked)
eval/run_match.py    # local self-play match runner                   (tracked)
eval/record_match.py # one match → JSONL trace (schema in trace.py)    (tracked)
eval/arena.py        # N-match arena: side-swap pairs, parallel        (tracked)
scripts/             # setup + build helpers                          (tracked)
cg/                  # cabt engine bindings (gitignored, license)
data/                # card CSVs (gitignored, license)
traces/ under eval/  # recorded match traces (gitignored, license)
```

## Setup
```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash scripts/setup_engine.sh          # copies cg/ + data/ from the Kaggle download
venv/bin/python eval/run_match.py     # run one local self-play match
```

## Multi-match evaluation
Run **N matches** of an agent-vs-agent card in one shot, with **side-swap pairing**
(each agent takes seat 0 in exactly half the matches, removing first-player bias)
and **process-pool parallelism** (E2: one match per process). Every match writes a
JSONL trace; a `report.json` summarises win rate, side balance, and failure
categories. Abnormal matches (agent exception / illegal move / timeout) are scored
as a loss for the offending agent, matching Kaggle's "agent error = loss" rule.
```bash
venv/bin/python eval/arena.py --games 100 --seed 42          # 100 matches, 50 pairs
venv/bin/python eval/arena.py --games 100 --agent-b random --workers 8 --level result
venv/bin/python eval/test_arena.py                           # standalone tests
```
The engine takes no seed (E1), so match *outcomes* are not reproducible; only the
*agent-side* RNG is (deterministic per-match seed derived from `--seed`).

## Aggregation report
Turn a directory of traces into a statistical summary: **win rate + Wilson 95% CI**
per agent, **decision-reason distribution**, **first/second-player win rate**, a
**deck × deck matchup table**, and turn / decision / per-decision thinking-time
distributions. Draws, truncated matches and abnormal (failure) losses are tallied
*separately* from normal decided games so they never skew the win rate.
```bash
venv/bin/python eval/report.py eval/traces/arena_<ts>        # text summary
venv/bin/python eval/report.py <dir> --json report.json      # also dump JSON
venv/bin/python eval/test_report.py                          # standalone tests
```
Per-decision thinking times require traces recorded at `--level logs` (RESULT-level
traces carry no decision records).

## Build a submission
```bash
bash scripts/build_submission.sh      # -> submission.tar.gz (main.py + deck.csv + cg/)
```
