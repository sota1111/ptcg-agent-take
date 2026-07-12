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
eval/counterfactual.py # "what if?" replay of a recorded position     (tracked)
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

## Game-record rendering / record-based replay
Render one match trace as a **human-readable game record** to review "why did this
side win/lose" and "was each decision reasonable" at the single-match level. Card
ids and attack ids are resolved to names via the engine masters
(`all_card_data()` / `all_attack()`) and the `data/*_Card_Data.csv` files (unknown
ids fall back to `#<id>`); each decision is shown from the acting side's viewpoint
(E4, opponent hand hidden) with its legal moves and chosen move. The engine takes
no seed (E1), so this recorded trace is the only faithful replay.
```bash
venv/bin/python eval/replay.py eval/traces/match.jsonl            # full record (ja names)
venv/bin/python eval/replay.py <trace.jsonl> --lang en            # English names
venv/bin/python eval/replay.py <trace.jsonl> --scenes             # decisive scenes only
venv/bin/python eval/replay.py <trace.jsonl> --scenes --hp-threshold 60
venv/bin/python eval/test_replay.py                               # standalone tests
```
`--scenes` extracts decisive decisions — knockouts, large HP swings
(`--hp-threshold`), and the last few decisions before the result. Rendering needs a
trace recorded at `--level logs` (RESULT-level traces carry no decision records).

## Counterfactual analysis ("what if I had played a different move?")
The final stage of decision auditing (E5): from a recorded position, roll out the
**actual** choice vs one or more **alternative** options and compare the outcomes. It
uses the engine search API (`search_begin`/`search_step`/`search_end`/`search_release`),
keyed by the `search_begin_input` stored in each decision, to reconstruct the position.
Hidden information (your deck/prize, the opponent's deck/prize/hand, a face-down active)
is supplied by a **swappable predictor** — the default `UniformDeckPredictor` uniformly
samples the cards not visible on the board from the known deck composition. With
`--manual-coin 1` (default) coin flips are held fixed, so a bounded rollout is
reproducible. The search session is **always released** (try/finally).
```bash
venv/bin/python eval/counterfactual.py <trace.jsonl> --decision 35   # actual vs alternatives
venv/bin/python eval/counterfactual.py <trace.jsonl> --decision 35 --alt-option 2 --max-depth 200
venv/bin/python eval/counterfactual.py <trace.jsonl> --decision 35 --rollouts 20 --coin random --json
venv/bin/python eval/test_counterfactual.py                          # standalone tests
```
Reconstruction needs the full observation, so record the trace at **`--level full_obs`**
(`eval/record_match.py`); a LOGS-only trace is rejected with a clear error. The rollout
policy defaults to a seeded random baseline (swap in a real agent via the library API
`analyze_decision(..., policy_factory=...)`). Rollouts deep into the late game can hit
the engine's seedless internal shuffles (C2), after which the search state may become
inconsistent — the tool records that as a per-branch error and still releases the
session; use bounded depth and/or a stronger predictor for clean comparisons.

## Build a submission
```bash
bash scripts/build_submission.sh      # -> submission.tar.gz (main.py + deck.csv + cg/)
```
