# 竹 compatibility adapter migration runbook

竹 keeps the existing `RuleBasedAgent` as the production default while exposing it through the common
`ptcg-deck-strategy/v1` / `ptcg-agent-adapter/v1` compatibility boundary. The switch is read once when
`main.py` starts, so each match process uses one stable mode.

| Mode | `PTCG_TAKE_MIGRATION_MODE` | Authoritative result | Purpose |
| --- | --- | --- | --- |
| Legacy | unset or `legacy` | existing direct agent | default and rollback |
| Shadow | `shadow` | existing direct agent | compare candidate without gameplay risk |
| Core | `core` | versioned strategy adapter | staged cutover |

Shadow mode writes one JSON object per decision to stderr. `matched` reports exact option-index parity;
stdout remains reserved for the battle protocol. Before cutover, run representative fixtures and battles
in `shadow` and require zero mismatches and zero agent/engine faults.

```bash
PTCG_TAKE_MIGRATION_MODE=shadow venv/bin/python agents/test_compatibility.py
PTCG_TAKE_MIGRATION_MODE=core venv/bin/python eval/arena.py --games 4 --agent-b random --workers 1
```

## Rollback

Set `PTCG_TAKE_MIGRATION_MODE=legacy` (or remove it) and restart the agent process. Verify startup by
requesting the initial observation and confirming a 60-card deck is returned, then run one fixture match.
No source revert or artifact rebuild is required. Unknown modes and incompatible future strategy/adapter
versions fail at startup instead of silently choosing a path.
