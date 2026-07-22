# SOT-1854 Kaggle 実戦 trace 分析とルール候補 A/B

## 結論

Kaggle 初回提出 `54904755` は **COMPLETE / public score 553.0** に収束した。
現 champion のローカル trace では未登録 context と fault はともに 0 であり、ランダム
fallback の再発はなかった。一方、Kaggle の `deck.csv` という stem が SOT-1734 の
適応探索対象表に含まれず、本番では同ルールが無効になる候補穴を特定した。

この穴を塞ぐ候補（`ADAPTIVE_SEARCH_DECKS` に `deck` を追加）を現 champion
`origin/main@873e295` と直接 A/B したが、有意改善を再現できなかったため**非採用**とした。
本 PR は実験記録のみで、agent の挙動を変更しない。非採用なので Kaggle 再提出条件は
発生しない。

## Kaggle 実績

2026-07-22 に Kaggle API で submissions を再取得した。

| ref | commit | status | public score |
| --- | --- | --- | ---: |
| 54904755 | 873e295 | COMPLETE | **553.0** |

Kaggle submissions API がこの提出について返す実戦情報は集約 score/status までで、個別
episode trace は公開されない。そのため敗着分類は同一 engine・ルールによるローカル
LOGS trace を代理データとして行った。

## 現 champion の trace 集計

`eval/trace_gap_report.py` を 25 デッキ × 10 戦（seed 1854）で実行した。

| 指標 | 結果 |
| --- | ---: |
| games / decisions | 250 / 72,155 |
| unhandled-context fallback | **0** |
| fault | **0** |
| prize_out | 155 (64.9%) |
| deck_out | 79 (33.1%) |
| no_active | 5 (2.1%) |

過去の大標本（prize 71% / deck_out 27% / no_active 2%）と同じく、支配的敗因は
prize_out、次点が deck_out であり、新たな context fallback 穴は観測されなかった。

## 候補と A/B

候補は Kaggle の既定 `RuleBasedAgent(deck_path="deck.csv")` でも SOT-1734 適応探索を
有効にする一行変更。対 Random と現 champion との side-swap A/B を同時に行う
`eval/regression.py --old-ref origin/main` で評価した。

| run | matchup | W-L-D | 勝率 | Wilson 95% CI | fault |
| --- | --- | ---: | ---: | --- | ---: |
| N=400, seed=1854 | candidate vs champion | 213-187-0 | 0.5325 | [0.4835, 0.5809] | 0 |
| N=400, seed=1854 | candidate vs Random | 351-49-0 | 0.8775 | [0.8417, 0.9061] | 0 |
| N=2000, seed=2854 | candidate vs champion | 983-1017-0 | 0.4915 | [0.4696, 0.5134] | 0 |
| N=2000, seed=2854 | candidate vs Random | 1793-207-0 | 0.8965 | [0.8824, 0.9091] | 0 |

対 champion の大標本 CI 下限は 0.5 を超えず、点推定も 0.5 未満に戻った。したがって
「有意改善時のみ採用」のゲートは不成立。対 Random は十分強く fault 0 だが、これは
candidate が champion より強い根拠にはならない。

対 matsu の再測定には `eval/battle_vs_matsu.py` を試したが、現 main の compatibility
bootstrap と worktree sandbox の組合せで contestant import が開始前に失敗したため、試合
データへ混入させず棄却した。既存の正常な集約値は take 13-37、勝率 0.2600、Wilson 95%
[0.1587, 0.3955]、fault 0。今回候補は直接 A/B で非採用が確定したため、壊れた対 matsu
計測を採用根拠には使用しない。

## 再現コマンド

```bash
kaggle competitions submissions -c pokemon-tcg-ai-battle --csv
venv/bin/python eval/trace_gap_report.py --games-per-deck 10 --seed 1854 \
  --out-root /tmp/sot1854-baseline-traces --json /tmp/sot1854-baseline-gap.json
venv/bin/python eval/regression.py --games 400 --seed 1854 --old-ref origin/main
venv/bin/python eval/regression.py --games 2000 --seed 2854 --old-ref origin/main
```
