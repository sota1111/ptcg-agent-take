# SOT-1694 25デッキ mirror トレース集計 — fallback穴とアーキタイプ別敗因

- ミラー自己対戦: 各デッキ 20 試合 × 25 デッキ = 500 試合 (decided 500, failures 0)
- 総決定数: 76376, seed=4694, traces: `eval/traces/sot1694_gap_after`
- 思考時間: p95=0.21ms max=1.05ms / decision

## (a) CONTEXT_HANDLERS 未登録 context への random fallback

未登録contextでの決定 = **0** / 76376 (0.0%)。TO_HAND のバウンス効果 defer（形状検出可能な既知の穴）= 0 決定。

| context | fallback決定数 | 全決定比 |
| --- | ---: | ---: |

### 改修前後比較 (before = 現行v2, after = 本Issue改修後)

| context | before (500試合) | after (500試合) |
| --- | ---: | ---: |
| TO_BENCH | 577 | 0 |
| ACTIVATE | 514 | 0 |
| SWITCH_ENERGY_CARD | 155 | 0 |
| SKILL_ORDER | 108 | 0 |
| ATTACK | 69 | 0 |
| DISCARD_TOOL_CARD | 63 | 0 |
| SWITCH_ENERGY | 54 | 0 |
| EVOLVE | 52 | 0 |
| DETACH_FROM | 50 | 0 |
| FIRST_EFFECT | 46 | 0 |
| TO_PRIZE | 23 | 0 |
| EVOLVES_TO | 13 | 0 |
| DISABLE_ATTACK | 7 | 0 |
| **合計 (全決定比)** | **1731** (2.4%) | **0** (0.0%) |

## (b) アーキタイプ別敗因分布

敗因 = 敗者側の決着理由 (prize_out=サイド取り切り / deck_out=山札切れ / no_active=場切れ / card_effect=カード効果)。

| archetype | 決着数 | no_active | prize_out | deck_out | card_effect | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 20 | 0 (0.0%) | 16 (80.0%) | 4 (20.0%) | 0 (0.0%) | 0 (0.0%) |
| ex_mega | 160 | 1 (0.6%) | 115 (71.9%) | 44 (27.5%) | 0 (0.0%) | 0 (0.0%) |
| midrange | 20 | 0 (0.0%) | 11 (55.0%) | 9 (45.0%) | 0 (0.0%) | 0 (0.0%) |
| single_prize | 60 | 6 (10.0%) | 29 (48.3%) | 25 (41.7%) | 0 (0.0%) | 0 (0.0%) |
| stage2 | 240 | 4 (1.7%) | 185 (77.1%) | 51 (21.2%) | 0 (0.0%) | 0 (0.0%) |
| **全体** | 500 | 11 (2.2%) | 356 (71.2%) | 133 (26.6%) | 0 (0.0%) | 0 (0.0%) |

## デッキ別内訳

| deck | archetype | games | 平均turn | fallback決定 | no_active | prize_out | deck_out | fault |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01_dragapult | stage2 | 20 | 35.5 | 0 | 1 | 12 | 7 | 0 |
| 02_raging_bolt_ogerpon | ex_mega | 20 | 27.1 | 0 | 1 | 19 | 0 | 0 |
| 03_dragapult_blaziken | stage2 | 20 | 34.5 | 0 | 0 | 17 | 3 | 0 |
| 04_dragapult_dusknoir | stage2 | 20 | 30.6 | 0 | 0 | 19 | 1 | 0 |
| 05_dragapult_dudunsparce | stage2 | 20 | 29.6 | 0 | 0 | 19 | 1 | 0 |
| 06_hydrapple | stage2 | 20 | 29.1 | 0 | 0 | 20 | 0 | 0 |
| 07_n_s_zoroark_n | ex_mega | 20 | 44.5 | 0 | 0 | 7 | 13 | 0 |
| 08_ogerpon_box | ex_mega | 20 | 39.4 | 0 | 0 | 18 | 2 | 0 |
| 09_slowking | stage2 | 20 | 35.4 | 0 | 0 | 14 | 6 | 0 |
| 10_hop_s_trevenant | single_prize | 20 | 31.2 | 0 | 3 | 11 | 6 | 0 |
| 11_lillie_s_clefairy | ex_mega | 20 | 27.2 | 0 | 0 | 20 | 0 | 0 |
| 12_alakazam_dudunsparce | stage2 | 20 | 35.9 | 0 | 0 | 8 | 12 | 0 |
| 13_festival_lead | single_prize | 20 | 33.3 | 0 | 3 | 9 | 8 | 0 |
| 14_mega_lucario_ex | ex_mega | 20 | 24.6 | 0 | 0 | 20 | 0 | 0 |
| 15_marnie_s_grimmsnarl_ex | stage2 | 20 | 21.8 | 0 | 2 | 18 | 0 | 0 |
| 16_crustle_mysterious_rock_inn | control | 20 | 29.1 | 0 | 0 | 16 | 4 | 0 |
| 17_rocket_s_mewtwo_ex | midrange | 20 | 31.9 | 0 | 0 | 11 | 9 | 0 |
| 18_rocket_s_honchkrow | single_prize | 20 | 40.0 | 0 | 0 | 9 | 11 | 0 |
| 19_ethan_s_typhlosion | stage2 | 20 | 43.3 | 0 | 0 | 15 | 5 | 0 |
| 20_cynthia_s_garchomp_ex | stage2 | 20 | 35.7 | 0 | 1 | 9 | 10 | 0 |
| 21_lillie_s_clefairy_ex_naic_champion | ex_mega | 20 | 31.9 | 0 | 0 | 19 | 1 | 0 |
| 22_dragapult_ex_naic_2nd | stage2 | 20 | 29.9 | 0 | 0 | 18 | 2 | 0 |
| 23_slowking_naic_4th | stage2 | 20 | 31.6 | 0 | 0 | 16 | 4 | 0 |
| 24_n_s_zoroark_ex_naic_10th | ex_mega | 20 | 50.4 | 0 | 0 | 7 | 13 | 0 |
| 25_mega_lopunny_ex | ex_mega | 20 | 43.3 | 0 | 0 | 5 | 15 | 0 |

## 結論・検証結果 (SOT-1694)

### 実装
- **fallback穴 13 context を全て手当て** — TO_BENCH / TO_FIELD / ACTIVATE / SWITCH_ENERGY(_CARD) /
  SKILL_ORDER / ATTACK / DISABLE_ATTACK / EVOLVE / EVOLVES_TO / DETACH_FROM / DISCARD_TOOL_CARD /
  FIRST_EFFECT / TO_PRIZE に汎用属性ベースのハンドラを追加（カードID直書きなし）。
  random fallback 決定は 500試合あたり **1731 (2.4%) → 0 (0.0%)**。
- **アーキタイプ適応** (`agents/archetype.py`) — デッキ構成（進化深度・ex/メガ比率・単一プライズ
  アタッカー比率・エネルギー分布）から純関数でスコアリングバンド係数を導出。係数は ±150 に有界で
  バンド順序 (200間隔) を跨がない。
- **敗因ガード** — 場切れ: ベンチ0枚時の Basic 展開を勝利攻撃の次点に昇格 (S_BENCH_INSURANCE)・
  doomed-Active guard の EVOLVE 拡張。山札切れ: 残デッキ ≤6 で DRAW_COUNT 最小draw /
  ACTIVATE=NO / サポーター見送り (DECK_LOW_THRESHOLD)。

### 検証（受け入れ条件との対応）
- **25デッキ mirror 新vs現行v2**: 最終確認 N=1250 で 604-646, 勝率 0.483 Wilson95 [0.456, 0.511]
  → 不採用条件（点推定<0.5 かつ CI上限<0.5）は**不成立 = 劣化なし**。同一方策の独立run
  (N=999, 0.519) と合算すると N=2249 で **0.499 [0.478, 0.520]** — v2 と統計的に同等。
  詳細は `docs/bench_25deck_sot1694.md`。
- **vs random**: 400試合 354-46 勝率 **0.885** Wilson95 [0.850, 0.913] — ゲート ≥0.85 維持。
- **fault / 違法手 0**（全ベンチ・全トレース run 通算 3000+ 試合）。思考時間 p95 ≈ 0.2ms/決定。
- **チャンピオンデッキ (deck.csv) new_vs_old**: 400試合 216-184 勝率 0.540 [0.491, 0.588]（回帰なし）。

### 学び（正直な負の結果を含む）
- Issue 前提の「ミラー敗因の93%が no-active」は旧チャンピオンデッキ単体ミラー (SOT-1682) の
  性質で、**25大会デッキには汎化しない**: 実測は prize_out 71% / deck_out 27% / no_active 2%。
  no-active 緩和よりも deck_out（特に長期戦デッキ: N's Zoroark 系・Mega Lopunny の 13-16/20）
  が構造的な穴。
- **バンド順序を跨がない範囲の適応係数は、この scoring アーキテクチャでは挙動にほぼ寄与しない**
  （同一バンド内の全選択肢が等しくシフトし、跨ぎは有界性により起きないため）。唯一の実効ノブ
  だったサポーター手札閾値の適応 (6→8) は N=1000 ablation で 0.497 vs 0.519 と劣位のため棄却し、
  v2 値 6 に固定。適応を「効かせる」には帯域間の意図的な跨ぎ（=順序変更）を許す設計が必要で、
  それは本Issueの「劣化なし必須」ゲートの下では正当化できなかった。
