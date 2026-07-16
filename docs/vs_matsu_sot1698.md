# SOT-1698 対松 敗因分類 — 竹 (RuleBased) vs 松 (MCTS champion)

> 本レポートは現 champion 竹（SOT-1694 v2）の対松敗因プロファイル。これを踏まえ試作した
> プライズレース計画ルールは 25デッキ mirror では中立（`docs/bench_25deck_sot1698.md`）で、
> 昇格ゲート（CI 下限 > 0.5）不成立のため未採用・champion 維持。対松の勝率改善は本 issue の
> 範囲では未確証（松 MCTS が重く CI 分離まで N を回すコストが高い＝issue でも「参考値」扱い）。

- 対戦: 竹 vs 松、mirror（同一デッキ両者・先後入替）、50 試合（decided 50, draw 0, unfinished 0）
- デッキプール: 25 decks (decks/initial), seed=1698
- fault: 竹=0 松=0
- **竹 対松勝率: 0.26 Wilson95 [0.1587, 0.3955]** （竹 13 / 松 37）

## 竹の敗因内訳（decided 竹敗北）

| 敗因 | 件数 | 敗北比 |
| --- | ---: | ---: |
| prize_out | 31 | 83.8% |
| deck_out | 4 | 10.8% |
| no_active | 2 | 5.4% |
| card_effect | 0 | 0.0% |
| other | 0 | 0.0% |
| **合計** | **37** | 100% |

SOT-1694 の 25デッキ *mirror*（同系対戦）実測は prize 71% / deck_out 27% / no_active 2%。上表と比較して対探索型（松）で敗因構成が変わるかを確認する。

## デッキ別（竹視点）

| deck | 竹W-松W | 竹勝率 | prize_out | deck_out | no_active | fault(竹/松) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 02_raging_bolt_ogerpon.csv | 1-3 | 0.25 | 3 | 0 | 0 | 0/0 |
| 06_hydrapple.csv | 0-2 | 0.0 | 2 | 0 | 0 | 0/0 |
| 07_n_s_zoroark_n.csv | 2-0 | 1.0 | 0 | 0 | 0 | 0/0 |
| 08_ogerpon_box.csv | 0-2 | 0.0 | 2 | 0 | 0 | 0/0 |
| 09_slowking.csv | 1-1 | 0.5 | 1 | 0 | 0 | 0/0 |
| 10_hop_s_trevenant.csv | 0-2 | 0.0 | 2 | 0 | 0 | 0/0 |
| 11_lillie_s_clefairy.csv | 2-4 | 0.3333 | 4 | 0 | 0 | 0/0 |
| 13_festival_lead.csv | 1-5 | 0.1667 | 3 | 2 | 0 | 0/0 |
| 14_mega_lucario_ex.csv | 2-4 | 0.3333 | 4 | 0 | 0 | 0/0 |
| 15_marnie_s_grimmsnarl_ex.csv | 0-2 | 0.0 | 1 | 0 | 1 | 0/0 |
| 16_crustle_mysterious_rock_inn.csv | 0-2 | 0.0 | 2 | 0 | 0 | 0/0 |
| 17_rocket_s_mewtwo_ex.csv | 1-3 | 0.25 | 2 | 1 | 0 | 0/0 |
| 18_rocket_s_honchkrow.csv | 1-1 | 0.5 | 1 | 0 | 0 | 0/0 |
| 19_ethan_s_typhlosion.csv | 0-2 | 0.0 | 1 | 0 | 1 | 0/0 |
| 20_cynthia_s_garchomp_ex.csv | 0-2 | 0.0 | 2 | 0 | 0 | 0/0 |
| 22_dragapult_ex_naic_2nd.csv | 2-0 | 1.0 | 0 | 0 | 0 | 0/0 |
| 23_slowking_naic_4th.csv | 0-2 | 0.0 | 1 | 1 | 0 | 0/0 |
