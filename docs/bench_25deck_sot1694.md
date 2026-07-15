# SOT-1694 25デッキ mirror rotation — 新スコアリング vs 現行v2 (old = git-ref `main`)

- 総合: 604-646-0 (N=1250, decided=1250) 勝率 **0.483** Wilson95 [0.456, 0.511] → 有意差なし (劣化なし)
- fault: 全体 0 (new=0, old=0)
- 思考時間/decision: p95=0.23ms max=43.12ms

| archetype | W-L | 勝率 | Wilson 95% CI |
| --- | ---: | ---: | --- |
| control | 23-27 | 0.460 | [0.330, 0.596] |
| ex_mega | 185-215 | 0.463 | [0.414, 0.511] |
| midrange | 30-20 | 0.600 | [0.462, 0.724] |
| single_prize | 69-81 | 0.460 | [0.382, 0.540] |
| stage2 | 297-303 | 0.495 | [0.455, 0.535] |

| deck | archetype | W-L-D | 勝率 | CI | fault |
| --- | --- | --- | ---: | --- | ---: |
| 01_dragapult | stage2 | 19-31-0 | 0.38 | [0.26, 0.52] | 0 |
| 02_raging_bolt_ogerpon | ex_mega | 21-29-0 | 0.42 | [0.29, 0.56] | 0 |
| 03_dragapult_blaziken | stage2 | 25-25-0 | 0.50 | [0.37, 0.63] | 0 |
| 04_dragapult_dusknoir | stage2 | 25-25-0 | 0.50 | [0.37, 0.63] | 0 |
| 05_dragapult_dudunsparce | stage2 | 24-26-0 | 0.48 | [0.35, 0.61] | 0 |
| 06_hydrapple | stage2 | 28-22-0 | 0.56 | [0.42, 0.69] | 0 |
| 07_n_s_zoroark_n | ex_mega | 23-27-0 | 0.46 | [0.33, 0.60] | 0 |
| 08_ogerpon_box | ex_mega | 27-23-0 | 0.54 | [0.40, 0.67] | 0 |
| 09_slowking | stage2 | 29-21-0 | 0.58 | [0.44, 0.71] | 0 |
| 10_hop_s_trevenant | single_prize | 23-27-0 | 0.46 | [0.33, 0.60] | 0 |
| 11_lillie_s_clefairy | ex_mega | 25-25-0 | 0.50 | [0.37, 0.63] | 0 |
| 12_alakazam_dudunsparce | stage2 | 18-32-0 | 0.36 | [0.24, 0.50] | 0 |
| 13_festival_lead | single_prize | 21-29-0 | 0.42 | [0.29, 0.56] | 0 |
| 14_mega_lucario_ex | ex_mega | 21-29-0 | 0.42 | [0.29, 0.56] | 0 |
| 15_marnie_s_grimmsnarl_ex | stage2 | 30-20-0 | 0.60 | [0.46, 0.72] | 0 |
| 16_crustle_mysterious_rock_inn | control | 23-27-0 | 0.46 | [0.33, 0.60] | 0 |
| 17_rocket_s_mewtwo_ex | midrange | 30-20-0 | 0.60 | [0.46, 0.72] | 0 |
| 18_rocket_s_honchkrow | single_prize | 25-25-0 | 0.50 | [0.37, 0.63] | 0 |
| 19_ethan_s_typhlosion | stage2 | 23-27-0 | 0.46 | [0.33, 0.60] | 0 |
| 20_cynthia_s_garchomp_ex | stage2 | 26-24-0 | 0.52 | [0.39, 0.65] | 0 |
| 21_lillie_s_clefairy_ex_naic_champion | ex_mega | 25-25-0 | 0.50 | [0.37, 0.63] | 0 |
| 22_dragapult_ex_naic_2nd | stage2 | 27-23-0 | 0.54 | [0.40, 0.67] | 0 |
| 23_slowking_naic_4th | stage2 | 23-27-0 | 0.46 | [0.33, 0.60] | 0 |
| 24_n_s_zoroark_ex_naic_10th | ex_mega | 19-31-0 | 0.38 | [0.26, 0.52] | 0 |
| 25_mega_lopunny_ex | ex_mega | 24-26-0 | 0.48 | [0.35, 0.61] | 0 |
