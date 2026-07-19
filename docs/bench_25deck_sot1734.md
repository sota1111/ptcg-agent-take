# SOT-1734 適応探索 — 25デッキ mirror baseline比較

- 条件: candidate vs `main@49f31f0`, 25デッキ各80戦、先後交替、seed 20260719
- 総合: **985-954-61** (N=2000, decided=1939)、勝率 **0.5080**、Wilson 95% CI **[0.4857, 0.5302]**
- baseline期待値: 同一エージェント mirror = 0.5000 → point estimate **+0.0080**
- 先攻: 504-465、勝率 0.5201、Wilson 95% CI [0.4887, 0.5514]
- 後攻: 481-489、勝率 0.4959、Wilson 95% CI [0.4645, 0.5273]
- fault: **0** (candidate=0, baseline=0)
- 思考時間: p95 0.131ms、max 11.21ms

先後とも Wilson CI が 0.5 を含み、重大な片側回帰は検出されなかった。

| deck | W-L-D | winrate | Wilson 95% CI |
| --- | ---: | ---: | --- |
| 01_dragapult | 46-34-0 | 0.575 | [0.466, 0.677] |
| 02_raging_bolt_ogerpon | 36-44-0 | 0.450 | [0.345, 0.559] |
| 03_dragapult_blaziken | 32-48-0 | 0.400 | [0.300, 0.509] |
| 04_dragapult_dusknoir | 42-38-0 | 0.525 | [0.417, 0.631] |
| 05_dragapult_dudunsparce | 40-40-0 | 0.500 | [0.393, 0.607] |
| 06_hydrapple | 39-41-0 | 0.488 | [0.381, 0.595] |
| 07_n_s_zoroark_n | 49-31-0 | 0.613 | [0.503, 0.712] |
| 08_ogerpon_box | 43-37-0 | 0.538 | [0.429, 0.642] |
| 09_slowking | 21-23-36 | 0.477 | [0.338, 0.619] |
| 10_hop_s_trevenant | 38-42-0 | 0.475 | [0.369, 0.583] |
| 11_lillie_s_clefairy | 41-39-0 | 0.513 | [0.405, 0.619] |
| 12_alakazam_dudunsparce | 46-34-0 | 0.575 | [0.466, 0.677] |
| 13_festival_lead | 37-43-0 | 0.463 | [0.358, 0.571] |
| 14_mega_lucario_ex | 43-37-0 | 0.538 | [0.429, 0.642] |
| 15_marnie_s_grimmsnarl_ex | 40-40-0 | 0.500 | [0.393, 0.607] |
| 16_crustle_mysterious_rock_inn | 41-39-0 | 0.513 | [0.405, 0.619] |
| 17_rocket_s_mewtwo_ex | 35-45-0 | 0.438 | [0.334, 0.547] |
| 18_rocket_s_honchkrow | 38-42-0 | 0.475 | [0.369, 0.583] |
| 19_ethan_s_typhlosion | 44-36-0 | 0.550 | [0.441, 0.655] |
| 20_cynthia_s_garchomp_ex | 43-37-0 | 0.538 | [0.429, 0.642] |
| 21_lillie_s_clefairy_ex_naic_champion | 46-34-0 | 0.575 | [0.466, 0.677] |
| 22_dragapult_ex_naic_2nd | 41-39-0 | 0.513 | [0.405, 0.619] |
| 23_slowking_naic_4th | 33-22-25 | 0.600 | [0.468, 0.719] |
| 24_n_s_zoroark_ex_naic_10th | 39-41-0 | 0.488 | [0.381, 0.595] |
| 25_mega_lopunny_ex | 32-48-0 | 0.400 | [0.300, 0.509] |

## 判定

25デッキ全件実行、全体point estimate改善、先後の有意な重大回帰なし、fault 0。受け入れ条件を満たす。
