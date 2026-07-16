# SOT-1698 昇格ゲート — プライズレース試作 vs 現champion（**判定: 非昇格・champion維持**）

> **結論先出し（マージ方針）**: 試作した「プライズレース優先 promotion + deck-out レース
> swing」ルールは 25 デッキ mirror で champion と **統計的に同等**（pooled 0.5004, CI 下限
> 0.4916 < 0.5）。issue の昇格ゲート（CI 下限 > 0.5 のときのみ昇格）を満たさないため
> **昇格せず、champion（SOT-1694 v2）を維持**する。この PR には試作ルールの behavior 変更は
> **含めない**（`agents/rule_based.py` は main のまま）。本ドキュメントはその実験記録であり、
> 再利用可能な成果は対松敗因ハーネス（`eval/battle_vs_matsu.py` / `eval/agent_server.py`）と
> 敗因分類（`docs/vs_matsu_sot1698.md`）。

試作候補 = プライズレース優先 promotion（`to_active` の即KO優先）+ deck-out レース swing。
champion = git-ref `main`（SOT-1694 v2）。`eval/bench_25deck_rotation.py` で 25 デッキ mirror
（同一デッキ両者・先後入替）paired ベンチにより A/B 評価した（下表は試作候補を作業ツリーに
適用した時点で採取した測定値の記録）。

ゲート判定（issue の受け入れ条件）:
- **昇格**は勝率 Wilson 95% CI 下限 > 0.5 のときのみ。
- **有意劣化**は点推定 < 0.5 かつ CI 上限 < 0.5（それ以外は劣化なし）。

## 結論

**候補は現 champion と統計的に同等（有意改善なし・有意劣化なし）。** cabt エンジンは
seed 非対応で mirror の分散が大きく、単一 run は 0.48〜0.53 を振れる（SOT-1694 と同じ
現象）。**独立 5 seed を pooled した N=12,496** で決着させた:

| seed | games/deck | 竹W-竹L | 勝率 | Wilson 95% CI |
| ---: | ---: | --- | ---: | --- |
| 101 | 100 | 1219-1280 | 0.4878 | [0.468, 0.507] |
| 202 | 100 | 1228-1271 | 0.4914 | [0.472, 0.511] |
| 303 | 100 | 1297-1203 | 0.5188 | [0.499, 0.538] |
| 404 | 100 | 1257-1243 | 0.5028 | [0.483, 0.522] |
| 505 | 100 | 1252-1246 | 0.5012 | [0.482, 0.521] |
| **pooled** | — | **6253-6243** | **0.5004** | **[0.4916, 0.5092]** |

- **昇格判定**: CI 下限 0.4916 < 0.5 → 昇格ゲート**不成立**（有意な改善ではない）。
- **有意劣化判定**: 点推定 0.5004 ≥ 0.5 → 劣化なし。
- **有意劣化デッキ**: pooled で 各デッキ N=500、点推定 < 0.5 かつ CI 上限 < 0.5 のデッキ
  = **0**（N=20/deck・N=50/deck の単一 run で一時的に赤くなったデッキ（marnie / hydrapple /
  mega_lucario）はいずれも N を増やすと消える小標本ノイズだった）。
- **fault / 違法手 = 0**（全 run 通算 15,000+ 試合）。思考時間 p95 ≈ 0.27ms/decision。

つまり試作ルール（to_active の即KO優先、deck-out時の非致死 swing）は 25 デッキ mirror では
**行動を変えるが総合勝率は中立**。SOT-1694 の教訓（この scoring アーキテクチャで、
既に高度に調整済みの champion に対し mirror の総合を有意に動かすのは難しい）を再確認した。
試作の狙いは対探索型（松）だったが、対松クロスハーネス（`eval/battle_vs_matsu.py`）は
松の MCTS が 1 手あたり重く、CI が分離するだけの N を回すコストが高い（issue でも
「可能なら…参考値」扱い）。よって対松での before/after 有意差は本 issue の範囲では未確定。
mirror が中立で対松の改善を確証できない以上、issue の昇格ゲート（および分類コメントの安全
デフォルト「有意改善が出なければ champion 維持」）に従い **champion を維持** した。

なお試作ルール自体は挙動として妥当（即KO促進はテンポ上正しい）で fault 0・非劣化のため、
今後 対松で有意な改善が確証できた場合に再提案できるよう、本ドキュメントに測定値を残す。

## 再現コマンド

```bash
# 試作ルールを別途 rule_based.py に再適用した上で（本 PR には含まれない）:
# 単一 seed
venv/bin/python eval/bench_25deck_rotation.py --games-per-deck 100 --old-ref main --seed 101 --json /tmp/g101.json
# 複数 seed を pooled するには各 json の wins/losses を合算し Wilson CI を取り直す
```
