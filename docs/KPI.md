# 竹(take) 対戦KPI定義と履歴記録 (SOT-1709)

竹エージェントの対戦性能向上を継続観測するためのKPI定義・測定方法・履歴運用。
1計測 = `eval/kpi_history.jsonl`(コミット対象)への1行追記。スキーマ `take-kpi-v1`。

## KPI一覧

| KPI | 定義 | 改善方向 | 根拠 |
| --- | --- | --- | --- |
| `mirror_winrate_vs_baseline` | 25デッキ side-swap ミラー対戦での現行エージェントの対固定ベースライン勝率 (Wilson 95% CI付き、decided戦のみ) | 高いほど良い | エージェント強さの主指標。ベースライン固定で世代間比較可能 |
| `prize_out_loss_rate` | 竹側の正常決着敗戦のうち `prize_out`(サイド取り切られ)が原因の割合 | 低いほど良い | 竹の支配的敗因 (SOT-1698 で83.8%)。プライズレース負けの解消度 |
| `fallback_decision_rate` | 竹側の全決定のうち `CONTEXT_HANDLERS` 未登録 context での random-fallback 決定の割合 | 低いほど良い | 戦術ロジックの被覆率 (SOT-1682/1694 で最効率だった穴検出) |
| `fault_total` | 全fault数 (agent例外・違法手= engine_error・timeout・worker_error、竹/ベースライン両側) | **常に0** (非0は即NG) | 提出安全性ゲート |
| `decision_time_mean_ms` | 竹側1決定あたり平均思考時間 (ms、max併記) | 低いほど良い (参考値) | 本番時間制限の監視 |

## 測定方法

### 通常計測 (全KPIが埋まる)

```bash
venv/bin/python eval/kpi.py --measure --games-per-deck 2 --issue SOT-XXXX
```

- `decks/initial/` の25デッキそれぞれで、現行 working-tree エージェント(A) vs
  固定ベースライン(B) の side-swap ミラーアリーナ (`eval/arena.py`) を実行。
- トレースは LOGS レベルで記録し、`select_player` により竹側の決定のみを
  集計 (fallback穴・思考時間)。竹の敗因はエンジン終端 `reason` で分類
  (fault起因の敗戦は `abnormal` として prize_out 率の分母から除外)。
- トレース出力は gitignore 済みの `eval/traces/kpi_<ts>/` (スクラッチ)。
- 履歴への追記を止めて記録内容だけ見るには `--no-append`。

### 既存25デッキ回転ベンチからの変換 (勝率/fault/時間のみ)

```bash
venv/bin/python eval/bench_25deck_rotation.py --games-per-deck 20 --kpi SOT-XXXX
# または既存レポートJSONから:
venv/bin/python eval/kpi.py --from-report report.json --issue SOT-XXXX
```

`bench_25deck_rotation.py` は決定の座席帰属と敗因を捨てるため、
`prize_out_loss_rate` / `fallback_decision_rate` は null になる。

## 履歴レコード仕様 (`take-kpi-v1`)

1行 = 1計測のJSON。主フィールド:
`ts` (UTC) / `git_sha` (計測時のHEAD) / `issue` (Linear ID) / `source`
(`kpi-measure` | `bench_25deck_rotation`) / `baseline_ref`・`baseline_sha` /
`deck_pool`・`n_decks`・`n_matches`・`games_per_deck`・`seed` /
`kpis.<name>.value` (+ KPI別の内訳: CI・敗因分布・fault内訳など)。

## トレンド確認

```bash
venv/bin/python eval/kpi_report.py
```

時系列テーブルと直近2計測の比較を表示。各KPIは `kpi.KPI_DIRECTIONS` の
改善方向で 改善/悪化/横ばい (閾値 `kpi_report.FLAT_EPS`) を判定。
`fault_total` はトレンドではなく 0 維持ゲート (非0 = NG)。

## ベースライン方針

- ベースラインは **git SHA固定** (`kpi.BASELINE_REF` =
  `b51da4f` — SOT-1682→SOT-1694 champion系譜、SOT-1700マージ時点)。
  `main` 追従にしないのは、KPIトレンドを「一定の相手に対する現行エージェントの
  強さ」として世代間比較可能に保つため。
- エンジンはシード注入不可 (E1) のため対戦結果は非再現。N と Wilson CI を
  必ず併記し、CI が分離した変化のみを有意とみなす。
- ベースラインを更新する場合 (現行が CI 分離で十分上回り飽和した場合のみ):
  `BASELINE_REF` と本表を同一コミットで更新し、履歴上の断絶を
  `kpi_history.jsonl` 直後レコードの issue で明示する。
