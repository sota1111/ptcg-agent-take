# 竹: シングルプライズ速攻デッキ（deck_tempo_metal）構築と A/B 評価レポート

SOT-1666 / 親 SOT-1665（D系列計画の D1）。竹 agent（scoring 方策）の短期決戦性向を活かす
「シングルプライズ速攻（テンポ）」デッキ `decks/deck_tempo_metal.csv` を構築し、現行 `deck.csv`
と repo 内 A/B 評価（`eval/compare_decks.py`, SOT-1661）で実測比較した。
**結論: 現行比で有意に弱い（改良再試行後も）→ 現行 `deck.csv` の維持を推奨（正直報告）。**
新デッキは D1 の成果物として、SOT-1668（竹梅クロス対戦評価）に供する。

## デッキリスト（60枚 / `deck_validator` 通過）

| 枚数 | カード | 役割 |
| --- | --- | --- |
| 33 | Basic {M} Energy (#8) | エネルギー（厚め: SOT-1662 の実測教訓） |
| 4 | Skarmory (#466) | 序盤アタッカー: Metal Claw {M}● 60 / Roost（回復） |
| 4 | Dialga (#696) | 中盤壁+打点: HP140, Beam {M} 30 / Chrono Burst {M}{M}● 80 |
| 4 | Duraludon (#992) | 大打点: HP130, Confront {M}{M} 50 / Duralubeam {M}{M}{M} 130 |
| 4 | Brave Bangle (#1175) | 非ルールボックス+30（対 ex） |
| 1 | Maximum Belt (#1158) | ACE SPEC, +50（対 ex） |
| 4 | Waitress (#1235) | エネ加速（山上6枚から付与） |
| 4 | Lillie's Determination (#1227) | ドロー（手札を戻して6ドロー） |
| 2 | Cheren (#1224) | ドロー（3枚） |

設計方針（Issue 指定に準拠）:

- 全ポケモンがルールボックス無し・単一プライズのたね、鋼モノタイプ。
- 全攻撃が**無条件ダメージ・低コスト（1〜3エネ）**（条件付き「不発」攻撃は不採用）。
- 鋼タイプは現行デッキの主軸 Snover / Mega Abomasnow ex の弱点（{M} ×2）を突く:
  Metal Claw 60×2=120 で Snover を進化前に一撃、Duralubeam は
  (130+30 Bangle)×2=320〜(130+50 Belt)×2=360 で Mega Abomasnow ex（HP350）圏内。
- エネルギー 33 枚（30±5 の上限側）。SOT-1662 の「エネ削減は全て有意に不利」の教訓どおり、
  予備評価でもエネ枚数が勝率の支配的変数だった（下記）。

## 評価条件

| 項目 | 値 |
| --- | --- |
| スクリプト | `eval/compare_decks.py`（SOT-1661; 同一 rule agent 両席固定・先後入替 paired） |
| policy | `scoring`（既定） |
| CI | Wilson 95%（z=1.96） |
| 独立 seed | `20260713`（N=200）と `924031`（N=400）の 2 系統 |
| 生成物 | `eval/traces/sot1666/*.json`（traces は license 制約で git 管理外） |

## 結果（本評価）

| 実行 | デッキ | N | seed | 勝率 | Wilson 95% CI | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 初回 | 初版（Duraludon×3+Togedemaru） | 200 | 20260713 | 0.255 | [0.200, 0.320] | 有意に弱い |
| 改良再試行（1回） | **最終版（上表 / Duraludon×4）** | 200 | 20260713 | 0.345 | [0.283, 0.413] | 有意に弱い |
| 採否判定 | 最終版 | 400 | 924031 | 0.352 | [0.307, 0.401] | 有意に弱い |

- 全 800 対戦とも decided=100%（draws 0 / undecided 0）・**異常終了 0（fault 0）・違法出力 0**。
- 2 独立 seed（20260713 / 924031）で最終版の勝率は 0.345 / 0.352 と整合。
  **CI 上限がいずれも 0.5 を下回り、「有意に弱くない」の受け入れ基準は未達。**

### 「不発」攻撃チェック（トレースサンプル N=20, LOGS レベル）

最終版の攻撃選択を全ログで確認: Beam（平均51: 弱点込み）/ Metal Claw（平均94）/
Chrono Burst（平均80）/ Confront（平均94）/ Duralubeam（平均313: 対 Mega 弱点×2+Bangle）。
**ダメージ0の「不発」攻撃を選び続ける挙動は無し**（Roost のダメージ0は回復ワザ仕様。
末尾2件の0はゲーム終了によるトレース境界）。

## 設計過程（予備評価: N=100, 複数 seed）

12 案を予備評価した主な結果（勝率 vs 現行）:

| 案 | 勝率（seed別） | 所見 |
| --- | --- | --- |
| 鋼 33エネ（Skarmory/Dialga/Duraludon 系） | 0.39 / 0.34 / 0.35 | **最良**。採用 |
| 鋼 35〜37エネ | 0.22〜0.35 | 33エネと有意差なし〜やや劣る |
| 鋼 30エネ | 0.18 | エネ薄は不利（SOT-1662 と一致） |
| Iron Crown / Genesect（3エネ100打点軸） | 0.12〜0.26 | 3エネ攻撃はこの agent には遅すぎる |
| 無色大打点（Tauros/Hop's Snorlax/Terapagos） | 0.07〜0.23 | 弱点を突けず不利 |
| 闘（Okidogi+Premium Power Pro） | 0.06 | 最弱。弱点なし+HP 不足 |

## 敗因分析（なぜシングルプライズ速攻は現行に勝てないか）

トレース集計（20 対戦）で、初期案では **Mega Abomasnow ex（HP350）が一度も KO されず**、
候補側の獲得プライズはほぼ Snover / Kyogre のみだった。最終版は Duralubeam+弱点で
Mega を圏内に捉えるが、(1) 3 エネ到達前に Kyogre 130 / Mega 200〜400 打点で処理される、
(2) 単プライズ側は 6 KO 必要 vs 相手は Mega 1 体で盤面を掃討、の 2 点で
プライズレースが構造的に不利。scoring agent は退避・温存をしないため、
「育てて 2 回殴る」プランの再現性が低い。

## 判断

- **現行 `deck.csv` の維持を推奨**（本 Issue の範囲では `deck.csv` は変更しない）。
  シングルプライズ速攻は 1 回の改良再試行（0.255→0.345）でも有意に弱い。
- `decks/deck_tempo_metal.csv` は D 系列 D1 の成果物として repo に記録し、
  SOT-1668 の竹梅クロス対戦評価に供する（梅の新デッキ D2 とは Pokémon/どうぐ構成が
  完全に異なる見込みで、共有され得る汎用カードは Waitress/Lillie/Belt の最大 9 枚 ≤ 15）。

## 再現方法

```bash
cd /workspaces/ptcg-agent-take
scripts/setup_engine.sh   # cg/ を用意（未実行の場合）
venv/bin/python eval/deck_validator.py decks/deck_tempo_metal.csv
mkdir -p /tmp/d && cp decks/deck_tempo_metal.csv /tmp/d/
venv/bin/python eval/compare_decks.py --games 400 --seed 924031 --decks-dir /tmp/d \
    --json eval/traces/sot1666/final_n400_seed924031.json
```
