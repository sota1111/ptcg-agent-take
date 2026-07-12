# 候補デッキ比較レポート（vs 現行デッキ, N=200）

SOT-1662 / 親 SOT-1640。候補デッキ群（SOT-1660）を、比較対戦スクリプト
`eval/compare_decks.py`（SOT-1661）で各候補 × 現行 `deck.csv` を **N=200 対戦（先手後手入替）**
実行した結果と、採用判断をまとめる。**現行 `deck.csv` の置き換えは行わない（人間判断に委ねる）。**

## 実行条件

| 項目 | 値 |
| --- | --- |
| スクリプト | `eval/compare_decks.py`（SOT-1661） |
| 対戦方式 | 同一 rule-based agent を両席に固定し、**デッキだけ**が差分（勝率＝デッキ強度） |
| ペアリング | side-swap（先手後手入替）で seat/先手バイアスを構造的に除去 |
| policy | `scoring`（既定） |
| N（候補ごと） | 200（decided=200, draws=0, undecided=0） |
| seed | `20260712`（agent RNG。エンジン自体は非決定的） |
| CI | Wilson 95%（z=1.96） |
| 現行デッキ | `deck.csv`（60枚, うち基本エネ×35） |
| 実行コマンド | `venv/bin/python eval/compare_decks.py --games 200 --seed 20260712 --json eval/traces/compare_decks_200/summary.json` |
| 生成物 | `eval/traces/compare_decks_200/summary.json` |
| 完走判定 | **GATE PASS** — 全候補で 200 対戦完走・異常終了 0 |

## 結果サマリ

勝率は「候補デッキ（agent A）の対現行デッキ勝率」。0.5 が互角。

| 候補デッキ | 勝率 | Wilson 95% CI | wins/losses | 対戦数 | 異常終了 | 現行比の判定 |
| --- | --- | --- | --- | --- | --- | --- |
| deck_balanced | 0.325 | [0.264, 0.393] | 65 / 135 | 200 | 0 | **有意に悪い**（CI上限 0.393 < 0.5） |
| deck_aggro | 0.260 | [0.204, 0.325] | 52 / 148 | 200 | 0 | **有意に悪い**（CI上限 0.325 < 0.5） |
| deck_lowenergy | 0.175 | [0.129, 0.234] | 35 / 165 | 200 | 0 | **有意に悪い**（CI上限 0.234 < 0.5） |

判定基準：Wilson 95% CI が 0.5 を跨がなければ現行比で有意。CI上限 < 0.5 → 有意に悪い、
CI下限 > 0.5 → 有意に良い、0.5 を含む → 判別不能。

## 判断

- **3候補すべてが現行 `deck.csv` より有意に弱い**（いずれも Wilson 95% CI の上限が 0.5 を下回る）。
  現行より有意に良い、または判別不能な候補は **無し**。
- 相対順位は deck_balanced（0.325）> deck_aggro（0.260）> deck_lowenergy（0.175）。基本エネ枚数が
  少ない候補ほど弱い傾向（現行 35枚 / balanced 16 / aggro 13 / lowenergy 10）で、現行デッキの
  エネルギー厚めの構成が、この rule-based agent とは相性が良いことを示唆する。

### 推奨（採用判断材料）

- **現行 `deck.csv` を維持することを推奨。** 今回の3候補に、現行を置き換える根拠となる勝率は無い。
- deck_lowenergy 方向（基本エネ削減）は明確に不利。今後デッキ改善を続けるなら、エネルギーを削らず
  進化/サポート枠を差し替える方向、または agent 側のエネルギー管理方策の見直しを検討する価値がある。
- 本Issueの範囲では `deck.csv` は変更しない。実際の置き換えは人間が判断する。

## 再現方法

```bash
cd /workspaces/ptcg-agent-take
scripts/setup_engine.sh   # cg/ を用意（未実行の場合）
venv/bin/python eval/compare_decks.py --games 200 --seed 20260712 \
    --json eval/traces/compare_decks_200/summary.json
```

生の集計は `eval/traces/compare_decks_200/summary.json` を参照。
