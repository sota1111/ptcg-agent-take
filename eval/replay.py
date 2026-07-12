"""Human-readable game-record rendering + record-based replay (SOT-1621).

Turns one match trace (the JSONL produced by ``eval/record_match.py`` /
``eval/arena.py``, SOT-1618 / SOT-1619) into a **human-readable game record** so a
person can answer "why did this side win/lose" and "was each individual decision
reasonable" at the single-match level.

The cabt engine takes no seed (E1): re-running a match does not reproduce it, so a
recorded trace is the *only* faithful replay. This module is that record-based
replay — it renders the recorded trace, it does not re-run the engine.

What it does:
  * **id → name resolution** — card ids and attack ids are resolved to names via
    the engine masters ``all_card_data()`` / ``all_attack()`` and, as a fallback /
    Japanese source, ``data/EN_Card_Data.csv`` / ``data/JP_Card_Data.csv``. An id
    with no known name falls back to ``#<id>`` (never raises).
  * **turn-ordered text record** — each decision is rendered as the acting side's
    legal-move list and its chosen move; each event log (ATTACK / HP_CHANGE /
    knockout / special conditions / COIN(head) … the 23 LogType kinds) is rendered
    as a sentence.
  * **decisive-scene extraction** — filters that pull out knockout decisions, large
    HP swings, and the last N decisions before the RESULT.
  * **viewpoint (E4)** — every decision is from the acting (turn) side's viewpoint;
    the opponent's hand and other hidden zones are shown as non-public.

Usage:
    venv/bin/python eval/replay.py eval/traces/match.jsonl              # full record
    venv/bin/python eval/replay.py <trace.jsonl> --lang ja             # Japanese names
    venv/bin/python eval/replay.py <trace.jsonl> --scenes              # decisive scenes only
    venv/bin/python eval/replay.py <trace.jsonl> --scenes --hp-threshold 60
    venv/bin/python eval/replay.py <trace.jsonl> --no-engine           # CSV-only names

Run from the repo root (after scripts/setup_engine.sh has populated cg/ + data/).
The name-resolution and rendering functions are pure over already-parsed records,
so they are unit-tested without the engine (see eval/test_replay.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# Enum labels (mirrors cg/api.py; kept local so rendering needs no engine import)
# --------------------------------------------------------------------------- #

AREA_NAMES = {
    1: "山札", 2: "手札", 3: "トラッシュ", 4: "バトル場", 5: "ベンチ",
    6: "サイド", 7: "スタジアム", 8: "エネルギー", 9: "どうぐ",
    10: "進化前", 11: "プレイヤー", 12: "見ているカード",
}

ENERGY_NAMES = {
    0: "無", 1: "草", 2: "炎", 3: "水", 4: "雷", 5: "超", 6: "闘",
    7: "悪", 8: "鋼", 9: "竜", 10: "虹", 11: "ロケット団",
}

SPECIAL_CONDITION_NAMES = {0: "どく", 1: "やけど", 2: "ねむり", 3: "マヒ", 4: "こんらん"}

# OptionType (cg/api.py OptionType) → short human label for legal-move rendering.
OPTION_TYPE_NAMES = {
    0: "数値", 1: "はい", 2: "いいえ", 3: "カード", 4: "どうぐ", 5: "エネルギーカード",
    6: "エネルギー", 7: "手札からプレイ", 8: "つける", 9: "進化", 10: "特性",
    11: "トラッシュ", 12: "にげる", 13: "ワザ", 14: "ターン終了", 15: "スキル順",
    16: "特殊状態",
}

# SelectContext (cg/api.py SelectContext) → short human label ("what is being chosen").
SELECT_CONTEXT_NAMES = {
    0: "メイン", 1: "最初のバトルポケモン", 2: "最初のベンチ", 3: "入れ替え",
    4: "バトル場へ", 5: "ベンチへ", 6: "場に出す", 7: "手札へ", 8: "トラッシュ",
    9: "山札へ", 10: "山札の下へ", 11: "サイドへ", 12: "そのまま残す",
    13: "ダメカン", 14: "ダメカン(任意)", 15: "ダメージを与える", 16: "ダメカン除去",
    17: "回復", 18: "進化元", 19: "進化先", 20: "退化", 21: "つける先",
    22: "つけるカード", 23: "外す先", 24: "見る", 25: "効果対象",
    26: "エネルギートラッシュ", 27: "どうぐトラッシュ", 28: "エネルギー入替",
    29: "トラッシュ", 30: "エネルギートラッシュ", 31: "エネルギーを手札へ",
    32: "エネルギーを山札へ", 33: "エネルギー入替", 34: "スキル順", 35: "ワザ",
    36: "ワザ無効", 37: "進化", 38: "ドロー枚数", 39: "ダメカン数",
    40: "ダメカン除去数", 41: "先攻を選ぶ", 42: "マリガン", 43: "効果を使う",
    44: "最初の効果", 45: "さらに退化", 46: "コインの表を選ぶ",
    47: "特殊状態を与える", 48: "特殊状態を回復",
}

# Match-result reason codes (LogType.RESULT.reason).
REASON_NAMES = {
    1: "サイド取り切り", 2: "山札切れ", 3: "バトル場不在", 4: "カード効果",
}


# --------------------------------------------------------------------------- #
# id → name resolution
# --------------------------------------------------------------------------- #

class NameResolver:
    """Resolves card ids and attack ids to human-readable names.

    Sources, in order of preference for a given language:
      * ``lang="en"`` (default): engine ``all_card_data()`` → EN CSV → ``#<id>``.
      * ``lang="ja"``: JP CSV → engine (EN) → EN CSV → ``#<id>``.
    Attack names come from the engine ``all_attack()`` only (the CSVs are not keyed
    by attack id); an unknown attack id falls back to ``#<id>``.

    Every source is loaded best-effort: a missing engine (cg/ absent) or missing
    CSV simply drops that source. Resolution never raises, so ``--no-engine`` and
    engine-less unit tests work, and unknown ids always fall back to ``#<id>`` — the
    acceptance-required behaviour.
    """

    def __init__(
        self,
        lang: str = "en",
        *,
        use_engine: bool = True,
        data_dir: Optional[str] = None,
        # Optional pre-built maps (used by tests to avoid the engine/CSVs entirely).
        engine_cards: Optional[dict] = None,
        engine_attacks: Optional[dict] = None,
    ):
        self.lang = lang
        self.data_dir = data_dir if data_dir is not None else os.path.join(REPO, "data")
        self._engine_cards = dict(engine_cards) if engine_cards is not None else {}
        self._engine_attacks = dict(engine_attacks) if engine_attacks is not None else {}
        self._csv_en: dict[int, str] = {}
        self._csv_ja: dict[int, str] = {}

        if use_engine and engine_cards is None and engine_attacks is None:
            self._load_engine()
        self._load_csvs()

    # -- source loading (all best-effort) ----------------------------------- #

    def _load_engine(self) -> None:
        try:
            sys.path.insert(0, REPO)
            from cg.api import all_card_data, all_attack  # type: ignore
            self._engine_cards = {c.cardId: c.name for c in all_card_data()}
            self._engine_attacks = {a.attackId: a.name for a in all_attack()}
        except Exception:
            # Engine not available — CSV fallback still gives card names.
            self._engine_cards = self._engine_cards or {}
            self._engine_attacks = self._engine_attacks or {}

    def _load_one_csv(self, filename: str, target: dict) -> None:
        path = os.path.join(self.data_dir, filename)
        try:
            with open(path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                id_col = reader.fieldnames[0] if reader.fieldnames else None
                name_col = reader.fieldnames[1] if reader.fieldnames and len(reader.fieldnames) > 1 else None
                if not id_col or not name_col:
                    return
                for row in reader:
                    raw = (row.get(id_col) or "").strip()
                    if not raw:
                        continue
                    try:
                        cid = int(raw)
                    except ValueError:
                        continue
                    # The CSV has one row per move, so keep the first (card name is
                    # identical across a card's rows).
                    if cid not in target:
                        target[cid] = (row.get(name_col) or "").strip()
        except OSError:
            return

    def _load_csvs(self) -> None:
        self._load_one_csv("EN_Card_Data.csv", self._csv_en)
        self._load_one_csv("JP_Card_Data.csv", self._csv_ja)

    # -- resolution --------------------------------------------------------- #

    def card(self, card_id: Optional[int]) -> str:
        """Resolve a card id to a name, or ``#<id>`` (``?`` if id is None)."""
        if card_id is None:
            return "?"
        if self.lang == "ja":
            order = (self._csv_ja, self._engine_cards, self._csv_en)
        else:
            order = (self._engine_cards, self._csv_en, self._csv_ja)
        for src in order:
            name = src.get(card_id)
            if name:
                return name
        return f"#{card_id}"

    def attack(self, attack_id: Optional[int]) -> str:
        """Resolve an attack id to a name, or ``#<id>`` (``?`` if id is None)."""
        if attack_id is None:
            return "?"
        name = self._engine_attacks.get(attack_id)
        return name if name else f"#{attack_id}"

    def card_with_id(self, card_id: Optional[int]) -> str:
        """``Name(#id)`` — name plus the raw id for traceability."""
        if card_id is None:
            return "?"
        return f"{self.card(card_id)}(#{card_id})"


# --------------------------------------------------------------------------- #
# Trace parsing
# --------------------------------------------------------------------------- #

def load_trace(path: str) -> dict:
    """Parse a trace JSONL file into ``{meta, decisions, result}``.

    Raises ``FileNotFoundError`` / ``ValueError`` on an unreadable or empty trace.
    """
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return split_records(records)


def split_records(records: list[dict]) -> dict:
    """Split already-parsed trace records into meta / decisions / result."""
    meta = None
    decisions: list[dict] = []
    result = None
    for rec in records:
        kind = rec.get("kind")
        if kind == "meta":
            meta = rec
        elif kind == "decision":
            decisions.append(rec)
        elif kind == "result":
            result = rec
    if meta is None and not decisions and result is None:
        raise ValueError("trace contains no meta/decision/result records")
    return {"meta": meta, "decisions": decisions, "result": result}


# --------------------------------------------------------------------------- #
# Option (legal-move) rendering
# --------------------------------------------------------------------------- #

def render_option(opt: dict, resolver: NameResolver) -> str:
    """Render one legal-move ``Option`` as a short human phrase."""
    if not isinstance(opt, dict):
        return str(opt)
    otype = opt.get("type")
    label = OPTION_TYPE_NAMES.get(otype, f"type{otype}")

    parts: list[str] = [label]
    # Card-bearing options: name the card / attack where the id is present.
    if opt.get("cardId") is not None:
        parts.append(resolver.card_with_id(opt["cardId"]))
    if opt.get("attackId") is not None:
        parts.append(f"ワザ={resolver.attack(opt['attackId'])}")
    if opt.get("area") is not None:
        parts.append(f"[{AREA_NAMES.get(opt['area'], opt['area'])}]")
    if opt.get("inPlayArea") is not None:
        parts.append(f"→[{AREA_NAMES.get(opt['inPlayArea'], opt['inPlayArea'])}]")
    if opt.get("number") is not None:
        parts.append(f"={opt['number']}")
    if opt.get("count") is not None:
        parts.append(f"x{opt['count']}")
    if opt.get("specialConditionType") is not None:
        parts.append(SPECIAL_CONDITION_NAMES.get(opt["specialConditionType"], "?"))
    return " ".join(str(p) for p in parts)


def _choice_indices(choice: Any) -> list[int]:
    """Normalize a decision ``choice`` (list of option indices) to a list of ints."""
    if isinstance(choice, list):
        return [c for c in choice if isinstance(c, int)]
    if isinstance(choice, int):
        return [choice]
    return []


# --------------------------------------------------------------------------- #
# Event-log rendering (the 23 LogType kinds)
# --------------------------------------------------------------------------- #

def _who(log: dict) -> str:
    p = log.get("playerIndex")
    return f"P{p}" if p in (0, 1) else "P?"


def render_log(log: dict, resolver: NameResolver) -> str:
    """Render one event log (a single LogType record) as a sentence.

    Covers all 23 LogType kinds; an unknown type degrades to a compact dump so the
    render never drops information.
    """
    t = log.get("type")
    who = _who(log)
    c = lambda: resolver.card_with_id(log.get("cardId"))
    tgt = lambda: resolver.card_with_id(log.get("cardIdTarget"))

    if t == 0:
        return f"{who} 山札をシャッフル"
    if t == 1:
        has = log.get("hasBasicPokemon")
        return f"{who} 手札のたねポケモン: {'あり' if has else 'なし(マリガン)'}"
    if t == 2:
        return f"{who} ターン開始"
    if t == 3:
        return f"{who} ターン終了"
    if t == 4:
        return f"{who} ドロー: {c()}"
    if t == 5:
        return f"{who} 相手がドロー(非公開)"
    if t == 6:
        fr = AREA_NAMES.get(log.get("fromArea"), log.get("fromArea"))
        to = AREA_NAMES.get(log.get("toArea"), log.get("toArea"))
        note = "  ★きぜつ" if log.get("fromArea") == 4 and log.get("toArea") == 3 else ""
        return f"{who} カード移動: {c()} [{fr}]→[{to}]{note}"
    if t == 7:
        fr = AREA_NAMES.get(log.get("fromArea"), log.get("fromArea"))
        to = AREA_NAMES.get(log.get("toArea"), log.get("toArea"))
        return f"{who} 非公開カード移動: [{fr}]→[{to}]"
    if t == 8:
        active = resolver.card_with_id(log.get("cardIdActive"))
        bench = resolver.card_with_id(log.get("cardIdBench"))
        return f"{who} ポケモン入れ替え: バトル場={bench} ベンチ={active}"
    if t == 9:
        before = resolver.card_with_id(log.get("cardIdBefore"))
        after = resolver.card_with_id(log.get("cardIdAfter"))
        return f"{who} ポケモン変更: {before}→{after}"
    if t == 10:
        return f"{who} 手札からプレイ: {c()}"
    if t == 11:
        return f"{who} つけた: {c()} → {tgt()}"
    if t == 12:
        return f"{who} 進化: {tgt()} → {c()}"
    if t == 13:
        return f"{who} 退化: {c()} ← {tgt()}"
    if t == 14:
        before = resolver.card_with_id(log.get("cardIdBefore"))
        after = resolver.card_with_id(log.get("cardIdAfter"))
        return f"{who} つけ替え: {c()} {before}→{after}"
    if t == 15:
        return f"{who} ワザ: {c()} の「{resolver.attack(log.get('attackId'))}」"
    if t == 16:
        value = log.get("value", 0) or 0
        kind = "ダメカン配置" if log.get("putDamageCounter") else "HP変化"
        return f"{who} {kind}: {c()} value={value}"
    if t in (17, 18, 19, 20, 21):
        cond = {17: "どく", 18: "やけど", 19: "ねむり", 20: "マヒ", 21: "こんらん"}[t]
        state = "回復" if log.get("isRecover") else "付与"
        return f"{who} 特殊状態[{cond}]{state}: {c()}"
    if t == 22:
        return f"{who} コイン: {'表(head)' if log.get('head') else '裏(tail)'}"
    if t == 23:
        result = log.get("result")
        reason = log.get("reason")
        if result == 2:
            outcome = "引き分け"
        elif result in (0, 1):
            outcome = f"P{result} の勝ち"
        else:
            outcome = "未決着"
        return f"◆結果: {outcome}（理由: {REASON_NAMES.get(reason, reason)}）"
    # Unknown / future LogType — dump the raw record rather than drop it.
    return f"{who} log(type={t}): {json.dumps(log, ensure_ascii=False)}"


# --------------------------------------------------------------------------- #
# Decisive-scene extraction
# --------------------------------------------------------------------------- #

def decision_is_knockout(decision: dict) -> bool:
    """True if a knockout occurred in the logs seen at this decision.

    A knockout is the Active Pokémon leaving the field to the discard pile — a
    ``MOVE_CARD`` (LogType 6) with ``fromArea==ACTIVE(4)`` and ``toArea==DISCARD(3)``.
    """
    for log in decision.get("logs", []) or []:
        if log.get("type") == 6 and log.get("fromArea") == 4 and log.get("toArea") == 3:
            return True
    return False


def decision_max_hp_change(decision: dict) -> int:
    """The largest absolute HP change (LogType 16) among this decision's logs."""
    best = 0
    for log in decision.get("logs", []) or []:
        if log.get("type") == 16:
            best = max(best, abs(log.get("value", 0) or 0))
    return best


def extract_scenes(
    trace: dict,
    *,
    hp_threshold: int = 30,
    endgame: int = 3,
    knockouts: bool = True,
    big_hp: bool = True,
    include_endgame: bool = True,
) -> list[dict]:
    """Return decisive-scene decisions with the reason(s) each was selected.

    Filters:
      * ``knockouts`` — decisions whose logs contain a knockout (Active → discard);
      * ``big_hp`` — decisions with an HP change of magnitude ≥ ``hp_threshold``;
      * ``include_endgame`` — the last ``endgame`` decisions before the RESULT.

    Each returned item is ``{"decision": <record>, "reasons": [<str>, …]}`` in
    decision order, de-duplicated so a decision matching several filters appears once.
    """
    decisions = trace.get("decisions", [])
    n = len(decisions)
    reasons_by_index: dict[int, list[str]] = {}

    def add(i: int, reason: str) -> None:
        reasons_by_index.setdefault(i, [])
        if reason not in reasons_by_index[i]:
            reasons_by_index[i].append(reason)

    for i, dec in enumerate(decisions):
        if knockouts and decision_is_knockout(dec):
            add(i, "きぜつ発生")
        if big_hp:
            mag = decision_max_hp_change(dec)
            if mag >= hp_threshold:
                add(i, f"大きなHP変化({mag})")
    if include_endgame and n:
        for i in range(max(0, n - endgame), n):
            add(i, "終盤(RESULT直前)")

    return [
        {"decision": decisions[i], "reasons": reasons_by_index[i]}
        for i in sorted(reasons_by_index)
    ]


# --------------------------------------------------------------------------- #
# Full-record / scene text rendering
# --------------------------------------------------------------------------- #

def render_decision(decision: dict, resolver: NameResolver, *, reasons: Optional[list[str]] = None) -> str:
    """Render one decision: viewpoint (E4), event logs, legal moves + the choice."""
    lines: list[str] = []
    idx = decision.get("index")
    turn = decision.get("turn")
    actor = decision.get("your_index")
    if actor not in (0, 1):
        actor = decision.get("select_player")
    header = f"[決定#{idx} turn={turn} 手番=P{actor}視点(E4)]"
    if reasons:
        header += "  << " + " / ".join(reasons) + " >>"
    lines.append(header)

    # Event logs that preceded this decision.
    for log in decision.get("logs", []) or []:
        lines.append(f"    ・{render_log(log, resolver)}")

    select = decision.get("select") or {}
    context = select.get("context")
    ctx_label = SELECT_CONTEXT_NAMES.get(context, context)
    options = select.get("option") or []
    chosen = set(_choice_indices(decision.get("choice")))
    mn, mx = select.get("minCount"), select.get("maxCount")
    lines.append(f"    選択: {ctx_label}（{len(options)}択, 選ぶ数 {mn}〜{mx}）— 相手の手札等は非公開")
    for i, opt in enumerate(options):
        mark = "✓" if i in chosen else " "
        lines.append(f"      {mark} [{i}] {render_option(opt, resolver)}")
    return "\n".join(lines)


def render_meta(meta: Optional[dict], resolver: NameResolver) -> str:
    """Render the meta header (agents, first player, engine stamp)."""
    if not meta:
        return "(meta なし)"
    lines = ["=" * 70, "PTCG 棋譜（記録ベースリプレイ / SOT-1621）", "=" * 70]
    lines.append(f"trace_id : {meta.get('trace_id')}")
    lines.append(f"created  : {meta.get('created_at')}")
    agents = meta.get("agents") or []
    for a in agents:
        lines.append(f"  P{a.get('index')} = {a.get('name')} v{a.get('version')} {a.get('params', {})}")
    fp = meta.get("first_player")
    lines.append(f"先攻     : P{fp}" if fp in (0, 1) else "先攻     : 未確定")
    eng = meta.get("engine") or {}
    lines.append(f"engine   : {eng.get('path')} sha256={(eng.get('sha256') or '')[:12]}…")
    if meta.get("start_error"):
        lines.append(f"start_error: {meta['start_error']}")
    return "\n".join(lines)


def render_result(result: Optional[dict], resolver: NameResolver) -> str:
    """Render the terminal result line."""
    if not result:
        return "(result なし)"
    winner = result.get("winner")
    if winner in (0, 1):
        outcome = f"P{winner} の勝ち"
    elif result.get("result") == 2:
        outcome = "引き分け"
    elif result.get("truncated"):
        outcome = "打ち切り(未決着)"
    else:
        outcome = "未決着"
    lines = ["-" * 70]
    reason = result.get("reason")
    lines.append(f"◆ 結果: {outcome}  理由={REASON_NAMES.get(reason, reason)}  "
                 f"最終turn={result.get('final_turn')} 決定数={result.get('total_decisions')}")
    if result.get("failure"):
        f = result["failure"]
        lines.append(f"  ⚠ 異常終了: category={f.get('category')} player=P{f.get('player')} {f.get('error')}")
    return "\n".join(lines)


def render_record(trace: dict, resolver: NameResolver) -> str:
    """Render the full turn-ordered human-readable game record."""
    parts = [render_meta(trace.get("meta"), resolver)]
    for dec in trace.get("decisions", []):
        parts.append(render_decision(dec, resolver))
    parts.append(render_result(trace.get("result"), resolver))
    return "\n".join(parts)


def render_scenes(trace: dict, resolver: NameResolver, **kwargs: Any) -> str:
    """Render only the decisive scenes extracted by ``extract_scenes``."""
    scenes = extract_scenes(trace, **kwargs)
    parts = [render_meta(trace.get("meta"), resolver)]
    parts.append(f"\n決定的場面 {len(scenes)} 件を抽出:\n")
    if not scenes:
        parts.append("  (該当する場面なし)")
    for scene in scenes:
        parts.append(render_decision(scene["decision"], resolver, reasons=scene["reasons"]))
        parts.append("")
    parts.append(render_result(trace.get("result"), resolver))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a PTCG match trace as a human-readable game record (SOT-1621)."
    )
    p.add_argument("trace", help="path to a trace .jsonl file (from record_match.py / arena.py)")
    p.add_argument("--lang", choices=["en", "ja"], default="ja", help="card/attack name language (default ja)")
    p.add_argument("--scenes", action="store_true", help="render only decisive scenes")
    p.add_argument("--hp-threshold", type=int, default=30, help="min |HP change| for a big-HP scene (default 30)")
    p.add_argument("--endgame", type=int, default=3, help="how many decisions before RESULT to include (default 3)")
    p.add_argument("--no-engine", action="store_true", help="skip the engine master; resolve names from CSVs only")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not os.path.isfile(args.trace):
        print(f"error: no such trace file: {args.trace}", file=sys.stderr)
        return 2
    try:
        trace = load_trace(args.trace)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: could not parse trace: {exc}", file=sys.stderr)
        return 1

    resolver = NameResolver(lang=args.lang, use_engine=not args.no_engine)
    if args.scenes:
        print(render_scenes(trace, resolver, hp_threshold=args.hp_threshold, endgame=args.endgame))
    else:
        print(render_record(trace, resolver))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
