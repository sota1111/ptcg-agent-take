"""竹 vs 松 loss-cause harness (SOT-1698).

Plays 竹 (this repo's Kaggle submission ``main.agent``) against 松
(``../ptcg-agent-matsu`` — the MCTS champion, SOT-1693) on the shared cabt
engine **hosted from this repo** and classifies **why 竹 loses**, using the
engine's terminal RESULT log ``reason`` (1=prize_out / 2=deck_out / 3=no_active /
4=card_effect). This is the "対松敗戦トレース分析" the issue asks for first: it
confirms whether 竹's loss profile against a search agent matches the 25-deck
*mirror* profile measured in SOT-1694 (prize 71% / deck_out 27%).

Both contestants run as isolated subprocesses (``eval/agent_server.py`` in their
own repo/venv) because the two ``agents`` packages have colliding module names.
This host process owns only the engine (this repo's ``cg.game``) and reads the
``reason`` off the terminal observation — so no per-agent trace instrumentation
is needed and the classification is agent-implementation agnostic.

Fairness (先後入替): every pairing is played in seat-alternating pairs. In mirror
mode both contestants pilot the same randomly-drawn deck per 先後 pair, so the
swap cancels deck strength and the result isolates piloting skill.

The engine has **no seed API**, so results are statistical (Wilson 95% CI). Run
enough matches (or aggregate shards) for the CIs to separate.

Usage (from this repo root; ``../ptcg-agent-matsu`` must exist as a sibling)::

    venv/bin/python eval/battle_vs_matsu.py --n 60 --decks-dir decks/initial \
        --seed 1698 --json /tmp/vs_matsu.json --md docs/vs_matsu_sot1698.md
    venv/bin/python eval/battle_vs_matsu.py --aggregate shard1.json shard2.json \
        --md docs/vs_matsu_sot1698.md            # merge shards, no matches played
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLINGS = os.path.dirname(REPO)

# Loss-reason codes from the engine RESULT log (cg/api.py; mirrors trace.py).
REASONS = {1: "prize_out", 2: "deck_out", 3: "no_active", 4: "card_effect"}
RESULT_LOG_TYPE = 23  # LogType.RESULT

DECK_SIZE = 60
MAX_DECISIONS = 100_000

# take is hosted from THIS repo; matsu is the sibling MCTS champion.
TAKE = ("take", "竹", REPO)
MATSU = ("matsu", "松", os.path.join(SIBLINGS, "ptcg-agent-matsu"))


# --------------------------------------------------------------------------- #
# Pure helpers (stdlib only — unit-testable without the engine or matsu)
# --------------------------------------------------------------------------- #
def load_deck(path: str) -> list[int]:
    with open(path, encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()][:DECK_SIZE]


def discover_decks(decks_dir: str) -> list[str]:
    files = [
        p for p in glob.glob(os.path.join(decks_dir, "*.csv"))
        if re.match(r"^\d+_", os.path.basename(p))
    ]
    if not files:
        raise SystemExit(f"no NN_*.csv decks found in {decks_dir}")
    files.sort(key=lambda p: int(re.match(r"^(\d+)_", os.path.basename(p)).group(1)))
    return files


def build_deck_schedule(n: int, deck_files: list[str],
                        rng: random.Random) -> list[str]:
    """One deck per match; each 先後 pair (2k / 2k+1) reuses one deck (mirror)."""
    sched: list[str] = []
    i = 0
    while i < n:
        d = rng.choice(deck_files)
        sched.append(d)
        if i + 1 < n:
            sched.append(d)
        i += 2
    return sched


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for ``wins/n`` (clamped to [0, 1])."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def extract_reason(obs: dict) -> Optional[int]:
    """Return the engine RESULT-log ``reason`` (1-4) from a terminal observation."""
    for log in (obs.get("logs") or []):
        if isinstance(log, dict) and log.get("type") == RESULT_LOG_TYPE:
            return log.get("reason")
    return None


def make_sandbox(repo: str, root: Optional[str] = None) -> str:
    """Per-contestant sandbox cwd: symlink the repo, copy deck.csv (writable).

    ``__set_deck__`` rewrites ``<cwd>/deck.csv``; copying it (never a symlink)
    guarantees the rewrite can never clobber the repo's committed deck.csv.
    """
    sb = tempfile.mkdtemp(prefix="sot1698_sb_", dir=root)
    for name in sorted(os.listdir(repo)):
        if name in ("deck.csv", ".git"):
            continue
        os.symlink(os.path.join(repo, name), os.path.join(sb, name))
    src_deck = os.path.join(repo, "deck.csv")
    dst_deck = os.path.join(sb, "deck.csv")
    if os.path.isfile(src_deck):
        shutil.copyfile(src_deck, dst_deck)
    if os.path.islink(dst_deck):
        raise RuntimeError(f"sandbox deck.csv is a symlink: {dst_deck}")
    return sb


@dataclass
class LossTally:
    """竹-centric tally of one deck's (or the whole run's) matches vs 松."""

    take_wins: int = 0
    matsu_wins: int = 0
    draws: int = 0
    unfinished: int = 0
    take_faults: int = 0
    matsu_faults: int = 0
    # 竹's loss cause histogram (reason name -> count), over decided 竹 losses.
    take_losses: dict = field(default_factory=dict)
    # 松's loss cause histogram, over decided 竹 wins (for symmetry / sanity).
    matsu_losses: dict = field(default_factory=dict)

    def record(self, *, take_won: Optional[bool], reason: Optional[int],
               fault_by: Optional[str]) -> None:
        if fault_by == "take":
            self.take_faults += 1
        elif fault_by == "matsu":
            self.matsu_faults += 1
        if take_won is None:
            if reason == -2:
                self.draws += 1
            else:
                self.unfinished += 1
            return
        name = REASONS.get(reason, "other")
        if take_won:
            self.take_wins += 1
            self.matsu_losses[name] = self.matsu_losses.get(name, 0) + 1
        else:
            self.matsu_wins += 1
            self.take_losses[name] = self.take_losses.get(name, 0) + 1

    @property
    def decided(self) -> int:
        return self.take_wins + self.matsu_wins

    def merge(self, other: "LossTally") -> None:
        self.take_wins += other.take_wins
        self.matsu_wins += other.matsu_wins
        self.draws += other.draws
        self.unfinished += other.unfinished
        self.take_faults += other.take_faults
        self.matsu_faults += other.matsu_faults
        for k, v in other.take_losses.items():
            self.take_losses[k] = self.take_losses.get(k, 0) + v
        for k, v in other.matsu_losses.items():
            self.matsu_losses[k] = self.matsu_losses.get(k, 0) + v

    def to_dict(self) -> dict:
        lo, hi = wilson_ci(self.take_wins, self.decided)
        return {
            "take_wins": self.take_wins,
            "matsu_wins": self.matsu_wins,
            "draws": self.draws,
            "unfinished": self.unfinished,
            "decided": self.decided,
            "take_win_rate": round(self.take_wins / self.decided, 4) if self.decided else None,
            "take_win_rate_ci95": [round(lo, 4), round(hi, 4)],
            "faults": {"take": self.take_faults, "matsu": self.matsu_faults},
            "take_losses": dict(sorted(self.take_losses.items())),
            "matsu_losses": dict(sorted(self.matsu_losses.items())),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LossTally":
        t = cls(
            take_wins=d.get("take_wins", 0),
            matsu_wins=d.get("matsu_wins", 0),
            draws=d.get("draws", 0),
            unfinished=d.get("unfinished", 0),
            take_faults=d.get("faults", {}).get("take", 0),
            matsu_faults=d.get("faults", {}).get("matsu", 0),
        )
        t.take_losses = dict(d.get("take_losses", {}))
        t.matsu_losses = dict(d.get("matsu_losses", {}))
        return t


# --------------------------------------------------------------------------- #
# Subprocess-isolated contestant (mirrors eval/agent_server protocol)
# --------------------------------------------------------------------------- #
@dataclass
class Contestant:
    label: str
    repo: str
    deck: list[int] = field(default_factory=list)
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    sandbox: Optional[str] = None
    _planner_deck: Optional[list[int]] = field(default=None, repr=False)

    @property
    def python(self) -> str:
        return os.path.join(self.repo, "venv", "bin", "python")

    @property
    def server(self) -> str:
        return os.path.join(self.repo, "eval", "agent_server.py")

    @property
    def cwd(self) -> str:
        return self.sandbox or self.repo

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.python, self.server],
            cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        line = self.proc.stderr.readline()
        if line.strip() != "READY":
            err = self.proc.stderr.read()
            raise RuntimeError(f"{self.label} agent failed to start: {line}{err}")
        self._planner_deck = None

    def set_deck(self, deck: list[int]) -> None:
        assert self.sandbox is not None, "set_deck needs a sandbox cwd"
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        if self._planner_deck == deck:
            return
        self.proc.stdin.write(json.dumps({"__set_deck__": deck}) + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if reply == "":
            raise RuntimeError(f"{self.label} server exited during set_deck")
        payload = json.loads(reply)
        if not (isinstance(payload, dict) and payload.get("__ok__")):
            raise RuntimeError(f"{self.label} set_deck failed: {payload}")
        self._planner_deck = list(deck)

    def act(self, obs: dict) -> list[int]:
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(obs) + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if reply == "":
            raise RuntimeError(f"{self.label} server exited unexpectedly")
        action = json.loads(reply)
        if isinstance(action, dict) and "__error__" in action:
            raise RuntimeError(f"{self.label} agent error: {action['__error__']}")
        return action

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort teardown
            self.proc.kill()
        self.proc = None

    def restart(self) -> None:
        self.stop()
        self.start()


def play_match(game, seat0: Contestant, seat1: Contestant) -> dict:
    """One engine match. Returns winner seat, terminal reason, and fault seat."""
    obs, start = game.battle_start(seat0.deck, seat1.deck)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}")
    steps = 0
    try:
        while steps < MAX_DECISIONS:
            cur = obs.get("current") or {}
            result = cur.get("result", -1)
            if result != -1:
                return {"result": result, "reason": extract_reason(obs),
                        "fault_seat": None}
            seat = cur.get("yourIndex", 0)
            agent = seat0 if seat == 0 else seat1
            try:
                action = agent.act(obs)
            except Exception:  # noqa: BLE001 - agent error => that seat's loss
                return {"result": 1 - seat, "reason": None, "fault_seat": seat}
            try:
                obs = game.battle_select(action)
            except Exception:  # noqa: BLE001 - engine reject => illegal move
                return {"result": 1 - seat, "reason": None, "fault_seat": seat}
            steps += 1
        return {"result": -1, "reason": None, "fault_seat": None}
    finally:
        game.battle_finish()


def run_pairing(game, take: Contestant, matsu: Contestant, schedule: list[str],
                deck_cache: dict[str, list[int]],
                per_deck: dict[str, LossTally], progress: bool = True) -> LossTally:
    """Play the scheduled seat-alternating 竹-vs-松 matches, tallying 竹 losses."""
    overall = LossTally()
    n = len(schedule)
    for i in range(n):
        take_first = (i % 2 == 0)
        deck_file = schedule[i]
        deck = deck_cache[deck_file]
        take.deck = deck
        matsu.deck = deck
        if take.sandbox is not None:
            take.set_deck(deck)
        if matsu.sandbox is not None:
            matsu.set_deck(deck)
        seat0, seat1 = (take, matsu) if take_first else (matsu, take)
        out = play_match(game, seat0, seat1)
        take_seat = 0 if take_first else 1
        result = out["result"]
        fault_by = None
        if out["fault_seat"] is not None:
            fault_by = "take" if out["fault_seat"] == take_seat else "matsu"
        if result == 2:
            take_won: Optional[bool] = None
            reason = -2
        elif result in (0, 1):
            take_won = (result == take_seat)
            reason = out["reason"]
        else:
            take_won = None
            reason = None
        name = os.path.basename(deck_file)
        tally = per_deck.setdefault(name, LossTally())
        for t in (overall, tally):
            t.record(take_won=take_won, reason=reason, fault_by=fault_by)
        if out["fault_seat"] is not None:
            (seat0 if out["fault_seat"] == 0 else seat1).restart()
        if progress and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}  竹 {overall.take_wins} / 松 {overall.matsu_wins} "
                  f"/ draw {overall.draws}  losses={overall.take_losses}",
                  file=sys.stderr, flush=True)
    return overall


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total else "n/a"


def render_md(rep: dict) -> str:
    L: list[str] = []
    L.append("# SOT-1698 対松 敗因分類 — 竹 (RuleBased) vs 松 (MCTS champion)")
    L.append("")
    ov = rep["overall"]
    L.append(f"- 対戦: 竹 vs 松、mirror（同一デッキ両者・先後入替）、{rep['n']} 試合"
             f"（decided {ov['decided']}, draw {ov['draws']}, unfinished {ov['unfinished']}）")
    L.append(f"- デッキプール: {rep['pool_size']} decks ({rep['decks_dir']}), seed={rep['seed']}")
    L.append(f"- fault: 竹={ov['faults']['take']} 松={ov['faults']['matsu']}")
    L.append(f"- **竹 対松勝率: {ov['take_win_rate']} Wilson95 {ov['take_win_rate_ci95']}** "
             f"（竹 {ov['take_wins']} / 松 {ov['matsu_wins']}）")
    L.append("")
    L.append("## 竹の敗因内訳（decided 竹敗北）")
    L.append("")
    tl = ov["take_losses"]
    total_l = sum(tl.values())
    L.append("| 敗因 | 件数 | 敗北比 |")
    L.append("| --- | ---: | ---: |")
    for k in ("prize_out", "deck_out", "no_active", "card_effect", "other"):
        L.append(f"| {k} | {tl.get(k, 0)} | {_pct(tl.get(k, 0), total_l)} |")
    L.append(f"| **合計** | **{total_l}** | 100% |")
    L.append("")
    L.append("SOT-1694 の 25デッキ *mirror*（同系対戦）実測は prize 71% / deck_out 27% / "
             "no_active 2%。上表と比較して対探索型（松）で敗因構成が変わるかを確認する。")
    L.append("")
    L.append("## デッキ別（竹視点）")
    L.append("")
    L.append("| deck | 竹W-松W | 竹勝率 | prize_out | deck_out | no_active | fault(竹/松) |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name in sorted(rep["per_deck"]):
        d = rep["per_deck"][name]
        wr = d["take_win_rate"]
        tlk = d["take_losses"]
        L.append(f"| {name} | {d['take_wins']}-{d['matsu_wins']} | {wr} "
                 f"| {tlk.get('prize_out', 0)} | {tlk.get('deck_out', 0)} "
                 f"| {tlk.get('no_active', 0)} | {d['faults']['take']}/{d['faults']['matsu']} |")
    L.append("")
    return "\n".join(L)


def build_report(args, overall: LossTally, per_deck: dict[str, LossTally],
                 pool_size: int) -> dict:
    return {
        "issue": "SOT-1698",
        "n": args.n,
        "seed": args.seed,
        "decks_dir": args.decks_dir,
        "pool_size": pool_size,
        "overall": overall.to_dict(),
        "per_deck": {k: v.to_dict() for k, v in sorted(per_deck.items())},
        "note": ("engine has no seed API; results statistical (Wilson CI). "
                 "loss reason from engine RESULT log (type 23)."),
    }


def aggregate_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("no reports to aggregate")
    overall = LossTally()
    per_deck: dict[str, LossTally] = {}
    total_n = 0
    for rep in reports:
        total_n += rep.get("n", 0)
        overall.merge(LossTally.from_dict(rep["overall"]))
        for name, d in rep.get("per_deck", {}).items():
            per_deck.setdefault(name, LossTally()).merge(LossTally.from_dict(d))
    base = reports[0]
    return {
        "issue": "SOT-1698",
        "aggregated_from": len(reports),
        "n": total_n,
        "seed": [r.get("seed") for r in reports],
        "decks_dir": base.get("decks_dir"),
        "pool_size": base.get("pool_size"),
        "overall": overall.to_dict(),
        "per_deck": {k: v.to_dict() for k, v in sorted(per_deck.items())},
        "note": "aggregated over independent seeded shards.",
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n", type=int, default=60, help="matches (seat-alternating)")
    p.add_argument("--decks-dir", default="decks/initial")
    p.add_argument("--seed", type=int, default=1698)
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    p.add_argument("--aggregate", nargs="+", default=None, metavar="SHARD.json",
                   help="merge shard JSONs and exit (no matches played)")
    args = p.parse_args(argv)

    if args.aggregate:
        reports = []
        for path in args.aggregate:
            with open(path, encoding="utf-8") as fh:
                reports.append(json.load(fh))
        report = aggregate_reports(reports)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as fh:
                fh.write(render_md(report))
        print(f"AGGREGATED shards={report['aggregated_from']} n={report['n']} "
              f"take_wr={report['overall']['take_win_rate']} "
              f"losses={report['overall']['take_losses']}")
        return 0

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from cg import game

    decks_dir = args.decks_dir
    if not os.path.isabs(decks_dir):
        decks_dir = os.path.join(REPO, decks_dir)
    deck_pool = discover_decks(decks_dir)
    deck_cache = {f: load_deck(f) for f in deck_pool}
    rng = random.Random(f"{args.seed}:take:matsu")
    schedule = build_deck_schedule(args.n, deck_pool, rng)
    print(f"竹 vs 松: {len(deck_pool)} decks, n={args.n}, seed={args.seed}",
          file=sys.stderr, flush=True)

    for _lb, _kanji, repo in (TAKE, MATSU):
        if not os.path.isfile(os.path.join(repo, "main.py")):
            raise SystemExit(f"contestant repo not found: {repo}")

    take = Contestant(label=TAKE[0], repo=TAKE[2])
    matsu = Contestant(label=MATSU[0], repo=MATSU[2])
    take.sandbox = make_sandbox(take.repo)
    matsu.sandbox = make_sandbox(matsu.repo)
    per_deck: dict[str, LossTally] = {}
    try:
        take.start()
        matsu.start()
        overall = run_pairing(game, take, matsu, schedule, deck_cache, per_deck)
    finally:
        for c in (take, matsu):
            c.stop()
            if c.sandbox is not None:
                shutil.rmtree(c.sandbox, ignore_errors=True)

    report = build_report(args, overall, per_deck, len(deck_pool))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(render_md(report))
    ov = report["overall"]
    print(f"竹 vs 松: 竹 {ov['take_wins']} / 松 {ov['matsu_wins']} "
          f"win_rate={ov['take_win_rate']} CI95={ov['take_win_rate_ci95']}  "
          f"竹losses={ov['take_losses']}  faults={ov['faults']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
