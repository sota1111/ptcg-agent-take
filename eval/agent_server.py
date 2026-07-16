"""Subprocess agent server for the cross-repo 松竹梅 battle (SOT-1681).

Runs one project's Kaggle submission agent (``main.agent``) in its OWN process,
working directory and virtualenv, and exposes it over a trivial line-delimited
JSON protocol so a host process (``eval/battle_matsu_take_ume.py``) can drive it
without importing that repo's ``agents`` / ``cg`` packages.

Why a subprocess instead of an in-process import: the three sibling repos
(``ptcg-agent-matsu`` / ``-take`` / ``-ume``) each ship a top-level ``agents``
package whose module names collide (``base``, ``random_agent``, ``search_agent``
exist in more than one), so they cannot all be imported side-by-side in one
interpreter. Isolating each agent in its own process side-steps the collision
entirely and lets each ``main.agent`` resolve its own ``deck.csv`` / native
engine relative to its repo root.

Protocol (one JSON value per line, both directions):

* stdin  ← ``obs_dict`` (the raw engine observation, exactly what the Kaggle
  harness passes to ``agent(obs_dict)``) — OR a control object
  ``{"__set_deck__": [card_ids...]}`` that swaps the deck this agent plans with
  (see below) and replies ``{"__ok__": true}``.
* stdout → the action, a ``list[int]`` of option indices; or, if the agent
  raised, ``{"__error__": "<ExceptionType>: <message>"}`` so the host can
  attribute the fault to this agent (a loss) instead of crashing the batch.

Deck injection (``__set_deck__``, SOT-1681 mirror/independent-random). For the
25-deck random modes the host must make the deck the engine deals match the deck
this agent's planner reasons about (松 MCTS determinizes from its ``deck.csv``;
梅 MCTS/harness reads ``deck.csv`` at construction; 竹 is deck-free and
unaffected). The server is launched with ``cwd`` set to a per-contestant
*sandbox* dir (symlinks to the repo, with a writable ``deck.csv``); on
``__set_deck__`` it rewrites ``<cwd>/deck.csv`` and ``importlib.reload(main)`` so
``main.agent`` is rebuilt from the new deck — an in-process rebuild (no process
respawn, so the heavy ``agents``/``cg`` imports stay cached). No source change to
any repo's ``main.py`` is needed; ``deck.csv`` is resolved relative to this
sandbox cwd (松) or next to ``main.py`` in the sandbox (梅).

The server prints a single ``READY`` line to stderr once ``main.agent`` is
importable, then serves requests until stdin is closed. It is launched as::

    <repo>/venv/bin/python <this file>   # with cwd=<repo> or <sandbox>

``cwd`` is prepended to ``sys.path`` so ``import main`` / the repo's ``cg``
resolve locally.
"""
import importlib
import json
import os
import sys


def _write_deck(deck: list) -> None:
    """Rewrite ``<cwd>/deck.csv`` with the given card ids (one per line)."""
    with open("deck.csv", "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(int(c)) for c in deck) + "\n")


def main() -> int:
    sys.path.insert(0, os.getcwd())  # repo/sandbox root: resolve `main` / `cg`
    import main as main_mod  # the project's Kaggle submission entry point
    assert hasattr(main_mod, "agent")

    sys.stderr.write("READY\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        # Control message: swap the deck the agent plans with, then rebuild it.
        if isinstance(msg, dict) and "__set_deck__" in msg:
            try:
                _write_deck(msg["__set_deck__"])
                main_mod = importlib.reload(main_mod)  # rebuild agent from deck
                assert hasattr(main_mod, "agent")
                payload = {"__ok__": True}
            except Exception as exc:  # noqa: BLE001 - report, never crash
                payload = {"__error__": f"{type(exc).__name__}: {exc}"}
            sys.stdout.write(json.dumps(payload))
            sys.stdout.write("\n")
            sys.stdout.flush()
            continue
        # Otherwise it is an observation: ask the agent for an action.
        try:
            action = main_mod.agent(msg)
        except Exception as exc:  # noqa: BLE001 - report, never crash the server
            payload = {"__error__": f"{type(exc).__name__}: {exc}"}
        else:
            payload = action
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
