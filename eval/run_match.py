"""Local self-play match runner for the PTCG AI Battle eval environment.
Loads the cabt engine (cg/) and plays a full agent-vs-agent match.
Usage: python eval/run_match.py [deck0.csv] [deck1.csv]
Run from repo root (after scripts/setup_engine.sh has populated cg/).
"""
import sys, os, random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)         # make `cg` importable
os.chdir(REPO)                   # so libcg.so & deck.csv resolve

from cg import game
from cg.api import to_observation_class

def load_deck(path):
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]

def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return None
    n = len(obs.select.option)
    k = max(obs.select.minCount, min(obs.select.maxCount, n))
    return random.sample(range(n), k) if n else []

def run(deck0, deck1, max_steps=100000):
    obs, start = game.battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(f"BattleStart failed: errorPlayer={start.errorPlayer} errorType={start.errorType}")
    steps = 0
    while steps < max_steps:
        cur = obs.get("current")
        if cur and cur.get("result", -1) != -1:
            return cur["result"], steps
        obs = game.battle_select(random_agent(obs))
        steps += 1
    return -1, steps

if __name__ == "__main__":
    random.seed(42)
    d0 = load_deck(sys.argv[1]) if len(sys.argv) > 1 else load_deck("deck.csv")
    d1 = load_deck(sys.argv[2]) if len(sys.argv) > 2 else d0
    result, steps = run(d0, d1)
    game.battle_finish()
    print(f"MATCH DONE: winner=player{result}  decisions={steps}")
