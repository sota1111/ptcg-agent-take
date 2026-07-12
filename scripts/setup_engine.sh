#!/bin/bash
# Copy the license-restricted engine (cg/) and card data into this repo (both gitignored).
# Override the source with: SRC=/path/to/pokemon-tcg-ai-battle/extracted bash scripts/setup_engine.sh
set -e
SRC="${SRC:-/workspaces/kaggle-ptcg-ume/data/simulation/extracted}"
SAMPLE="$SRC/sample_submission/sample_submission"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[ -d "$SAMPLE/cg" ] || { echo "engine not found at $SAMPLE/cg — set SRC="; exit 1; }
rm -rf "$REPO/cg"; cp -r "$SAMPLE/cg" "$REPO/cg"
mkdir -p "$REPO/data"; cp "$SRC"/*.csv "$REPO/data/" 2>/dev/null || true
echo "engine -> $REPO/cg ; data -> $REPO/data (both gitignored)"
