#!/bin/bash
# Pack a Kaggle submission: main.py + deck.csv + cg/ + agents/ at the archive top level.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
[ -d cg ] && [ -d agents ] && [ -f main.py ] && [ -f deck.csv ] || { echo "missing cg/ or agents/ or main.py or deck.csv (run setup_engine.sh)"; exit 1; }
tar --exclude='__pycache__' -czf submission.tar.gz main.py deck.csv cg agents
echo "wrote $REPO/submission.tar.gz"; tar -tzf submission.tar.gz | head
