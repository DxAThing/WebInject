#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

MODE="${1:-all}"

if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

$PYTHON main_pipeline.py "$MODE"