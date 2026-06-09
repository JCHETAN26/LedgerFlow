#!/usr/bin/env bash
#
# Reproduce a LedgerFlow experiment from a clean checkout.
#
# Usage:
#   scripts/reproduce_experiment.sh [GIT_COMMIT]
#
# With no argument it reproduces the current HEAD. With a commit hash it checks
# out that commit, pulls the matching DVC-tracked artifacts, and re-runs the
# full pipeline so the experiment is bit-for-bit reproducible.
set -euo pipefail

COMMIT="${1:-}"

if [[ -n "${COMMIT}" ]]; then
  echo ">> Checking out ${COMMIT}"
  git checkout "${COMMIT}"
fi

echo ">> Pulling DVC-tracked artifacts (skipped if no remote configured)"
dvc pull || echo "   (dvc pull skipped — remote not reachable)"

echo ">> Reproducing the pipeline"
dvc repro

echo ">> Metrics for this run:"
dvc metrics show

echo ">> Done. Recommendation memo: reports/recommendation_memo.md"
