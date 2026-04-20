#!/usr/bin/env bash
# Run every (non-smoke) registered experiment, in the correct dependency order.
#
# Order matters for one pair: cartpole_sarsa must finish before
# cartpole_vi_trained_eps_* (those load SARSA Q-tables as the sampling policy).
#
# Uses `;` (not `&&`) between phases so a failure in one phase doesn't
# prevent later phases from running. Individual experiment failures
# inside a phase are logged by run.py and the sweep continues.

set -u
LOG_DIR=".logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/run_all_${STAMP}.log"

# Prefer the repo-local venv if present, else fall back to `python` on PATH.
# Override explicitly with `PY=/path/to/python bash scripts/run_all_experiments.sh`.
if [ -z "${PY:-}" ]; then
    if [ -x ".venv/bin/python" ]; then
        PY=".venv/bin/python"
    else
        PY=$(command -v python)
    fi
fi

phase() {
    local name=$1
    local prefix=$2
    echo "=========================================================="  | tee -a "$MASTER_LOG"
    echo "[$(date +%T)] PHASE: $name (prefix=$prefix)"                  | tee -a "$MASTER_LOG"
    echo "=========================================================="  | tee -a "$MASTER_LOG"
    "$PY" scripts/run.py --prefix "$prefix" 2>&1 | tee -a "$MASTER_LOG"
    echo "[$(date +%T)] PHASE DONE: $name"                              | tee -a "$MASTER_LOG"
}

echo "[$(date +%T)] START run_all" | tee -a "$MASTER_LOG"

phase "blackjack (DP + tabular, 28 exps)"       "blackjack_"
phase "cartpole SARSA (13 exps)"                 "cartpole_sarsa_"
phase "cartpole Q-learning (13 exps)"            "cartpole_qlearning_"
phase "cartpole VI on estimated MDP (16 exps)"   "cartpole_vi_"
phase "cartpole PI on estimated MDP (5 exps)"    "cartpole_pi_"
phase "DQN Rainbow-medium ablation (6 exps)"     "dqn_ablation_"

echo "[$(date +%T)] ALL DONE" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
