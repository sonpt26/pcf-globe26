#!/bin/bash
# PCF-MORL: Run all experiments for GLOBECOM 2026 paper
#
# Usage:
#   ./run_experiments.sh              # Run all experiments
#   ./run_experiments.sh 1            # Run only Exp 1
#   ./run_experiments.sh 1 2 3        # Run Exp 1, 2, 3
#   ./run_experiments.sh --seeds 3    # Override seed count
#
# Prerequisites:
#   - ns-3 binary built: 5g-factory-sim/ns-3/build/contrib/nr/examples/ns3.46-pcf-morl-scenario-default
#   - Python deps: morl-baselines, pymoo, scipy, torch
#   - (Optional) Trained GPI-PD checkpoint for Exp 1-3

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──
SEEDS=5
BASE_SEED=42
DEVICE="auto"
MAX_STEPS=100
GPIPD_CKPT=""
EXPS=()

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)     SEEDS="$2"; shift 2 ;;
        --base-seed) BASE_SEED="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --checkpoint) GPIPD_CKPT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [EXP_NUMS...] [OPTIONS]"
            echo ""
            echo "Experiments:"
            echo "  1  QoS Performance (PCF-MORL vs A1-A3 vs DQN on E1-E5)"
            echo "  2  Zero-shot adaptation (GPI vs Oracle, 20 test weights)"
            echo "  3  24h factory cycle (8 phase transitions)"
            echo "  4  Parallel scaling (P={1,2,4,8,16})"
            echo "  5  K tuning (K={50,100,200,500,1000}, P=8)"
            echo "  6  SF sharing ablation (4 strategies, P=8)"
            echo ""
            echo "Options:"
            echo "  --seeds N        Number of seeds (default: 5)"
            echo "  --base-seed N    Base seed (default: 42)"
            echo "  --device DEV     auto|cuda|cpu (default: auto)"
            echo "  --max-steps N    Steps per episode (default: 100)"
            echo "  --checkpoint P   Path to trained GPI-PD checkpoint"
            exit 0
            ;;
        [1-6]) EXPS+=("$1"); shift ;;
        *)     echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Default: run all
if [ ${#EXPS[@]} -eq 0 ]; then
    EXPS=(1 2 3 4 5 6)
fi

# ── Preflight checks ──
echo "============================================================"
echo "PCF-MORL Experiment Runner"
echo "============================================================"
echo "Experiments: ${EXPS[*]}"
echo "Seeds:       $SEEDS (base=$BASE_SEED)"
echo "Device:      $DEVICE"
echo "Max steps:   $MAX_STEPS"
echo "Checkpoint:  ${GPIPD_CKPT:-none}"
echo ""

NS3_BIN="$SCRIPT_DIR/../5g-factory-sim/ns-3/build/contrib/nr/examples/ns3.46-pcf-morl-scenario-default"
if [ ! -f "$NS3_BIN" ]; then
    echo "ERROR: ns-3 binary not found at $NS3_BIN"
    echo "Build it first: cd ../5g-factory-sim/ns-3 && ./ns3 build"
    exit 1
fi
echo "ns-3 binary: OK"

python3 -c "from experiments.experiment_runner import EXPERIMENTS; print(f'Python imports: OK ({len(EXPERIMENTS)} experiments)')" || {
    echo "ERROR: Python import failed. Check dependencies."
    exit 1
}
echo ""

# ── Build checkpoint flag ──
CKPT_FLAG=""
if [ -n "$GPIPD_CKPT" ]; then
    CKPT_FLAG="--gpipd-checkpoint $GPIPD_CKPT"
fi

# ── Run experiments ──
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

LOGFILE="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOGFILE"
echo ""

run_exp() {
    local exp_num=$1
    local start_time=$(date +%s)

    echo "============================================================"
    echo "Experiment $exp_num — started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    python3 -m experiments.experiment_runner \
        --exp "$exp_num" \
        --seeds "$SEEDS" \
        --base-seed "$BASE_SEED" \
        --device "$DEVICE" \
        --max-steps "$MAX_STEPS" \
        $CKPT_FLAG \
        2>&1 | tee -a "$LOGFILE"

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "Experiment $exp_num finished in ${elapsed}s ($(date '+%H:%M:%S'))"
    echo ""
}

TOTAL_START=$(date +%s)

for exp in "${EXPS[@]}"; do
    run_exp "$exp"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "============================================================"
echo "All experiments complete"
echo "Total time: ${TOTAL_ELAPSED}s ($(printf '%02d:%02d:%02d' $((TOTAL_ELAPSED/3600)) $(((TOTAL_ELAPSED%3600)/60)) $((TOTAL_ELAPSED%60))))"
echo "Results in: $RESULTS_DIR/"
echo "Log: $LOGFILE"
echo "============================================================"
