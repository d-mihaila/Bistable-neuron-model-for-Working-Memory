#!/usr/bin/env bash
# Input- and weight-noise sweeps (coefficient of variation) for every saved N-MNIST run.
# Same levels as MNIST/run_noise_cv.sh:
#
#   input  CV: 0, 10, 20, 30, 40, 50 %   (events removed / copied per channel, see exp_noise_input_cv.py)
#   weight CV: 0, 10, 20, 30, 40, 50 %   (each synapse x (1 + CV xi), see exp_noise_weights_cv.py)
#   5 noise draws per CV, the same 1280 test digits for every network,
#   read-out at each run's trained delay (100 ms).
#
# Usage (activate your usual Python environment first):
#   bash run_noise_cv.sh              # trained delay only
#   bash run_noise_cv.sh 1000 5000    # also read out after 1 s and 5 s of silence (one extra CSV each)
#
# Results:  results/nmnist_w0_experiments/post_hoc/input_noise_cv/input_noise_cv_sweep*.csv
#           results/nmnist_w0_experiments/post_hoc/weight_noise_cv/weight_noise_cv_sweep*.csv
# Logs:     results/nmnist_w0_experiments/post_hoc/logs/

set -euo pipefail
cd "$(dirname "$0")"                                # N_MNIST/ (data, cache and results paths are relative)

LOG_DIR=results/nmnist_w0_experiments/post_hoc/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M)

INPUT_CV="0 0.10 0.20 0.30 0.40 0.50"
WEIGHT_CV="0 0.10 0.20 0.30 0.40 0.50"
REPEATS=5

run() {   # $1 = optional --t_wait value (ms)
    local extra="" tag="trained"
    if [ -n "${1:-}" ]; then extra="--t_wait $1"; tag="twait$1"; fi
    echo ">>> input noise ($tag)"
    python exp_noise_input_cv.py   --cv_values $INPUT_CV  --n_repeats $REPEATS $extra \
        2>&1 | tee "$LOG_DIR/input_noise_cv_${tag}_${STAMP}.log"
    echo ">>> weight noise ($tag)"
    python exp_noise_weights_cv.py --cv_values $WEIGHT_CV --n_repeats $REPEATS $extra \
        2>&1 | tee "$LOG_DIR/weight_noise_cv_${tag}_${STAMP}.log"
}

run                       # each run's trained delay
for T in "$@"; do run "$T"; done

echo "Done. CSVs in results/nmnist_w0_experiments/post_hoc/{input,weight}_noise_cv/"
