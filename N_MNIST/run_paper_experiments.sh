#!/usr/bin/env bash
# Every N-MNIST experiment of the paper, in order. Safe to run again: a network that is already
# trained (results/nmnist_w0_experiments/<run>/weights.pt exists) is skipped, so the shipped
# networks are kept and only missing ones are trained. The post-hoc sweeps are re-run on all.
#
#   1. train the 10 networks of the paper (seed 42, 100 ms input + 100 ms silent delay)
#   2. accuracy against delay, 0.1-5 s           exp_waiting.py         -> post_hoc/waiting/
#   3. spikes per 100 ms over a 5 s trial        exp_delay_activity.py  -> post_hoc/delay_activity/
#   4. robustness: weight and input noise, CV 0-50 %   run_noise_cv.sh -> post_hoc/{weight,input}_noise_cv/
#
# Usage (from anywhere; activate your Python environment first):
#   bash run_paper_experiments.sh
# N-MNIST is downloaded by tonic into ./data the first time (and cached as frames in ./cache).

set -euo pipefail
cd "$(dirname "$0")"
OUT=results/nmnist_w0_experiments

train() {   # $1 = run name, rest = command
    local run=$1; shift
    if [ -f "$OUT/$run/weights.pt" ]; then echo "    $run: already trained, skipped"; else "$@"; fi
}

echo ">>> 1. training"
# every network: 50 epochs. (The shipped constant-w0, heterogeneous and LIF networks were trained in two
# stages, 25 + 25 epochs and 10 + 40 for the LIF, with continue_training.py; see "resume_log" in their config.json.)
train all_lif_wait100 python run_all_lif.py --t_wait 100 --nb_epochs 50
for w0 in -2 -1 -0.5 0.5 1 2; do
    train "$(printf 'const_w0_%+.2f' "$w0")" python run_const_w0.py --w0 "$w0" --nb_epochs 50
done
train lif_hidden_bistable_output_w0_-1.00 python run_lif_hidden.py --w0 -1 --nb_epochs 50    # heterogeneous
train dist_two_tailed_mean-1.00_std0.50_learnable \
    python run_distribution_w0.py --dist two_tailed --mean -1 --std 0.5 --learnable --nb_epochs 50
train dist_uniform_mean-1.00_std0.50_fixed \
    python run_distribution_w0.py --dist uniform --mean -1 --std 0.5 --nb_epochs 50

echo ">>> 2. accuracy against delay"
python exp_waiting.py
echo ">>> 3. activity during the delay"
python exp_delay_activity.py
echo ">>> 4. robustness"
bash run_noise_cv.sh
echo "Done. Figures: ../NICE_figures/fig2_mnist_nmnist.ipynb and supplementary.ipynb"
