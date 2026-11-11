"""
exp_loop_rates.py

Firing rate per cue presentation, for the six 500 ms-cue networks (MNIST-LIF I neurons).

Re-runs section 3 of ODR_w0_gating.ipynb exactly (ten cue -> distractor -> readout
-> forget loops per trial, 2 batches of 8 trials, the same seeds, angles and delay shifts, 500 ms gaps)
and keeps, for every memory, the firing rate over its whole cue presentation:

    loop k = from the start of the forget window before cue k (t = 0 for cue 1)
             to the start of the next forget window (the end of the trial for the last cue)

which covers forget window -> gap -> cue -> hold with the distractor -> readout -> gap. The window is
the same in every network (networks without a forget signal still have the 200 ms window).
New columns (w0_gating.sequence_table):
    loop_t0_ms, loop_t1_ms   the window
    rate_loop_e_hz           mean rate per E neuron (1024)
    rate_loop_i_hz           mean rate per I neuron (256)
    rate_loop_hz             mean rate over all 1280 neurons (E + I): the "total" rate

Every other column is the same as in cue500_repeats.csv; the script checks that the errors match.

Output: results/w0_gating_cue500_mnistLIF/cue500_repeats_looprates.csv

Usage (from ODR/):
    python exp_loop_rates.py
"""
import os
import time

import numpy as np
import pandas as pd

import w0_experiments as X

OUT = "results/w0_gating_cue500_mnistLIF"
CSV = os.path.join(OUT, "cue500_repeats_looprates.csv")


def main():
    X.I_MODEL = "lif"                                     # as ODR_w0_gating.ipynb
    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    reps = X.repeat_sweep(X.CUE500_CASES, n_loops=10, n_batches=2, n_trials=8, gap=500)
    reps.to_csv(CSV, index=False)
    print(f"saved {CSV} ({len(reps)} rows) in {(time.time() - t) / 60:.1f} min")

    # same networks, seeds and angles as the notebook: the errors should be the ones in cue500_repeats.csv
    old_path = os.path.join(OUT, "cue500_repeats.csv")
    if os.path.exists(old_path):
        old = pd.read_csv(old_path)
        key = ["setup", "batch", "trial", "memory"]
        m = reps.merge(old[key + ["error_deg"]], on=key, suffixes=("", "_notebook"))
        diff = np.abs(m.error_deg - m.error_deg_notebook)
        print(f"error vs cue500_repeats.csv: {len(m)} memories compared, max difference {diff.max():.3g} deg, "
              f"{(diff > 1e-3).mean() * 100:.1f} % differ")

    d = reps[reps.memory >= 2]
    print("\nmemories 2-10, mean over trials:")
    print(d.groupby("setup")[["error_deg", "rate_loop_hz", "rate_loop_e_hz", "rate_loop_i_hz"]].mean()
          .reindex(list(X.CUE500_CASES)).round(2).to_string())


if __name__ == "__main__":
    main()
