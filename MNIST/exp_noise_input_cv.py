"""
exp_noise_input_cv.py

Input-noise robustness of the trained MNIST runs, with the noise given as a
COEFFICIENT OF VARIATION (CV) of the input rate. No retraining.

How the noise is made
---------------------
Each pixel's spike rate r (pixel intensity in [0, 1]) is scaled by a random factor:

    r_noisy = clamp( r * (1 + CV * xi), 0, 1 ),   xi ~ N(0, 1)

so the rate goes up or down, with std(r_noisy) / r = CV for every pixel. A pixel
with r = 0 stays at 0 (the noise is relative, it adds no background spikes).
The factor is drawn ONCE per pixel per digit and kept for the whole 100 ms
stimulus (the rate itself changes, not each time step). Spikes are then sampled
as usual: Bernoulli(r_noisy) at every step. The clamp caps rates at 1; at 0 it acts
only when xi < -1/CV (5 s.d. at CV = 0.2; 2 s.d., about 2 % of pixels, at CV = 0.5).

This differs from exp_noise_input.py (kept unchanged), which adds N(0, std) to
every rate at every step, so its noise is not relative to the rate.

Each CV is evaluated `--n_repeats` times with a different seed (different noise
and Bernoulli draws), on the same `--max_batches` x 128 test digits.

Output
------
    results/mnist_ms_experiments/post_hoc/input_noise_cv/input_noise_cv_sweep[_twait<T>].csv
    columns: run_name, label, cv, repeat, seed, t_wait, n_digits, test_acc

Usage (from MNIST/)
----------------------
    python exp_noise_input_cv.py
    python exp_noise_input_cv.py --cv_values 0 0.05 0.1 0.15 0.2 --n_repeats 5
    python exp_noise_input_cv.py --t_wait 5000          # read out after 5 s of silence instead
    python exp_noise_input_cv.py --runs const_w0_-1.00 all_lif_wait100 --append
"""
import argparse
import os

import pandas as pd
import torch

import mnist_lib as lib
from save_experiment import load_all_trained_models

DEFAULT_CV_VALUES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_SAVE_CSV = "input_noise_cv_sweep.csv"


def make_cv_encode_batch(cv, generator):
    """Drop-in replacement for mnist_lib.encode_batch: multiplicative rate noise with
    coefficient of variation `cv`, frozen over the stimulus, drawn from `generator` (CPU)."""

    def encode(imgs, device, dtype, num_steps=lib.NUM_STEPS):
        rates = imgs.reshape(imgs.shape[0], -1).to(dtype=dtype)                 # (batch, N_IN), on CPU
        if cv > 0:
            xi = torch.randn(rates.shape, generator=generator, dtype=dtype)
            rates = torch.clamp(rates * (1.0 + cv * xi), 0.0, 1.0)
        rates = rates.unsqueeze(1).expand(-1, num_steps, -1).contiguous()     # same rate at every step
        spikes = torch.bernoulli(rates, generator=generator)
        return spikes.to(device=device)

    return encode


def evaluate_with_input_cv(model, loader, device, dtype, cv, seed, max_batches):
    """lib.evaluate() with mnist_lib.encode_batch swapped for the noisy version; always restored."""
    g = torch.Generator().manual_seed(int(seed))
    original = lib.encode_batch
    lib.encode_batch = make_cv_encode_batch(cv, g)
    try:
        return lib.evaluate(model, loader, device, dtype, max_batches=max_batches)
    finally:
        lib.encode_batch = original


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=lib.DEFAULT_OUT_DIR)
    parser.add_argument("--cv_values", type=float, nargs="+", default=list(DEFAULT_CV_VALUES))
    parser.add_argument("--n_repeats", type=int, default=5, help="noise draws per CV (different seeds)")
    parser.add_argument("--seed", type=int, default=0, help="repeat k uses seed + k")
    parser.add_argument("--max_batches", type=int, default=10, help="10 x 128 = 1280 test digits")
    parser.add_argument("--batch_size", type=int, default=lib.DEFAULT_BATCH_SIZE)
    parser.add_argument("--t_wait", type=int, default=None,
                        help="silent delay (ms) before the read-out; default: each run's trained T_wait")
    parser.add_argument("--runs", type=str, nargs="*", default=None,
                        help="only these run names (default: every saved run)")
    parser.add_argument("--append", action="store_true",
                        help="keep the existing CSV, replace only the rows of the runs evaluated now")
    args = parser.parse_args()

    device, dtype = lib.get_device(), torch.float
    print("device:", device)
    entries = load_all_trained_models(out_dir=args.out_dir, device=device, dtype=dtype)
    if args.runs:
        entries = [e for e in entries if e["run_name"] in args.runs]
    print(f"Rebuilt {len(entries)} trained model(s) from {args.out_dir}")
    if not entries:
        return

    save_dir = os.path.join(args.out_dir, "post_hoc", "input_noise_cv")
    save_csv = DEFAULT_SAVE_CSV if args.t_wait is None else DEFAULT_SAVE_CSV.replace(".csv", f"_twait{args.t_wait}.csv")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_csv)

    _, testloader = lib.get_dataloaders(batch_size=args.batch_size)
    n_digits = args.max_batches * args.batch_size
    records = []
    for e in entries:
        run_name, model = e["run_name"], e["model"]
        if args.t_wait is not None:
            model.T_wait = args.t_wait
        print(f"{'=' * 60}\n  {run_name}  ({e['label']}), T_wait = {model.T_wait} ms\n{'=' * 60}")
        for cv in args.cv_values:
            accs = []
            for k in range(args.n_repeats):
                seed = args.seed + k
                acc = evaluate_with_input_cv(model, testloader, device, dtype, cv, seed, args.max_batches)
                accs.append(acc)
                records.append(dict(run_name=run_name, label=e["label"], cv=cv, repeat=k, seed=seed,
                                    t_wait=model.T_wait, n_digits=n_digits, test_acc=acc))
            m = sum(accs) / len(accs)
            print(f"    input CV = {cv * 100:4.1f}%   test_acc = {m * 100:6.2f}%  "
                  f"(min {min(accs) * 100:.2f}, max {max(accs) * 100:.2f}, n = {len(accs)})")
        # save after every run, so an interrupted sweep keeps what is done
        df = pd.DataFrame(records)
        if args.append and os.path.exists(out_path):
            old = pd.read_csv(out_path)
            df = pd.concat([old[~old.run_name.isin(df.run_name.unique())], df], ignore_index=True)
        df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
