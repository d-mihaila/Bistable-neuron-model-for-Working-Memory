"""
exp_noise_weights_cv.py

Weight-noise robustness of the trained N-MNIST runs, with the noise given as a
COEFFICIENT OF VARIATION (CV) of each weight. No retraining.
Identical to MNIST/exp_noise_weights_cv.py, except for the data and the fixed test subset.

How the noise is made
---------------------
Every synaptic weight of w1 (input -> hidden) and w2 (hidden -> output) is scaled
by its own random factor:

    w_noisy = w * (1 + CV * xi),   xi ~ N(0, 1), one xi per synapse

so each weight goes up or down by CV of its own size. A weight changes sign only if
xi < -1/CV: never in practice at CV = 0.1 (10 s.d.), about 2 % of the weights at CV = 0.5. w0 is never touched. The clean
weights are restored after every evaluation, so noise never accumulates.

Test digits
-----------
N_MNIST's test loader is shuffled, so a FIXED random subset of `--n_digits` test
digits (seed `--subset_seed`) is used for every network, CV and repeat -- the same
subset as exp_noise_input_cv.py with the same arguments.

Each CV is evaluated `--n_repeats` times, each with a new noise draw (seed + k).

Output
------
    results/nmnist_w0_experiments/post_hoc/weight_noise_cv/weight_noise_cv_sweep[_twait<T>].csv
    columns: run_name, label, cv, repeat, seed, t_wait, n_digits, test_acc

Usage (from N_MNIST/)
---------------------
    python exp_noise_weights_cv.py
    python exp_noise_weights_cv.py --cv_values 0 0.025 0.05 0.075 0.1 --n_repeats 5
    python exp_noise_weights_cv.py --t_wait 5000        # read out after 5 s of silence instead
"""
import argparse
import os

import pandas as pd
import torch

import nmnist_w0_lib as lib
from rebuild_network import load_all_trained_models
from exp_noise_input_cv import fixed_test_loader

DEFAULT_CV_VALUES = (0.0, 0.025, 0.05, 0.075, 0.10)
DEFAULT_SAVE_CSV = "weight_noise_cv_sweep.csv"


def evaluate_with_weight_cv(model, loader, device, dtype, cv, seed):
    """Scales w1/w2 in place by (1 + cv * xi), evaluates on every digit of `loader`,
    restores the exact clean weights."""
    clean = {n: getattr(model, n).detach().clone() for n in ("w1", "w2")}
    g = torch.Generator().manual_seed(int(seed))
    try:
        if cv > 0:
            with torch.no_grad():
                for n, w in clean.items():
                    xi = torch.randn(tuple(w.shape), generator=g, dtype=w.dtype).to(w.device)
                    getattr(model, n).copy_(w * (1.0 + cv * xi))
        return lib.evaluate(model, loader, device, dtype, max_batches=None)
    finally:
        with torch.no_grad():
            for n, w in clean.items():
                getattr(model, n).copy_(w)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=lib.DEFAULT_OUT_DIR)
    parser.add_argument("--cv_values", type=float, nargs="+", default=list(DEFAULT_CV_VALUES))
    parser.add_argument("--n_repeats", type=int, default=5, help="noise draws per CV (different seeds)")
    parser.add_argument("--seed", type=int, default=0, help="repeat k uses seed + k")
    parser.add_argument("--n_digits", type=int, default=1280, help="fixed test subset (1280 = as MNIST)")
    parser.add_argument("--subset_seed", type=int, default=0)
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

    save_dir = os.path.join(args.out_dir, "post_hoc", "weight_noise_cv")
    save_csv = DEFAULT_SAVE_CSV if args.t_wait is None else DEFAULT_SAVE_CSV.replace(".csv", f"_twait{args.t_wait}.csv")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_csv)

    loader, idx = fixed_test_loader(args.batch_size, args.n_digits, args.subset_seed)
    print(f"{len(idx)} fixed test digits (subset seed {args.subset_seed})")
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
                acc = evaluate_with_weight_cv(model, loader, device, dtype, cv, seed)
                accs.append(acc)
                records.append(dict(run_name=run_name, label=e["label"], cv=cv, repeat=k, seed=seed,
                                    t_wait=model.T_wait, n_digits=len(idx), test_acc=acc))
            m = sum(accs) / len(accs)
            print(f"    weight CV = {cv * 100:4.1f}%   test_acc = {m * 100:6.2f}%  "
                  f"(min {min(accs) * 100:.2f}, max {max(accs) * 100:.2f}, n = {len(accs)})")
        df = pd.DataFrame(records)                       # save after every run
        if args.append and os.path.exists(out_path):
            old = pd.read_csv(out_path)
            df = pd.concat([old[~old.run_name.isin(df.run_name.unique())], df], ignore_index=True)
        df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
