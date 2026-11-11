"""
exp_noise_input_cv.py

Input-noise robustness of the trained N-MNIST runs, with the noise given as a
COEFFICIENT OF VARIATION (CV) of each input channel's event rate. No retraining.
Same definition as MNIST/exp_noise_input_cv.py, adapted to event data.

How the noise is made
---------------------
The input is a tensor of DVS event COUNTS per 1 ms bin (tonic ToFrame, 100 bins,
2 x 34 x 34 channels = polarity x pixel). Each channel of each digit gets ONE
random factor, kept for the whole stimulus:

    f = clamp(1 + CV * xi, 0, 2),   xi ~ N(0, 1)

and its events are changed so that, on average, the channel fires f times as much:

    f < 1 : events are REMOVED  -- each event is kept with probability f
            (count -> Binomial(count, f))
    f > 1 : events are ADDED    -- each event gets a copy with probability f - 1,
            in the same time bin (count -> count + Binomial(count, f - 1))

So the expected count of every channel is f x its clean count, the spread of f across
channels is CV, and the timing of the events is kept (new events only land where real
events are). A silent channel stays silent (the noise is relative). The clamp acts only
when |xi| > 1/CV: never in practice for CV <= 0.2 (5 s.d.), about 2 % of channels at each
end at CV = 0.5 (2 s.d.: silenced, or doubled).

This differs from exp_noise_input.py (kept unchanged), which adds independent
Bernoulli(p_noise) background spikes everywhere.

Test digits
-----------
N_MNIST's test loader is shuffled, so every evaluation would see other digits. Here a
FIXED random subset of `--n_digits` test digits (seed `--subset_seed`) is used for every
network, CV and repeat, so the curves differ only by the noise.

Each CV is evaluated `--n_repeats` times with a different noise seed (seed + k).

Output
------
    results/nmnist_w0_experiments/post_hoc/input_noise_cv/input_noise_cv_sweep[_twait<T>].csv
    columns: run_name, label, cv, repeat, seed, t_wait, n_digits, event_ratio, test_acc
    (event_ratio = noisy / clean event count over all digits: a check, ~1 at every CV)

Usage (from N_MNIST/)
---------------------
    python exp_noise_input_cv.py
    python exp_noise_input_cv.py --cv_values 0 0.05 0.1 0.15 0.2 --n_repeats 5
    python exp_noise_input_cv.py --t_wait 5000          # read out after 5 s of silence instead
    python exp_noise_input_cv.py --runs const_w0_-1.00 all_lif_wait100 --append
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import nmnist_w0_lib as lib
from rebuild_network import load_all_trained_models

DEFAULT_CV_VALUES = (0.0, 0.05, 0.10, 0.15, 0.20)
DEFAULT_SAVE_CSV = "input_noise_cv_sweep.csv"


def fixed_test_loader(batch_size, n_digits, subset_seed):
    """The same `n_digits` test digits every time (fixed random subset, not shuffled)."""
    _, testloader = lib.get_dataloaders(batch_size=batch_size)
    ds = testloader.dataset
    idx = np.sort(np.random.default_rng(subset_seed).choice(len(ds), size=n_digits, replace=False))
    return DataLoader(Subset(ds, idx.tolist()), batch_size=batch_size, shuffle=False,
                      collate_fn=testloader.collate_fn), idx


def add_remove_events(x, cv, generator):
    """x: event counts (batch, T, ...) on CPU. Returns the noisy counts (float, same shape)."""
    x = x.float()
    if cv <= 0:
        return x
    shape = (x.shape[0], 1) + tuple(x.shape[2:])                   # one factor per digit and channel
    f = torch.clamp(1.0 + cv * torch.randn(shape, generator=generator), 0.0, 2.0).expand_as(x)
    keep = torch.binomial(x, torch.clamp(f, max=1.0), generator=generator)          # f < 1: thin
    extra = torch.binomial(x, torch.clamp(f - 1.0, min=0.0), generator=generator)   # f > 1: copy
    return torch.where(f < 1.0, keep, x + extra)


def evaluate_with_input_cv(model, loader, device, dtype, cv, seed):
    """Accuracy and noisy/clean event ratio on every digit of `loader`."""
    g = torch.Generator().manual_seed(int(seed))
    correct = total = 0
    n_clean = n_noisy = 0.0
    with torch.no_grad():
        for x, y in loader:
            xn = add_remove_events(x, cv, g)
            n_clean += float(x.sum())
            n_noisy += float(xn.sum())
            spike_count, _ = model.forward(lib.flatten_batch(xn, device, dtype))
            correct += (spike_count.argmax(1).cpu() == y).sum().item()
            total += y.numel()
    return correct / total, n_noisy / max(n_clean, 1.0)


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

    save_dir = os.path.join(args.out_dir, "post_hoc", "input_noise_cv")
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
                acc, ratio = evaluate_with_input_cv(model, loader, device, dtype, cv, seed)
                accs.append(acc)
                records.append(dict(run_name=run_name, label=e["label"], cv=cv, repeat=k, seed=seed,
                                    t_wait=model.T_wait, n_digits=len(idx), event_ratio=ratio, test_acc=acc))
            m = sum(accs) / len(accs)
            print(f"    input CV = {cv * 100:4.1f}%   test_acc = {m * 100:6.2f}%  "
                  f"(min {min(accs) * 100:.2f}, max {max(accs) * 100:.2f}, n = {len(accs)})   "
                  f"events x{ratio:.3f}")
        df = pd.DataFrame(records)                       # save after every run
        if args.append and os.path.exists(out_path):
            old = pd.read_csv(out_path)
            df = pd.concat([old[~old.run_name.isin(df.run_name.unique())], df], ignore_index=True)
        df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
