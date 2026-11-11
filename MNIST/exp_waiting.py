"""
exp_waiting.py

Increasing-wait-time experiment for the MNIST runs, ported directly
from N_MNIST/exp_waiting.py: evaluate ALREADY-TRAINED networks (loaded by
save_experiment.load_all_trained_models, no retraining, no optimizer)
across a range of T_wait values (now in "ms", see mnist_lib.py's module
docstring), to see how far into silence each run's persistent memory
still holds the right answer.

T_wait is stored as a plain attribute on the model instance
(network_architecture.SNN's self.T_wait), so overriding it per-evaluation
is just:
    model.T_wait = t_wait
restored in a try/finally so a later call always starts from the model's
normal (trained) T_wait=100ms, no matter what happens in between.

Usage
-----
    python exp_waiting.py
    python exp_waiting.py --t_wait_values 100 500 1000 2000 5000
    python exp_waiting.py --max_batches 20 --out_dir results/mnist_ms_experiments
"""
import argparse
import os

import pandas as pd
import torch

import mnist_lib as lib
from save_experiment import load_all_trained_models

DEFAULT_T_WAIT_VALUES = (100, 200, 500, 1000, 2000, 5000)
DEFAULT_SAVE_DIR = os.path.join(lib.DEFAULT_OUT_DIR, "post_hoc", "waiting")
DEFAULT_SAVE_CSV = "twait_sweep.csv"


_APPEND = False   # set by --append in main()

def _evaluate_for_t_wait(model, loader, device, dtype, t_wait, max_batches):
    """Runs lib.evaluate() with model.T_wait temporarily overridden,
    restoring it afterwards no matter what."""
    original_t_wait = model.T_wait
    model.T_wait = t_wait
    try:
        acc = lib.evaluate(model, loader, device, dtype, max_batches=max_batches)
    finally:
        model.T_wait = original_t_wait
    return acc


def run_twait_sweep(entries, testloader, device, dtype,
                     t_wait_values=DEFAULT_T_WAIT_VALUES,
                     max_batches=10,
                     save_dir=DEFAULT_SAVE_DIR,
                     save_csv=DEFAULT_SAVE_CSV):
    """entries: the list returned by save_experiment.load_all_trained_models.

    Returns a tidy DataFrame: run_name, label, t_wait, test_acc (t_wait in ms).
    Also writes save_dir/save_csv.
    """
    records = []
    for entry in entries:
        run_name, label, model = entry["run_name"], entry["label"], entry["model"]
        print(f"{'=' * 60}\n  {run_name}  ({label})\n{'=' * 60}")
        for t_wait in t_wait_values:
            acc = _evaluate_for_t_wait(model, testloader, device, dtype, t_wait, max_batches)
            print(f"    T_wait = {t_wait:>6d} ms   test_acc = {acc * 100:6.2f}%")
            records.append({"run_name": run_name, "label": label, "t_wait": t_wait, "test_acc": acc})

    results_df = pd.DataFrame(records)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_csv)
    if _APPEND and os.path.exists(out_path):
        old = pd.read_csv(out_path)
        results_df = pd.concat([old[~old.run_name.isin(results_df.run_name.unique())], results_df],
                               ignore_index=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved waiting-time sweep results -> {out_path}")
    return results_df


def load_twait_sweep(save_dir=DEFAULT_SAVE_DIR, save_csv=DEFAULT_SAVE_CSV):
    """Reload a previously saved waiting-time sweep back into a tidy DataFrame."""
    return pd.read_csv(os.path.join(save_dir, save_csv))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=lib.DEFAULT_OUT_DIR,
                         help="folder the trained runs were saved under (run_*.py's --out_dir)")
    parser.add_argument("--t_wait_values", type=int, nargs="+", default=list(DEFAULT_T_WAIT_VALUES))
    parser.add_argument("--max_batches", type=int, default=10,
                         help="test batches per T_wait per run (keeps this tractable; raise for a fuller estimate)")
    parser.add_argument("--batch_size", type=int, default=lib.DEFAULT_BATCH_SIZE)
    parser.add_argument("--runs", type=str, nargs="*", default=None,
                        help="only evaluate these run names (default: every saved run)")
    parser.add_argument("--append", action="store_true",
                        help="keep the existing CSV and replace only the rows of the runs evaluated now")
    args = parser.parse_args()
    global _APPEND
    _APPEND = args.append

    device = lib.get_device()
    dtype = torch.float
    print("device:", device)

    entries = load_all_trained_models(out_dir=args.out_dir, device=device, dtype=dtype)
    if args.runs:
        entries = [en for en in entries if en["run_name"] in args.runs]
    print(f"Rebuilt {len(entries)} trained model(s) from {args.out_dir}")
    if not entries:
        print("Nothing to run -- no saved runs found.")
        return

    _, testloader = lib.get_dataloaders(batch_size=args.batch_size)
    run_twait_sweep(entries, testloader, device, dtype,
                     t_wait_values=tuple(args.t_wait_values), max_batches=args.max_batches,
                     save_dir=os.path.join(args.out_dir, "post_hoc", "waiting"))


if __name__ == "__main__":
    main()
