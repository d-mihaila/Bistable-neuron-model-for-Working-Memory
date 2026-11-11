"""
exp_delay_activity.py

Sustained-activity and efficiency analysis over a LONG silent delay, for
every trained run in --out_dir (no retraining). Same folder convention as
exp_waiting.py / exp_noise_*.py: results go to
<out_dir>/post_hoc/delay_activity/delay_activity.csv.

Each test trial is run ONCE with a long silence after the stimulus
(T_wait = --t_wait, default 4900 -> 5 s in total). The run is cut into
100 ms bins (bin 0 = the stimulus itself, bins 1.. = the delay). Per run and
per bin it reports:

    delay_ms                 end of the bin in ms after the input stops (0 = the stimulus bin);
                             the same number exp_waiting.py calls t_wait
    acc                      readout accuracy if the class were read from THIS bin's
                             output spike counts (argmax, same rule as training/eval)
    frac_output_silent       fraction of trials with no output spike at all in the bin
    frac_correct_only        fraction of trials where ONLY the correct output neuron fires
    hidden_spikes, output_spikes, input_spikes
                             mean spikes per trial in the bin (whole layer)
    frac_hidden_active       mean fraction of hidden neurons with >= 1 spike in the bin
    synops                   synaptic events per trial in the bin:
                             input spikes x n_hidden + hidden spikes x n_out

Reading the class from bin k is the same quantity exp_waiting.py measures for
T_wait = 100*k (its readout is the last 100 ms of the wait), so the two agree
up to sampling noise -- but here one long run per trial gives the whole time
course, plus the spike counts needed for an efficiency comparison.

Usage:
    python exp_delay_activity.py                      # all runs
    python exp_delay_activity.py --runs all_lif_wait100 all_lif_wait0 --append
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import nmnist_w0_lib as lib
from rebuild_network import load_all_trained_models

BIN = 100


def encode(x, device, dtype):
    return lib.flatten_batch(x, device, dtype)


def run_one(entry, testloader, device, dtype, t_wait, max_batches, seed):
    model = entry["model"]
    old_wait = model.T_wait
    model.T_wait = t_wait
    torch.manual_seed(seed)
    n_bins = (lib.NUM_STEPS + t_wait) // BIN
    acc = np.zeros(n_bins); silent = np.zeros(n_bins); only = np.zeros(n_bins)
    hid = np.zeros(n_bins); out = np.zeros(n_bins); inp = np.zeros(n_bins); act = np.zeros(n_bins)
    n = 0
    try:
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                if i >= max_batches:
                    break
                x = encode(x, device, dtype)
                y = y.to(device)
                _, recs = model.forward(x)
                B = y.shape[0]
                T = recs["spk"].shape[1]
                hs = recs["spk"][:, :n_bins * BIN].reshape(B, n_bins, BIN, -1).sum(2)       # (B, bins, n_hidden)
                os_ = recs["out_spk"][:, :n_bins * BIN].reshape(B, n_bins, BIN, -1).sum(2)  # (B, bins, n_out)
                pred = os_.argmax(-1)
                acc += (pred == y[:, None]).float().sum(0).cpu().numpy()
                tot = os_.sum(-1)
                silent += (tot == 0).float().sum(0).cpu().numpy()
                corr = os_.gather(-1, y[:, None, None].expand(-1, n_bins, 1)).squeeze(-1)
                only += ((corr > 0) & (tot == corr)).float().sum(0).cpu().numpy()
                hid += hs.sum(-1).sum(0).cpu().numpy()
                out += tot.sum(0).cpu().numpy()
                act += (hs > 0).float().mean(-1).sum(0).cpu().numpy()
                xin = torch.zeros(B, n_bins, device=device)
                xin[:, :lib.NUM_STEPS // BIN] = x.reshape(B, lib.NUM_STEPS // BIN, BIN, -1).sum((2, 3))
                inp += xin.sum(0).cpu().numpy()
                n += B
                del recs
    finally:
        model.T_wait = old_wait
    rows = []
    for k in range(n_bins):
        rows.append(dict(run_name=entry["run_name"], label=entry["label"],
                         variant=entry["config"].get("variant"),
                         bin_start_ms=k * BIN, delay_ms=k * BIN,
                         acc=acc[k] / n, frac_output_silent=silent[k] / n, frac_correct_only=only[k] / n,
                         hidden_spikes=hid[k] / n, output_spikes=out[k] / n, input_spikes=inp[k] / n,
                         frac_hidden_active=act[k] / n,
                         synops=(inp[k] * lib.N_HIDDEN + hid[k] * lib.N_OUT) / n,
                         n_trials=n))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=str, default=lib.DEFAULT_OUT_DIR)
    p.add_argument("--t_wait", type=int, default=4900)
    p.add_argument("--max_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=str, nargs="*", default=None, help="only these run names (default: all)")
    p.add_argument("--append", action="store_true", help="replace only these runs' rows in an existing CSV")
    args = p.parse_args()

    device = torch.device("cpu") if not torch.cuda.is_available() else lib.get_device()
    dtype = torch.float
    torch.manual_seed(args.seed)
    _, testloader = lib.get_dataloaders(batch_size=args.batch_size)
    entries = load_all_trained_models(out_dir=args.out_dir, device=device, dtype=dtype)
    if args.runs:
        entries = [e for e in entries if e["run_name"] in args.runs]
    out_path = os.path.join(args.out_dir, "post_hoc", "delay_activity", "delay_activity.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for e in entries:
        r = run_one(e, testloader, device, dtype, args.t_wait, args.max_batches, args.seed)
        print(f"{e['run_name']:48s} acc stim {r[0]['acc']:.2f} | +100 ms {r[1]['acc']:.2f} | "
              f"+1 s {r[min(10, len(r)-1)]['acc']:.2f} | +{(len(r)-1)*BIN/1000:.1f} s {r[-1]['acc']:.2f} | "
              f"hidden spikes/100ms in delay {np.mean([x['hidden_spikes'] for x in r[1:]]):.1f}", flush=True)
        rows += r
    df = pd.DataFrame(rows)
    if args.append and os.path.exists(out_path):
        old = pd.read_csv(out_path)
        df = pd.concat([old[~old.run_name.isin(df.run_name.unique())], df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print("saved", out_path, len(df), "rows")


if __name__ == "__main__":
    main()
