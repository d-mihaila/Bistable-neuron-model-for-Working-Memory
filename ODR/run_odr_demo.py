"""
run_odr_demo.py

Single-trial sanity-check / demo: does the ODR task's bump form, survive
the delay, and resist a distractor, for a bistable-E ring network at a
given size? No training anywhere -- fixed Mexican-hat connectivity (see
network_architecture.py).

Its DEFAULT_KWARGS are the network settings used by every ODR experiment
(w0_experiments.py). The paper's ODR results are in ODR_w0_gating.ipynb.

Usage:
    python run_odr_demo.py                       # cue at 180 deg, full size
    python run_odr_demo.py --cue_angle 90
    python run_odr_demo.py --n_excit 128 --n_inhib 32   # smaller/faster
    python run_odr_demo.py --distractor 60              # add a distractor trial too
    python run_odr_demo.py --w0 -1.0
"""

import argparse
import os

import torch

import plots
import task
from network_architecture import RingSNN

# ---------------------------------------------------------------------------
# Calibrated connection/noise/stim scales, found by empirical sweeps at the
# full 1024/256 size (see network_architecture.py's module docstring,
# CALIBRATION NOTE section). They are not assumed to be
# size-invariant: the paper uses only the full 1024/256 size.
# ---------------------------------------------------------------------------
DEFAULT_KWARGS = dict(
    g_ee_scale=1.6, g_ei_scale=1.2, g_ie_scale=1.2, g_ii_scale=0.8,
    bg_weight_e=0.0105, bg_weight_i=0.0080, stim_weight_e=0.0105,
)

# Peak external rate (Hz) at the cue/distractor's center angle -- see
# network_architecture.py's module docstring, "INPUT IS NOW SPIKING": the
# cue is a Poisson-generated spike barrage now, not a raw current, so its
# strength is a RATE, not a current amplitude.
DEFAULT_STIM_RATE_HZ = 6000.0


def run_one(cue_angle_deg, n_excit, n_inhib, distractor_angle_deg,
            t_fix, t_cue, t_delay, device, seed, e_kwargs,
            stim_rate_hz=DEFAULT_STIM_RATE_HZ, net_kwargs=None):
    net_kwargs = dict(DEFAULT_KWARGS if net_kwargs is None else net_kwargs)
    net = RingSNN(n_excit=n_excit, n_inhib=n_inhib, e_kwargs=e_kwargs,
                  device=device, seed=seed, **net_kwargs)
    stim_rate, timing = task.build_odr_trial(
        n_excit, net.angles_deg, cue_angle_deg,
        t_fix=t_fix, t_cue=t_cue, t_delay=t_delay,
        stim_rate_hz=stim_rate_hz, distractor_rate_hz=stim_rate_hz,
        distractor_angle_deg=distractor_angle_deg,
        batch_size=1, device=device)
    with torch.no_grad():
        recs = net.run(stim_rate)
    return net, recs, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cue_angle", type=float, default=180.0)
    parser.add_argument("--n_excit", type=int, default=1024)
    parser.add_argument("--n_inhib", type=int, default=256)
    parser.add_argument("--t_fix", type=int, default=100)
    parser.add_argument("--t_cue", type=int, default=250)
    parser.add_argument("--t_delay", type=int, default=1500)
    parser.add_argument("--distractor", type=float, default=None,
                         help="angle (deg) of an optional distractor pulse mid-delay")
    parser.add_argument("--w0", type=float, default=-0.5, help="bistable neuron's w0")
    parser.add_argument("--stim_rate_hz", type=float, default=DEFAULT_STIM_RATE_HZ,
                         help="peak Poisson rate (Hz) driving the cue/distractor -- see module docstring")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/odr_demo")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    net, recs, timing = run_one(
        args.cue_angle, args.n_excit, args.n_inhib, args.distractor,
        args.t_fix, args.t_cue, args.t_delay, args.device, args.seed,
        e_kwargs={"w0": args.w0}, stim_rate_hz=args.stim_rate_hz)

    decoded_at_readout = recs["decoded_final_deg"][0]
    resultant_at_readout = recs["resultant_final"][0]
    err = task.np.abs(((decoded_at_readout - args.cue_angle + 180) % 360) - 180)
    print(f"n_excit={args.n_excit} n_inhib={args.n_inhib} w0={args.w0} "
          f"distractor={args.distractor}")
    print(f"  decoded angle at readout (last {recs['readout_window']} ms, confidence-weighted "
          f"circular mean -- see network_architecture.py's run() for why): "
          f"{decoded_at_readout:.1f} deg (true {args.cue_angle:.0f} deg, error {err:.1f} deg, "
          f"resultant length {resultant_at_readout:.2f})")

    run_name = f"n{args.n_excit}_w0_{args.w0}_cue{args.cue_angle:.0f}"
    fig0 = plots.plot_input_raster(recs, timing, run_name, args.cue_angle)
    fig0.savefig(os.path.join(args.out_dir, f"{run_name}_input.png"), dpi=130, bbox_inches="tight")
    fig1 = plots.plot_odr_raster(recs, timing, run_name, args.cue_angle)
    fig1.savefig(os.path.join(args.out_dir, f"{run_name}_raster.png"), dpi=130, bbox_inches="tight")
    fig2 = plots.plot_bump_profile_snapshots(recs, timing, run_name, args.cue_angle)
    fig2.savefig(os.path.join(args.out_dir, f"{run_name}_bump_profile.png"), dpi=130, bbox_inches="tight")
    print(f"  saved {run_name}_input.png / {run_name}_raster.png / {run_name}_bump_profile.png -> {args.out_dir}")


if __name__ == "__main__":
    main()
