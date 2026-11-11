"""
run_distribution_w0.py

Per-neuron w0 (instead of one shared scalar), sampled from a distribution
around a target mean -- covers all 5 of the requested distribution
variants via two orthogonal flags, --dist and --learnable:

    fixed,  normal      (not learnable):
        python run_distribution_w0.py --dist normal --mean -1.25 --std 0.5
    fixed,  uniform      (not learnable):
        python run_distribution_w0.py --dist uniform --mean -1.25 --std 0.5
    learnable, normal_1 init:
        python run_distribution_w0.py --dist normal_1 --mean -1.25 --std 0.5 --learnable
    learnable, half +1.25 / half -1.25 init ("two_tailed"):
        python run_distribution_w0.py --dist two_tailed --mean -1.25 --std 0.5 --learnable
    learnable, normal_0 init:
        python run_distribution_w0.py --dist normal_0 --mean 0.0 --std 0.5 --learnable

('two_tailed' works with --learnable omitted too, i.e. fixed after
sampling -- not one of the 5 requested runs, but available if useful.)

--mean's SIGN doesn't matter for 'two_tailed' (only |mean| is used, as
the magnitude of both peaks) -- pass -1.25 or 1.25, same result.

'normal_0' and 'normal_1' are named aliases of the same underlying
'normal' sampling in nmnist_w0_lib.py (which only knows 'normal',
'uniform', and 'two_tailed') -- they exist purely so run_name/config
record which of the two learnable-normal-init runs this is, and
default --mean to 0.0 (normal_0) vs -1.25 (normal_1, same default as
plain normal/uniform/two_tailed) when --mean isn't passed explicitly.

See nmnist_w0_lib.py's module docstring for sample_w0's exact definitions
(ported from netowork_mix_w0.py) and for why a per-neuron/learnable w0
needs BistableNeuronVarW0 rather than the plain BistableNeuron.
"""

import argparse

import torch

import nmnist_w0_lib as lib
from save_experiment import save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dist", choices=["normal", "uniform", "two_tailed", "normal_0", "normal_1"],
                         required=True,
                         help="'normal_0'/'normal_1' are aliases for 'normal' "
                              "(see module docstring) -- kept as distinct choices so "
                              "run_name/config record which learnable-normal-init run this is.")
    parser.add_argument("--mean", type=float, default=None,
                         help="target mean w0 (or peak magnitude for two_tailed); "
                              "defaults to 0.0 for --dist normal_0, else -1.25")
    parser.add_argument("--std", type=float, default=0.5, help="target std of the sampled w0; default 0.5")
    parser.add_argument("--learnable", action="store_true",
                         help="if set, w0 is trained jointly with w1/w2 instead of staying fixed after sampling")
    lib.add_common_args(parser)
    args = parser.parse_args()

    if args.mean is None:
        args.mean = 0.0 if args.dist == "normal_0" else -1.25

    # nmnist_w0_lib.sample_w0 only knows 'normal'/'uniform'/'two_tailed' --
    # map the two aliases down to 'normal' for the actual sampling call
    # while keeping args.dist as-is for run_name/config/logging below.
    sample_dist = "normal" if args.dist in ("normal_0", "normal_1") else args.dist

    if args.dist in ("normal_0", "normal_1") and not args.learnable:
        print(f"warning: --dist {args.dist} is intended for the learnable case; "
              f"without --learnable this is identical to --dist normal --mean {args.mean}")

    lib.set_seed(args.seed)
    lib.print_timescale_check()
    device = lib.get_device()
    dtype = torch.float
    print("device:", device)

    trainloader, testloader = lib.get_dataloaders(batch_size=args.batch_size)

    hidden_neuron, output_neuron = lib.make_distribution_neurons(
        sample_dist, args.mean, args.std, args.learnable, device, dtype=dtype)
    model = lib.build_model(hidden_neuron, output_neuron, device, dtype=dtype)

    learn_tag = "learnable" if args.learnable else "fixed"
    run_name = f"dist_{args.dist}_mean{args.mean:+.2f}_std{args.std:.2f}_{learn_tag}"
    print(f"=== {run_name} ===")
    print(f"w0_hidden initial: mean={hidden_neuron.w0_initial.mean().item():.4f}  "
          f"std={hidden_neuron.w0_initial.std().item():.4f}")
    print(f"w0_output initial: mean={output_neuron.w0_initial.mean().item():.4f}  "
          f"std={output_neuron.w0_initial.std().item():.4f}")

    history = lib.train_one_run(
        model, trainloader, testloader, run_name, device, dtype,
        lr=args.lr, nb_epochs=args.nb_epochs, max_batches_per_epoch=args.max_batches_per_epoch,
        print_every=args.print_every, eval_max_batches=args.eval_max_batches,
    )

    config = dict(vars(args), variant="distribution_w0", run_name=run_name,
                  num_steps=lib.NUM_STEPS, t_wait=lib.T_WAIT, t_eval=lib.T_EVAL,
                  n_in=lib.N_IN, n_hidden=lib.N_HIDDEN, n_out=lib.N_OUT,
                  v_th=lib.V_TH, eps=lib.EPS, a=lib.A, d=lib.D, c=lib.C,
                  tau=lib.TAU, dt_eff=lib.DT_EFF)
    save_run(model, history, config, out_dir=args.out_dir, run_name=run_name)


if __name__ == "__main__":
    main()
