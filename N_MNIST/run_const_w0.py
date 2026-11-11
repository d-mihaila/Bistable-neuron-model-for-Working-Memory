"""
run_const_w0.py

The 6-value constant-w0 sweep, one cluster job per value: every neuron in
both layers shares the SAME scalar w0 (not learnable, not per-neuron) --
same setup as network_previous.py's own run_sweep(w0_values), just N-MNIST
instead of Poisson-encoded MNIST, and with this experiment set's
MNIST-matched timescales (see nmnist_w0_lib.py's module docstring).

Usage (one job per w0 value -- this is what makes these 6 runs
independent cluster jobs instead of one long sequential sweep):
    python run_const_w0.py --w0 -2
    python run_const_w0.py --w0 -1
    python run_const_w0.py --w0 -0.5
    python run_const_w0.py --w0 0.5
    python run_const_w0.py --w0 1
    python run_const_w0.py --w0 2
"""

import argparse

import torch

import nmnist_w0_lib as lib
from save_experiment import save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w0", type=float, required=True, help="shared scalar w0 for every neuron")
    lib.add_common_args(parser)
    args = parser.parse_args()

    lib.set_seed(args.seed)
    lib.print_timescale_check()
    device = lib.get_device()
    dtype = torch.float
    print("device:", device)

    trainloader, testloader = lib.get_dataloaders(batch_size=args.batch_size)

    hidden_neuron, output_neuron = lib.make_const_w0_neurons(args.w0, device, dtype=dtype)
    model = lib.build_model(hidden_neuron, output_neuron, device, dtype=dtype)

    run_name = f"const_w0_{args.w0:+.2f}"
    print(f"=== {run_name} ===")

    history = lib.train_one_run(
        model, trainloader, testloader, run_name, device, dtype,
        lr=args.lr, nb_epochs=args.nb_epochs, max_batches_per_epoch=args.max_batches_per_epoch,
        print_every=args.print_every, eval_max_batches=args.eval_max_batches,
    )

    config = dict(vars(args), variant="const_w0", run_name=run_name,
                  num_steps=lib.NUM_STEPS, t_wait=lib.T_WAIT, t_eval=lib.T_EVAL,
                  n_in=lib.N_IN, n_hidden=lib.N_HIDDEN, n_out=lib.N_OUT,
                  v_th=lib.V_TH, eps=lib.EPS, a=lib.A, d=lib.D, c=lib.C,
                  tau=lib.TAU, dt_eff=lib.DT_EFF)
    save_run(model, history, config, out_dir=args.out_dir, run_name=run_name)


if __name__ == "__main__":
    main()
