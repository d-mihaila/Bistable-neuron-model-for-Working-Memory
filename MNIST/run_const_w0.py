"""
run_const_w0.py

Plain-MNIST, constant scalar w0, biologically-timed rebuild -- the direct
analog of N_MNIST/run_const_w0.py, using this folder's mnist_lib.py
instead of nmnist_w0_lib.py. Same neuron model, same network architecture
(reused byte-for-byte from N_MNIST), same NUM_STEPS/T_WAIT/T_EVAL/dt_eff,
epoch-based training -- only the dataset and input encoding differ (see
mnist_lib.encode_batch's docstring).

Usage (one job per w0 value, matching N_MNIST's own 6-value sweep):
    python run_const_w0.py --w0 -2
    python run_const_w0.py --w0 -1
    python run_const_w0.py --w0 -0.5
    python run_const_w0.py --w0 0.5
    python run_const_w0.py --w0 1
    python run_const_w0.py --w0 2
"""

import argparse

import torch

import mnist_lib as lib
from save_experiment import save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w0", type=float, default=lib.W0_DEFAULT, help="shared scalar w0 for every neuron")
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
