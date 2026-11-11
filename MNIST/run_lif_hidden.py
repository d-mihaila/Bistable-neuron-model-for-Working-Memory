"""
run_lif_hidden.py

Hidden layer: LIF (LIF_simple -- current-based, no synaptic filter, no w0
at all). Output layer: plain BistableNeuronVarW0 with a constant scalar
w0. Isolates whether the bistable, persistent-memory-holding dynamics
matter specifically at the readout layer, decoupled from the hidden
layer -- same split as network_with_lif.py's own Network class on plain
MNIST (LIF hidden / bistable output), ported to this folder's
biologically-timed conventions. Direct analog of
N_MNIST/run_lif_hidden.py, using this folder's mnist_lib.py instead of
nmnist_w0_lib.py.

--w0 defaults to -1 (this experiment set's baseline elsewhere); override
it if you want the output layer at a different point in the const-w0
sweep instead.

See also run_lif_output.py -- the reverse split (bistable hidden / LIF
output), a new addition not present in N_MNIST, added to isolate the same
question at the hidden layer instead.

Usage:
    python run_lif_hidden.py
    python run_lif_hidden.py --w0 -1
"""

import argparse

import numpy as np
import torch

import mnist_lib as lib
from save_experiment import save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w0", type=float, default=-1.0, help="scalar w0 for the bistable OUTPUT layer only")
    lib.add_common_args(parser)
    args = parser.parse_args()

    lib.set_seed(args.seed)
    lib.print_timescale_check()
    device = lib.get_device()
    dtype = torch.float
    print("device:", device)
    print(f"LIF hidden layer: tau_mem={lib.LIF_TAU_MEM:.5f}  time_step={lib.DT_EFF:.5f}  "
          f"-> beta={float(np.exp(-lib.DT_EFF / lib.LIF_TAU_MEM)):.4f} "
          f"(matches params.py's lif_beta=0.5 used for the equivalent plain-MNIST comparison)")

    trainloader, testloader = lib.get_dataloaders(batch_size=args.batch_size)

    hidden_neuron, output_neuron = lib.make_lif_hidden_bistable_output_neurons(args.w0, device, dtype=dtype)
    model = lib.build_model(hidden_neuron, output_neuron, device, dtype=dtype)

    run_name = f"lif_hidden_bistable_output_w0_{args.w0:+.2f}"
    print(f"=== {run_name} ===")

    history = lib.train_one_run(
        model, trainloader, testloader, run_name, device, dtype,
        lr=args.lr, nb_epochs=args.nb_epochs, max_batches_per_epoch=args.max_batches_per_epoch,
        print_every=args.print_every, eval_max_batches=args.eval_max_batches,
    )

    config = dict(vars(args), variant="lif_hidden_bistable_output", run_name=run_name,
                  num_steps=lib.NUM_STEPS, t_wait=lib.T_WAIT, t_eval=lib.T_EVAL,
                  n_in=lib.N_IN, n_hidden=lib.N_HIDDEN, n_out=lib.N_OUT,
                  v_th=lib.V_TH, eps=lib.EPS, a=lib.A, d=lib.D, c=lib.C,
                  tau=lib.TAU, dt_eff=lib.DT_EFF,
                  lif_tau_mem=lib.LIF_TAU_MEM, lif_threshold=lib.LIF_THRESHOLD)
    save_run(model, history, config, out_dir=args.out_dir, run_name=run_name)


if __name__ == "__main__":
    main()
