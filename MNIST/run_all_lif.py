"""
run_all_lif.py

NEW -- the no-memory baseline: BOTH layers LIF_simple (no w0, no bistable
dynamics anywhere), same LIF parameters as the LIF layer in
run_lif_hidden.py (beta = 0.5 per step by default).

Trained two ways, to separate "can it learn the task?" from "can it hold
the answer through silence?":

    python run_all_lif.py --t_wait 100   # same protocol as every other run:
                                         # the class is read from spike counts
                                         # during 100 ms of silence after the input
    python run_all_lif.py --t_wait 0     # no wait: the class is read during the
                                         # input itself (shows LIF learns the task)

exp_waiting.py then evaluates both at longer waits, like every other run.
--tau_mem overrides the LIF membrane time constant (default: lib.LIF_TAU_MEM).
"""

import argparse

import numpy as np
import torch

import mnist_lib as lib
from save_experiment import save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t_wait", type=int, default=lib.T_WAIT,
                        help="silent ms after the stimulus during TRAINING (0 = read out during the stimulus)")
    parser.add_argument("--tau_mem", type=float, default=None, help="LIF membrane time constant (default lib.LIF_TAU_MEM)")
    lib.add_common_args(parser)
    args = parser.parse_args()

    lib.set_seed(args.seed)
    device = lib.get_device()
    dtype = torch.float
    tau = lib.LIF_TAU_MEM if args.tau_mem is None else args.tau_mem
    print("device:", device)
    print(f"all-LIF: tau_mem={tau:.5f}  time_step={lib.DT_EFF:.5f}  -> beta={float(np.exp(-lib.DT_EFF / tau)):.4f}")

    trainloader, testloader = lib.get_dataloaders(batch_size=args.batch_size)

    hidden_neuron, output_neuron = lib.make_all_lif_neurons(device, dtype=dtype, tau_mem=tau)
    model = lib.build_model(hidden_neuron, output_neuron, device, dtype=dtype)
    model.T_wait = args.t_wait

    run_name = f"all_lif_wait{args.t_wait}" + ("" if args.tau_mem is None else f"_tau{args.tau_mem:g}")
    print(f"=== {run_name} ===")

    history = lib.train_one_run(
        model, trainloader, testloader, run_name, device, dtype,
        lr=args.lr, nb_epochs=args.nb_epochs, max_batches_per_epoch=args.max_batches_per_epoch,
        print_every=args.print_every, eval_max_batches=args.eval_max_batches,
    )

    config = dict(vars(args), variant="all_lif", run_name=run_name, w0=None,
                  num_steps=lib.NUM_STEPS, t_wait=args.t_wait, t_eval=lib.T_EVAL,
                  n_in=lib.N_IN, n_hidden=lib.N_HIDDEN, n_out=lib.N_OUT,
                  v_th=lib.V_TH, eps=lib.EPS, a=lib.A, d=lib.D, c=lib.C,
                  tau=lib.TAU, dt_eff=lib.DT_EFF,
                  lif_tau_mem=tau, lif_threshold=lib.LIF_THRESHOLD)
    save_run(model, history, config, out_dir=args.out_dir, run_name=run_name)


if __name__ == "__main__":
    main()
