"""
rebuild_network.py

Rebuilds a trained SNNVarW0 model from what save_experiment.py saved for
one run (weights.pt + config.json), so the post-hoc experiment scripts
(exp_waiting.py, exp_noise_weights.py, exp_noise_input.py) can evaluate
already-trained networks without retraining -- same idea as
data_read_save.py's build_net_from_row / build_net_from_mixed_row on the
plain-MNIST side, adapted to this folder's variant-tagged config.json
convention instead of that side's parsed-text-file convention.

Because BistableNeuronVarW0 already accepts w0 as either a plain scalar
or a per-neuron tensor transparently (see nmnist_w0_lib.py), rebuilding
here needs no monkey-patching of Network.step() the way
build_net_from_mixed_row() does on the plain-MNIST side -- we just
construct the neuron with whatever w0 the run actually used.

For the distribution_w0 variant, the neuron is built from the run's
SAVED w0 (w0_hidden_final / w0_output_final in weights.pt) -- the ACTUAL
per-neuron values that run ended training with -- not a fresh resample
from the run's --dist/--mean/--std config. Re-sampling would silently
evaluate a different network than the one that was actually trained.

Usage
-----
    import rebuild_network as rb
    model, history, config = rb.load_trained_model("results/nmnist_w0_experiments", "const_w0_-1.00")

    entries = rb.load_all_trained_models()  # every saved run, rebuilt
"""
import torch

import nmnist_w0_lib as lib
from save_experiment import list_runs, load_run


def run_label(config):
    """Short, human-readable tag for one run -- used as a column/legend
    label everywhere downstream, playing the role plain-MNIST's
    init_type/w0 columns played in data_read_save.py."""
    variant = config.get("variant")
    if variant == "all_lif":
        return f"all_LIF (trained wait {int(config.get('t_wait', 100))} ms)"
    if variant == "const_w0":
        return f"const_w0={float(config['w0']):+.2f}"
    if variant == "lif_hidden_bistable_output":
        return f"lif_hidden_w0={float(config['w0']):+.2f}"
    if variant == "distribution_w0":
        tag = "learnable" if config.get("learnable") else "fixed"
        return f"dist_{config['dist']}_{tag}"
    return config.get("run_name", "unknown")


def load_trained_model(out_dir, run_name, device=None, dtype=torch.float):
    """Rebuilds the SNNVarW0 model that produced this run's saved weights,
    with w1/w2 (and w0, for the per-neuron variants) restored to their
    FINAL trained values.

    Returns (model, history, config).
    """
    device = device or lib.get_device()
    weights, history, config = load_run(out_dir, run_name)
    variant = config.get("variant")

    if variant == "const_w0":
        hidden_neuron, output_neuron = lib.make_const_w0_neurons(
            float(config["w0"]), device, dtype=dtype)

    elif variant == "lif_hidden_bistable_output":
        hidden_neuron, output_neuron = lib.make_lif_hidden_bistable_output_neurons(
            float(config["w0"]), device, dtype=dtype)

    elif variant == "distribution_w0":
        kwargs = lib.bistable_kwargs_common()
        w0_hidden = weights["w0_hidden_final"].to(device=device, dtype=dtype)
        w0_output = weights["w0_output_final"].to(device=device, dtype=dtype)
        hidden_neuron = lib.BistableNeuronVarW0(w0=w0_hidden, device=device, dtype=dtype, **kwargs)
        output_neuron = lib.BistableNeuronVarW0(w0=w0_output, device=device, dtype=dtype, **kwargs)
        # save_experiment.py also saves the PRE-training (sampled) w0 for
        # this variant (w0_hidden_initial/w0_output_initial in
        # weights.pt) -- restore it here too (not just the final value
        # used to rebuild the neuron above) so callers like
        # plots.plot_learnable_w0_histograms can compare initial vs
        # learnt without re-sampling (which would silently not match what
        # this run actually started from).
        if "w0_hidden_initial" in weights:
            hidden_neuron.w0_initial = weights["w0_hidden_initial"].to(device=device, dtype=dtype)
        if "w0_output_initial" in weights:
            output_neuron.w0_initial = weights["w0_output_initial"].to(device=device, dtype=dtype)

    elif variant == "all_lif":
        tau = config.get("lif_tau_mem") or lib.LIF_TAU_MEM
        hidden_neuron, output_neuron = lib.make_all_lif_neurons(device, dtype=dtype, tau_mem=float(tau))

    else:
        raise ValueError(f"Unknown variant '{variant}' in {run_name}/config.json -- "
                          f"don't know how to rebuild this run's neurons.")

    model = lib.build_model(hidden_neuron, output_neuron, device, dtype=dtype)
    model.T_wait = int(config.get("t_wait", model.T_wait))   # all_lif runs may be trained without a wait
    with torch.no_grad():
        model.w1.copy_(weights["w1"].to(device=device, dtype=dtype))
        model.w2.copy_(weights["w2"].to(device=device, dtype=dtype))

    return model, history, config


def load_all_trained_models(out_dir=None, device=None, dtype=torch.float):
    """Every saved run under out_dir, rebuilt. Returns a list of dicts:
        {"run_name":..., "label":..., "model":..., "history":..., "config":...}

    Skips (with a printed warning, without crashing the whole batch) any
    run whose config.json names a variant this file doesn't know how to
    rebuild, or whose weights.pt is missing an expected key -- so one
    malformed run directory doesn't stop you from evaluating the rest.
    """
    out_dir = out_dir or lib.DEFAULT_OUT_DIR
    device = device or lib.get_device()
    entries = []
    for run_name in list_runs(out_dir):
        try:
            model, history, config = load_trained_model(out_dir, run_name, device=device, dtype=dtype)
        except Exception as e:
            print(f"  [skip] {run_name}: {e}")
            continue
        entries.append({"run_name": run_name, "label": run_label(config),
                         "model": model, "history": history, "config": config})
    return entries


if __name__ == "__main__":
    device = lib.get_device()
    print("device:", device)
    entries = load_all_trained_models(device=device)
    print(f"Rebuilt {len(entries)} trained model(s):")
    for e in entries:
        print(f"  {e['run_name']:45s} -> label={e['label']}")
