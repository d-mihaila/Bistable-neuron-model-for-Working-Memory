"""
save_experiment.py

Saving, loading, and reconstructing trained runs for the MNIST
experiment set. Ported from N_MNIST/save_experiment.py, with one
structural difference: N_MNIST splits this into two files
(save_experiment.py for save_run/load_run/list_runs, a separate
rebuild_network.py for load_trained_model/load_all_trained_models) --
here everything lives in this one file, by request.

  - save_run(model, history, config, ...) -- writes weights.pt,
    history.json, config.json for one run (unchanged from N_MNIST).
  - load_run(...) / list_runs(...) -- read those back (unchanged).
  - load_trained_model(...) / load_all_trained_models(...) -- ported
    from N_MNIST/rebuild_network.py: rebuilds a LIVE, ready-to-evaluate
    model from a saved run, with w1/w2 (and w0, for the per-neuron
    variants) restored to their FINAL trained values -- extended to
    cover all four variants this folder trains (const_w0,
    distribution_w0, lif_hidden_bistable_output, and
    bistable_hidden_lif_output -- the last one is new, not in N_MNIST,
    since run_lif_output.py has no N_MNIST equivalent to rebuild from).

config.json fields
------------------
  - w0 arrays, ONLY when w0 is not a constant shared scalar. A per-neuron
    w0 (fixed-distribution or learnable) is saved as BOTH its initial
    (sampled) and final (post-training) values, so drift under training
    is directly inspectable; a constant scalar w0 is already fully
    captured by config.json's "w0" field, so nothing extra is written
    for it. A layer with no w0 at all (an LIF layer) is silently
    skipped, same as the constant case.

Usage from a driver script
---------------------------
    from save_experiment import save_run
    save_run(model, history, config, out_dir=args.out_dir, run_name=run_name)

Usage from a post-hoc experiment script (exp_waiting.py etc.)
---------------------------------------------------------------
    from save_experiment import load_all_trained_models
    entries = load_all_trained_models(out_dir="results/mnist_ms_experiments")
    # entries: [{"run_name":..., "label":..., "model":..., "history":..., "config":...}, ...]
"""

import json
import os

import torch

import mnist_lib as lib


# ---------------------------------------------------------------------------
# Save / load raw run artifacts -- unchanged from N_MNIST's save_experiment.py
# ---------------------------------------------------------------------------

def _w0_arrays(neuron):
    """(initial, final) numpy arrays for a per-neuron w0, or (None, None)
    if this neuron's w0 is a plain scalar (nothing to save beyond what's
    already in config.json) or doesn't exist at all (e.g. LIF_simple has
    no w0)."""
    w0 = getattr(neuron, "w0", None)
    if isinstance(w0, torch.Tensor):
        final = w0.detach().cpu().numpy()
        initial_t = getattr(neuron, "w0_initial", w0)
        initial = initial_t.detach().cpu().numpy() if isinstance(initial_t, torch.Tensor) else final
        return initial, final
    return None, None


def save_run(model, history, config, out_dir="results/mnist_ms_experiments", run_name="run"):
    """
    Saves:
      out_dir/run_name/weights.pt   -- w1, w2, and w0_hidden_initial/final,
                                        w0_output_initial/final IF that
                                        layer's w0 is per-neuron (skipped
                                        entirely for a constant scalar or
                                        a layer with no w0, e.g. LIF)
      out_dir/run_name/history.json -- per-epoch train_loss/train_acc/test_acc
      out_dir/run_name/config.json  -- every hyperparameter this run used

    `history` is the dict returned by mnist_lib.train_one_run:
      {"train_loss": [...], "train_acc": [...], "test_acc": [...]}
    `config` is any JSON-serializable dict of hyperparameters you want
    preserved alongside the run (w0 value/dist/mean/std, nb_epochs, lr,
    seed, learnable flag, etc.) -- build it in the driver script from
    argparse's `vars(args)` plus whatever else is worth recording.
    """
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    weights = {"w1": model.w1.detach().cpu(), "w2": model.w2.detach().cpu()}

    w0_hidden_initial, w0_hidden_final = _w0_arrays(model.hidden_neuron)
    if w0_hidden_final is not None:
        weights["w0_hidden_initial"] = torch.as_tensor(w0_hidden_initial)
        weights["w0_hidden_final"] = torch.as_tensor(w0_hidden_final)

    w0_output_initial, w0_output_final = _w0_arrays(model.output_neuron)
    if w0_output_final is not None:
        weights["w0_output_initial"] = torch.as_tensor(w0_output_initial)
        weights["w0_output_final"] = torch.as_tensor(w0_output_final)

    torch.save(weights, os.path.join(run_dir, "weights.pt"))

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    saved_extra = [k for k in weights if k not in ("w1", "w2")]
    print(f"Saved weights ({'+ ' + ', '.join(saved_extra) if saved_extra else 'w1, w2 only'}) "
          f"+ history + config -> {run_dir}")


def load_run(out_dir, run_name, map_location="cpu"):
    """Loads back everything save_run wrote for one run.

    Returns (weights_dict, history_dict, config_dict). weights_dict has
    whichever keys were saved (always w1/w2; w0_*_initial/final only if
    that layer had a per-neuron w0) -- check with `"w0_hidden_final" in
    weights` rather than assuming.
    """
    run_dir = os.path.join(out_dir, run_name)
    weights = torch.load(os.path.join(run_dir, "weights.pt"), map_location=map_location)
    with open(os.path.join(run_dir, "history.json")) as f:
        history = json.load(f)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    return weights, history, config


def list_runs(out_dir="results/mnist_ms_experiments"):
    """Every run_name saved under out_dir (i.e. every subfolder containing
    a config.json) -- a quick way to see what's there before writing the
    comparison/plotting pass later."""
    if not os.path.isdir(out_dir):
        return []
    return sorted(name for name in os.listdir(out_dir)
                  if os.path.isfile(os.path.join(out_dir, name, "config.json")))


# ---------------------------------------------------------------------------
# Reconstruct a live, trained model from what save_run wrote -- ported from
# N_MNIST/rebuild_network.py, extended with the bistable_hidden_lif_output
# variant (new in this folder, no N_MNIST equivalent).
# ---------------------------------------------------------------------------

def run_label(config):
    """Short, human-readable tag for one run -- used as a column/legend
    label everywhere downstream. Ported from N_MNIST/rebuild_network.py's
    run_label(), extended with the new reversed-LIF variant."""
    variant = config.get("variant")
    if variant == "all_lif":
        return f"all_LIF (trained wait {int(config.get('t_wait', 100))} ms)"
    if variant == "const_w0":
        return f"const_w0={float(config['w0']):+.2f}"
    if variant == "lif_hidden_bistable_output":
        return f"lif_hidden_w0={float(config['w0']):+.2f}"
    if variant == "bistable_hidden_lif_output":
        return f"lif_output_w0={float(config['w0']):+.2f}"
    if variant == "distribution_w0":
        tag = "learnable" if config.get("learnable") else "fixed"
        return f"dist_{config['dist']}_{tag}"
    return config.get("run_name", "unknown")


def load_trained_model(out_dir, run_name, device=None, dtype=torch.float):
    """Rebuilds the SNNVarW0 model that produced this run's saved weights,
    with w1/w2 (and w0, for the per-neuron/LIF variants) restored to
    their FINAL trained values -- which network TYPE to rebuild (LIF or
    bistable in which layer, and what w0) is read straight from this
    run's own config.json["variant"], so each run is reconstructed
    exactly the way it was originally created, not guessed at.

    For the distribution_w0 variant, the neuron is built from the run's
    SAVED w0 (w0_hidden_final / w0_output_final in weights.pt) -- the
    ACTUAL per-neuron values that run ended training with -- not a fresh
    resample from the run's --dist/--mean/--std config. Re-sampling would
    silently evaluate a different network than the one that was actually
    trained. Ported from N_MNIST/rebuild_network.py's load_trained_model().

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

    elif variant == "bistable_hidden_lif_output":
        hidden_neuron, output_neuron = lib.make_bistable_hidden_lif_output_neurons(
            float(config["w0"]), device, dtype=dtype)

    elif variant == "distribution_w0":
        kwargs = lib.bistable_kwargs_common()
        w0_hidden = weights["w0_hidden_final"].to(device=device, dtype=dtype)
        w0_output = weights["w0_output_final"].to(device=device, dtype=dtype)
        hidden_neuron = lib.BistableNeuronVarW0(w0=w0_hidden, device=device, dtype=dtype, **kwargs)
        output_neuron = lib.BistableNeuronVarW0(w0=w0_output, device=device, dtype=dtype, **kwargs)
        # also restore the PRE-training (sampled) w0 (w0_hidden_initial /
        # w0_output_initial in weights.pt), not just the final value used
        # to rebuild the neuron above, so callers comparing initial vs
        # learnt don't have to re-sample (which would silently not match
        # what this run actually started from).
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
    Ported from N_MNIST/rebuild_network.py's load_all_trained_models().
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
