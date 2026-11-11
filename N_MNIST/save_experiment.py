"""
save_experiment.py

Saving/loading for the N-MNIST w0-variation experiment set. Modeled on
HSD's save.py (save_run -> weights.pt + history.json), extended with:

  - config.json          -- every hyperparameter the run used, so a saved
                             run is fully self-describing later without
                             having to remember which driver script /
                             CLI args produced it.
  - w0 arrays, ONLY when w0 is not a constant shared scalar. A per-neuron
    w0 (fixed-distribution or learnable) is saved as BOTH its initial
    (sampled) and final (post-training) values, so drift under training
    is directly inspectable; a constant scalar w0 is already fully
    captured by config.json's "w0" field, so nothing extra is written
    for it. A layer with no w0 at all (the LIF hidden layer in
    run_lif_hidden.py) is silently skipped, same as the constant case.

Usage from a driver script
---------------------------
    from save_experiment import save_run
    save_run(model, history, config, out_dir=args.out_dir, run_name=run_name)
"""

import json
import os

import torch


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


def save_run(model, history, config, out_dir="results/nmnist_w0_experiments", run_name="run"):
    """
    Saves:
      out_dir/run_name/weights.pt   -- w1, w2, and w0_hidden_initial/final,
                                        w0_output_initial/final IF that
                                        layer's w0 is per-neuron (skipped
                                        entirely for a constant scalar or
                                        a layer with no w0, e.g. LIF)
      out_dir/run_name/history.json -- per-epoch train_loss/train_acc/test_acc
      out_dir/run_name/config.json  -- every hyperparameter this run used

    `history` is the dict returned by nmnist_w0_lib.train_one_run:
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


def list_runs(out_dir="results/nmnist_w0_experiments"):
    """Every run_name saved under out_dir (i.e. every subfolder containing
    a config.json) -- a quick way to see what's there before writing the
    comparison/plotting pass later."""
    if not os.path.isdir(out_dir):
        return []
    return sorted(name for name in os.listdir(out_dir)
                  if os.path.isfile(os.path.join(out_dir, name, "config.json")))
