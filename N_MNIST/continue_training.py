"""
continue_training.py

Resumes training for every ALREADY-SAVED run under
results/nmnist_w0_experiments/ whose w0 is NOT being jointly learned --
i.e. everything except the 4 "..._learnable" distribution_w0 runs -- from
its saved epoch-25 weights.pt up to --target_nb_epochs (default 50), and
overwrites that run's weights.pt / history.json / config.json in place
with the extended result.

Why the learnable runs are skipped by default
-----------------------------------------------
rebuild_network.load_trained_model() (used below to rebuild each network)
restores w0_hidden_final / w0_output_final as plain tensors, not as
torch.nn.Parameter(requires_grad=True). For the FIXED distribution runs
that's exactly correct -- w0 was never supposed to be trainable there
anyway. For the LEARNABLE runs it's a real bug: SNNVarW0.parameters()
only picks up w0 when w0.requires_grad is True, so resuming one of those
naively would silently freeze w0 at its epoch-25 value while w1/w2 kept
training -- a different (wrong) model, no error raised.

That IS fixable (see _make_w0_learnable_again below) and this script can
do it -- pass --include_learnable to resume those 4 too instead of
training them from scratch. Off by default: the learnable runs of the
paper were trained from scratch for 50 epochs.

What does NOT get restored: the Adam optimizer's own momentum/variance
state (never saved by save_experiment.py in the first place). Resuming
without it is standard practice -- train_one_run() below builds a fresh
Adam optimizer each time this script runs, which just means a short,
mild re-warm-up over the first part of the resumed epochs.

Safety: before overwriting a run's saved files, the CURRENT (pre-resume)
weights.pt/history.json/config.json are copied to
    {out_dir}/_pre_resume_backups/{run_name}_epoch{N}/
so the original epoch-25 checkpoint isn't lost if something about the
resumed run looks wrong later. Pass --no_backup to skip this.

Resumable: a run whose saved config.json already has nb_epochs >=
--target_nb_epochs is skipped (with a message), so re-running this
script after a partial batch (or a crash partway through) only continues
runs that still need it -- same idea as this codebase's other scripts'
manifest-skip pattern, just driven by config.json's own nb_epochs
instead of a separate manifest file.

Usage
-----
    python continue_training.py                        # all fixed-w0 runs, 25 -> 50
    python continue_training.py --target_nb_epochs 40   # 25 -> 40 instead
    python continue_training.py --include_learnable     # also resume the 4 learnable runs
    python continue_training.py --run_name const_w0_-1.00   # just one run, for testing
    python continue_training.py --max_batches_per_epoch 2 --target_nb_epochs 26  # smoke test
"""

import argparse
import json
import os
import shutil
import time

import torch

import nmnist_w0_lib as lib
import rebuild_network as rb
from save_experiment import list_runs, save_run

# variant is always fine to resume: w0 is either a constant scalar (already
# fully captured by config.json, nothing to re-learn) or doesn't exist at
# all (LIF hidden layer) -- there's no learnable-w0 pitfall for either.
ALWAYS_RESUMABLE_VARIANTS = {"const_w0", "lif_hidden_bistable_output",
                             "all_lif"}   # all-LIF: no w0 at all, nothing learnable to lose


def _make_w0_learnable_again(model):
    """Re-wraps hidden_neuron.w0 / output_neuron.w0 as
    torch.nn.Parameter(requires_grad=True), in place, for a model rebuilt
    by rebuild_network.load_trained_model() -- undoes exactly the bug
    described in this file's module docstring. Mirrors what
    nmnist_w0_lib.make_distribution_neurons does for a fresh learnable run.
    No-op for any neuron whose w0 isn't a tensor (nothing to do)."""
    for neuron in (model.hidden_neuron, model.output_neuron):
        w0 = getattr(neuron, "w0", None)
        if isinstance(w0, torch.Tensor) and not w0.requires_grad:
            neuron.w0 = torch.nn.Parameter(w0.detach().clone())


def _select_runs(out_dir, include_learnable, only_run_name=None):
    """Every run_name under out_dir whose config.json says it's safe (or,
    with --include_learnable, has been made safe) to resume, paired with
    its parsed config. Prints (and skips) anything filtered out, so it's
    obvious from the console output which runs were and weren't touched."""
    selected = []
    for run_name in list_runs(out_dir):
        if only_run_name is not None and run_name != only_run_name:
            continue

        config_path = os.path.join(out_dir, run_name, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        variant = config.get("variant")

        if variant in ALWAYS_RESUMABLE_VARIANTS:
            selected.append((run_name, config))
        elif variant == "distribution_w0":
            if config.get("learnable"):
                if include_learnable:
                    selected.append((run_name, config))
                else:
                    print(f"  [skip] {run_name}: learnable distribution_w0 -- "
                          f"pass --include_learnable to resume it too")
            else:
                selected.append((run_name, config))
        else:
            print(f"  [skip] {run_name}: unrecognised variant {variant!r}")

    return selected


def _backup_run(out_dir, run_name, already_trained_epochs):
    """Copies the run's CURRENT saved files to
    {out_dir}/_pre_resume_backups/{run_name}_epoch{N}/ before they get
    overwritten. Skipped (with a message, not an error) if that exact
    backup already exists -- e.g. this run was already resumed once
    before and is now being extended further."""
    src = os.path.join(out_dir, run_name)
    dst = os.path.join(out_dir, "_pre_resume_backups", f"{run_name}_epoch{already_trained_epochs}")
    if os.path.exists(dst):
        print(f"  [backup] {dst} already exists, leaving it as-is")
        return
    shutil.copytree(src, dst)
    print(f"  [backup] {src} -> {dst}")


def resume_one_run(out_dir, run_name, config, target_nb_epochs, device, dtype,
                    include_learnable, make_backup, lr_override,
                    max_batches_per_epoch_override, print_every, eval_max_batches):
    already_trained = int(config.get("nb_epochs", 0))
    if already_trained >= target_nb_epochs:
        print(f"[{run_name}] already at {already_trained} epochs "
              f">= target {target_nb_epochs} -- skipping")
        return

    additional_epochs = target_nb_epochs - already_trained
    print(f"\n=== {run_name}: resuming {already_trained} -> {target_nb_epochs} "
          f"({additional_epochs} more epoch(s)) ===")

    if make_backup:
        _backup_run(out_dir, run_name, already_trained)

    # Rebuilds the network with w1/w2 (and w0, for fixed-distribution runs)
    # restored to their exact epoch-{already_trained} values. For const_w0 /
    # lif_hidden_bistable_output / fixed-distribution runs this is already
    # fully correct as-is (see module docstring); only the learnable case
    # needs the extra fix-up below.
    model, old_history, loaded_config = rb.load_trained_model(out_dir, run_name, device=device, dtype=dtype)

    if config.get("variant") == "distribution_w0" and config.get("learnable"):
        if not include_learnable:
            # _select_runs already filters these out, but guard here too in
            # case resume_one_run is ever called directly with a learnable run.
            raise ValueError(f"{run_name} is a learnable distribution_w0 run -- "
                              f"pass include_learnable=True to resume it")
        _make_w0_learnable_again(model)
        print(f"  [fix] re-wrapped w0 as a learnable Parameter for {run_name}")

    trainloader, testloader = lib.get_dataloaders(batch_size=config.get("batch_size", lib.DEFAULT_BATCH_SIZE))

    lib.set_seed(config.get("seed", lib.DEFAULT_SEED))
    lr = lr_override if lr_override is not None else config.get("lr", lib.DEFAULT_LR)
    max_batches_per_epoch = (max_batches_per_epoch_override if max_batches_per_epoch_override is not None
                              else config.get("max_batches_per_epoch"))

    new_history = lib.train_one_run(
        model, trainloader, testloader, run_name, device, dtype,
        lr=lr, nb_epochs=additional_epochs, max_batches_per_epoch=max_batches_per_epoch,
        print_every=print_every, eval_max_batches=eval_max_batches,
    )

    # Continuous 25+25=50-epoch record, not a separate second history --
    # old_history/new_history share the same 3 keys (train_loss/train_acc/test_acc).
    merged_history = {k: old_history[k] + new_history[k] for k in new_history}

    updated_config = dict(config)
    updated_config["nb_epochs"] = target_nb_epochs
    updated_config.setdefault("resume_log", [])
    updated_config["resume_log"].append({
        "from_epoch": already_trained, "to_epoch": target_nb_epochs,
        "lr_used": lr, "resumed_w0_learnable": bool(
            config.get("variant") == "distribution_w0" and config.get("learnable")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    save_run(model, merged_history, updated_config, out_dir=out_dir, run_name=run_name)
    print(f"[{run_name}] done -- now has {len(merged_history['test_acc'])} epochs on disk")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=lib.DEFAULT_OUT_DIR)
    parser.add_argument("--target_nb_epochs", type=int, default=50)
    parser.add_argument("--include_learnable", action="store_true",
                         help="also resume the 4 learnable distribution_w0 runs "
                              "(off by default)")
    parser.add_argument("--run_name", type=str, default=None,
                         help="resume just this one run (by its results-folder name) instead of every eligible run")
    parser.add_argument("--no_backup", action="store_true",
                         help="skip copying each run's pre-resume files to _pre_resume_backups/ first")
    parser.add_argument("--lr", type=float, default=None,
                         help="override lr for the resumed epochs (default: reuse each run's own saved lr)")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                         help="override for every run (e.g. small number for a quick smoke test of this script "
                              "itself); default: reuse each run's own saved setting (usually None = full epoch)")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--eval_max_batches", type=int, default=5)
    args = parser.parse_args()

    device = lib.get_device()
    dtype = torch.float
    print("device:", device)

    selected = _select_runs(args.out_dir, args.include_learnable, only_run_name=args.run_name)
    if not selected:
        print("Nothing to resume (see [skip] lines above, or check --run_name / --out_dir).")
        return

    print(f"\n{len(selected)} run(s) selected to resume -> {args.target_nb_epochs} epochs:")
    for run_name, config in selected:
        print(f"  {run_name}  (currently {config.get('nb_epochs', '?')} epochs, "
              f"variant={config.get('variant')})")

    run_times = []
    for i, (run_name, config) in enumerate(selected, start=1):
        run_start = time.time()
        try:
            resume_one_run(
                args.out_dir, run_name, config, args.target_nb_epochs, device, dtype,
                include_learnable=args.include_learnable, make_backup=not args.no_backup,
                lr_override=args.lr, max_batches_per_epoch_override=args.max_batches_per_epoch,
                print_every=args.print_every, eval_max_batches=args.eval_max_batches,
            )
        except Exception as e:
            print(f"  [ERROR] {run_name} failed to resume: {e!r} -- skipping, moving on to the next run")
            continue

        run_times.append(time.time() - run_start)
        avg = sum(run_times) / len(run_times)
        remaining = len(selected) - i
        if remaining > 0:
            print(f"  [batch estimate] avg run time so far: {avg/60:.1f} min -> "
                  f"~{avg*remaining/60:.0f} min left for the {remaining} remaining run(s)")

    print(f"\nAll done. {len(run_times)}/{len(selected)} run(s) resumed successfully.")


if __name__ == "__main__":
    main()
