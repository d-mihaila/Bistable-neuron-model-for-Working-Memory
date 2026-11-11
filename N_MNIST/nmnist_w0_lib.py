"""
nmnist_w0_lib.py

Shared library for the N-MNIST w0-variation experiment set. Every
`run_*.py` driver script in this folder imports from here rather than
redefining data loading / neuron construction / the training loop --
the only thing that differs between driver scripts is *which* w0
configuration gets built and what the run is called.

Architecture and all MNIST-matched timescale parameters below are carried
over unchanged from `bistable_wait_minimal.ipynb` (which itself matched
`network_previous.py` + `params.py`, the well-working plain-MNIST run):
plain `network_architecture.SNN` (input -> hidden -> output, spike-count
vote), plain `BistableNeuron` (no synaptic trace), `NUM_STEPS=100`
stimulus steps + `T_WAIT=100` silent steps, vote read from the entire
wait period, `dt_eff=0.025` (1.25 "cycles" of the neuron's own recovery
timescale during the stimulus, matching MNIST's working run), weight
init N(0, 0.5).

This file must sit in the same folder as `neuron_model.py` and
`network_architecture.py` (N_MNIST/), same as `bistable_wait_minimal.ipynb`.

Requires
--------
- neuron_model.py (this folder's plain, non-syn BistableNeuron + spike_fn)
- network_architecture.py (the generic SNN class, same one HSD uses)

What's NEW here, and why
-------------------------
1. `BistableNeuronVarW0` (subclasses `neuron_model.BistableNeuron`):
   the parent class caches `self.I_TC = w0**2/(1-a**2)` ONCE at
   __init__ time and never recomputes it. That's fine for a w0 that
   never changes after construction, but silently WRONG the moment w0
   varies per-neuron and/or is learnable (self.w0 becomes a tensor the
   optimizer updates) -- I_TC would then be computed from the *initial*
   w0 forever, never the current one. This subclass just recomputes I_TC
   live inside step(), from whatever self.w0 currently is (scalar float
   or tensor, leaf Parameter or not) -- identical behavior to the parent
   for a constant scalar w0, correct behavior for every other case.
   This mirrors how network_previous.py / netowork_mix_w0.py /
   network_learnable_w0.py's own `bistable_neuron()` function always
   recomputes I_TC from the w0 argument passed to that specific call.

2. `LIF_simple`: N_MNIST's own neuron_model.py only exposes LIF via
   snnTorch's `snn.Leaky` (an nn.Module with a different calling
   convention -- `spk, mem = lif(I)`, no functional
   init_state/step/voltage interface). `network_architecture.SNN`
   needs that functional interface for hidden_neuron/output_neuron, so
   this is a small port of HSD's neuron_model.py's LIF_simple class
   (same current-based, no-synaptic-filter LIF), rewired to use THIS
   folder's own `spike_fn` (from neuron_model.py) instead of HSD's
   separate SharedSurrGrad, so every neuron in this experiment set --
   bistable or LIF -- shares the same surrogate gradient.

3. `sample_w0`: ported unchanged from netowork_mix_w0.py -- draws a
   per-neuron w0 from 'normal', 'uniform' (variance-matched to normal),
   or 'two_tailed' (half strongly negative, half strongly positive).
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuron_model import BistableNeuron, spike_fn          # noqa: E402  (this folder's own, non-syn)
from network_architecture import SNN                        # noqa: E402  (generic, shared with HSD)


# ---------------------------------------------------------------------------
# MNIST-matched constants -- carried over unchanged from bistable_wait_minimal.ipynb
# ---------------------------------------------------------------------------
NUM_STEPS = 100                 # fixed N-MNIST input frames (ToFrame n_time_bins)
T_WAIT = 100                    # silent steps appended after the stimulus
T_EVAL = 100                    # vote read from the ENTIRE wait period

N_IN = 2 * 34 * 34
N_HIDDEN = 100
N_OUT = 10

V_TH, EPS, A, D, C = 1.0, 0.5, 0.1, 1.0, 0.0
TAU = 1.0                       # neutralizes BistableNeuron's /tau division (a no-op in this file's class)
DT_EFF = 2.5 / NUM_STEPS         # -> cycles_stim = NUM_STEPS * DT_EFF * EPS = 1.25, matches MNIST

# LIF timescale, for the hidden=LIF/output=bistable variant: tau_mem chosen so that,
# at time_step=DT_EFF (same integration step as the bistable neurons here, for a fair
# comparison), beta = exp(-DT_EFF/tau_mem) = 0.5, matching params.py's lif_beta=0.5
# used for the equivalent hidden-LIF comparison on plain MNIST (network_with_lif.py).
LIF_TAU_MEM = DT_EFF / math.log(2)
LIF_THRESHOLD = 1.0

DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 128
DEFAULT_NB_EPOCHS = 25 # before i did 10
DEFAULT_SEED = 42
DEFAULT_OUT_DIR = "results/nmnist_w0_experiments"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available()
                         else "cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size=DEFAULT_BATCH_SIZE, data_dir="./data", cache_dir="./cache/nmnist_w0"):
    """Same fixed-frame-count pipeline as bistable_wait_minimal.ipynb --
    ToFrame(n_time_bins=NUM_STEPS) gives every sample the same shape, no
    padding needed. Disk-cached, so re-running a different driver script
    against the same cache_dir only pays the decode cost once."""
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=NUM_STEPS),
    ])
    trainset = tonic.datasets.NMNIST(save_to=data_dir, transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to=data_dir, transform=frame_transform, train=False)

    cached_trainset = DiskCachedDataset(trainset, cache_path=os.path.join(cache_dir, "train"))
    cached_testset = DiskCachedDataset(testset, cache_path=os.path.join(cache_dir, "test"))

    trainloader = DataLoader(cached_trainset, batch_size=batch_size, shuffle=True,
                              collate_fn=tonic.collation.PadTensors(batch_first=True))
    testloader = DataLoader(cached_testset, batch_size=batch_size, shuffle=True,
                             collate_fn=tonic.collation.PadTensors(batch_first=True))
    return trainloader, testloader


def flatten_batch(x, device, dtype):
    return x.reshape(x.shape[0], x.shape[1], -1).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Neurons
# ---------------------------------------------------------------------------

class BistableNeuronVarW0(BistableNeuron):
    """See module docstring, point 1: identical to BistableNeuron for a
    constant scalar w0, but recomputes I_TC live from self.w0 every step
    instead of using the parent's cached self.I_TC -- required whenever
    w0 is per-neuron and/or learnable."""

    def step(self, state, I):
        v, w = state["v"], state["w"]
        I_eff = I
        I_TC = self.w0 ** 2 / (1 - self.a ** 2)  # live, not self.I_TC

        def dv(v_, w_):
            return v_ ** 2 - w_ ** 2 + I_eff + I_TC

        def dw(v_, w_):
            return 0.5 * (self.a * v_ - w_ + self.w0)

        dt_eff = self.dt_eff
        kv1 = dv(v, w)
        kw1 = dw(v, w)

        kv2 = dv(v + dt_eff / 2 * kv1, w + dt_eff / 2 * kw1)
        kw2 = dw(v + dt_eff / 2 * kv1, w + dt_eff / 2 * kw1)

        kv3 = dv(v + dt_eff / 2 * kv2, w + dt_eff / 2 * kw2)
        kw3 = dw(v + dt_eff / 2 * kv2, w + dt_eff / 2 * kw2)

        kv4 = dv(v + dt_eff * kv3, w + dt_eff * kw3)
        kw4 = dw(v + dt_eff * kv3, w + dt_eff * kw3)

        v = v + (dt_eff / 6) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
        w = w + (dt_eff / 6) * (kw1 + 2 * kw2 + 2 * kw3 + kw4)

        spk = spike_fn(v - self.v_th)
        fired = spk.detach() > 0.5

        v = torch.where(fired, torch.full_like(v, self.c), v)
        w = torch.where(fired, torch.full_like(w, self.d), w)

        return {"v": v, "w": w}, spk


class LIF_simple:
    """Ported from HSD's neuron_model.py (see module docstring, point 2):
    current-based LIF, no synaptic filter, direct current injection --
    same convention as BistableNeuron's I_eff = I. Uses THIS folder's
    spike_fn (ATan surrogate, shared with the bistable neurons), not
    HSD's separate SharedSurrGrad."""

    def __init__(self, tau_mem=LIF_TAU_MEM, time_step=DT_EFF, threshold=LIF_THRESHOLD,
                 device="cpu", dtype=torch.float):
        self.beta = float(np.exp(-time_step / tau_mem))
        self.threshold = threshold
        self.time_step = time_step
        self.device = device
        self.dtype = dtype

    def init_state(self, batch_size, n_units):
        mem = torch.zeros((batch_size, n_units), device=self.device, dtype=self.dtype)
        return {"mem": mem}

    def step(self, state, I):
        mem = state["mem"]
        mthr = mem - self.threshold
        spk = spike_fn(mthr)
        rst = spk.detach()
        new_mem = (self.beta * mem + I) * (1.0 - rst)
        return {"mem": new_mem}, spk

    def voltage(self, state):
        return state["mem"]


def sample_w0(n, dist, mean=0.0, std=1.0):
    """Ported unchanged from netowork_mix_w0.py -- draw n per-neuron w0
    values instead of using one shared scalar.

    dist: 'normal', 'uniform', or 'two_tailed'.
    For 'uniform', half-width = std*sqrt(3) so Var(uniform) = std**2,
    matching 'normal' for a fair comparison. For 'two_tailed', |mean| is
    the magnitude of both peaks: half the neurons ~ N(-|mean|, std), half
    ~ N(+|mean|, std) -- e.g. mean=-1.25 -> half at -1.25, half at +1.25.
    """
    if dist == "normal":
        w0 = torch.normal(mean=torch.full((n,), float(mean)), std=float(std))
    elif dist == "uniform":
        half_width = float(std) * math.sqrt(3.0)
        low, high = mean - half_width, mean + half_width
        w0 = torch.empty(n).uniform_(low, high)
    elif dist == "two_tailed":
        mu = abs(float(mean))
        n_neg = n // 2
        n_pos = n - n_neg
        neg = torch.normal(mean=torch.full((n_neg,), -mu), std=float(std))
        pos = torch.normal(mean=torch.full((n_pos,), mu), std=float(std))
        w0 = torch.cat([neg, pos])
        w0 = w0[torch.randperm(n)]
    else:
        raise ValueError(f"Unknown w0 distribution '{dist}'; use 'normal', 'uniform', or 'two_tailed'.")
    return w0


def bistable_kwargs_common():
    return dict(v_th=V_TH, eps=EPS, a=A, d=D, c=C, tau=TAU, dt_eff=DT_EFF)


def make_const_w0_neurons(w0_value, device, dtype=torch.float):
    """One shared scalar w0 for every neuron in a layer -- the 6-value
    sweep (-2, -1, -0.5, 0.5, 1, 2) uses this."""
    kwargs = bistable_kwargs_common()
    hidden_neuron = BistableNeuronVarW0(w0=w0_value, device=device, dtype=dtype, **kwargs)
    output_neuron = BistableNeuronVarW0(w0=w0_value, device=device, dtype=dtype, **kwargs)
    hidden_neuron.w0_initial = hidden_neuron.w0
    output_neuron.w0_initial = output_neuron.w0
    return hidden_neuron, output_neuron


def make_distribution_neurons(dist, mean, std, learnable, device, dtype=torch.float):
    """Per-neuron w0 sampled from a distribution, either left fixed after
    sampling or made learnable (jointly trained with w1/w2). For
    dist='two_tailed', the hidden layer is the mixed (half +|mean|, half
    -|mean|) population; the output layer follows netowork_mix_w0.py's own
    convention of using a plain negative-only normal(-|mean|, std) there
    instead -- not mixed twice."""
    kwargs = bistable_kwargs_common()

    w0_hidden_vals = sample_w0(N_HIDDEN, dist, mean=mean, std=std).to(device=device, dtype=dtype)
    if dist == "two_tailed":
        w0_output_vals = sample_w0(N_OUT, "normal", mean=-abs(mean), std=std).to(device=device, dtype=dtype)
    else:
        w0_output_vals = sample_w0(N_OUT, dist, mean=mean, std=std).to(device=device, dtype=dtype)

    hidden_neuron = BistableNeuronVarW0(w0=w0_hidden_vals, device=device, dtype=dtype, **kwargs)
    output_neuron = BistableNeuronVarW0(w0=w0_output_vals, device=device, dtype=dtype, **kwargs)

    hidden_neuron.w0_initial = hidden_neuron.w0.detach().clone()
    output_neuron.w0_initial = output_neuron.w0.detach().clone()

    if learnable:
        hidden_neuron.w0 = torch.nn.Parameter(hidden_neuron.w0.detach().clone())
        output_neuron.w0 = torch.nn.Parameter(output_neuron.w0.detach().clone())

    return hidden_neuron, output_neuron


def make_lif_hidden_bistable_output_neurons(w0_value, device, dtype=torch.float):
    """Hidden layer: LIF_simple (no w0 at all -- not a bistable neuron).
    Output layer: BistableNeuronVarW0 with a constant scalar w0."""
    hidden_neuron = LIF_simple(tau_mem=LIF_TAU_MEM, time_step=DT_EFF, threshold=LIF_THRESHOLD,
                                device=device, dtype=dtype)
    output_neuron = BistableNeuronVarW0(w0=w0_value, device=device, dtype=dtype, **bistable_kwargs_common())
    output_neuron.w0_initial = output_neuron.w0
    return hidden_neuron, output_neuron


def make_all_lif_neurons(device, dtype=torch.float, tau_mem=None):
    """NEW (no-memory baseline) -- BOTH layers LIF_simple, no w0 anywhere.
    Same LIF parameters as the LIF layer of make_lif_hidden_bistable_output_neurons
    (beta = exp(-DT_EFF / tau_mem) = 0.5 by default), so the only difference to
    that variant is that the OUTPUT layer is LIF too. Used by run_all_lif.py to
    show what a network WITHOUT the bistable mechanism can and cannot hold
    through the silent wait period."""
    tau_mem = LIF_TAU_MEM if tau_mem is None else tau_mem
    hidden_neuron = LIF_simple(tau_mem=tau_mem, time_step=DT_EFF, threshold=LIF_THRESHOLD,
                               device=device, dtype=dtype)
    output_neuron = LIF_simple(tau_mem=tau_mem, time_step=DT_EFF, threshold=LIF_THRESHOLD,
                               device=device, dtype=dtype)
    return hidden_neuron, output_neuron


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SNNVarW0(SNN):
    """Identical to network_architecture.SNN, except parameters() also
    yields hidden_neuron.w0 / output_neuron.w0 whenever they are a Tensor
    with requires_grad=True (the learnable-w0 case) -- so the SAME
    optimizer.step() that updates w1/w2 also updates w0, with no other
    change to the training loop needed. For a constant scalar w0 (a
    plain float) or a fixed (non-learnable) per-neuron tensor, this is
    identical to the parent class -- nothing extra gets optimized."""

    def parameters(self):
        params = list(super().parameters())
        for neuron in (self.hidden_neuron, self.output_neuron):
            w0 = getattr(neuron, "w0", None)
            if isinstance(w0, torch.Tensor) and w0.requires_grad:
                params.append(w0)
        return params


def build_model(hidden_neuron, output_neuron, device, dtype=torch.float):
    return SNNVarW0(N_IN, N_HIDDEN, N_OUT, NUM_STEPS, hidden_neuron=hidden_neuron,
                     output_neuron=output_neuron, T_eval=T_EVAL, T_wait=T_WAIT,
                     device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, dtype, max_batches=5):
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = flatten_batch(x, device, dtype), y.to(device)
            spike_count, _ = model.forward(x)
            correct += (spike_count.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def _w0_mean_str(neuron):
    w0 = getattr(neuron, "w0", None)
    if isinstance(w0, torch.Tensor):
        return f"{w0.mean().item():+.4f}"
    if isinstance(w0, (int, float)):
        return f"{w0:+.4f}"
    return "n/a"


def train_one_run(model, trainloader, testloader, run_name, device, dtype,
                   lr=DEFAULT_LR, nb_epochs=DEFAULT_NB_EPOCHS, max_batches_per_epoch=None,
                   print_every=20, eval_max_batches=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    run_start = time.time()

    for epoch in range(nb_epochs):
        epoch_start = time.time()
        last_loss = None
        for i, (x, y) in enumerate(trainloader):
            if max_batches_per_epoch is not None and i >= max_batches_per_epoch:
                break
            x, y = flatten_batch(x, device, dtype), y.to(device)
            spike_count, recs = model.forward(x)
            loss_val = loss_fn(spike_count, y)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            last_loss = loss_val.item()

            if i % print_every == 0:
                print(f"  [{run_name}] epoch {epoch+1} batch {i:4d}  loss={last_loss:.4f}")

        train_acc = evaluate(model, trainloader, device, dtype, max_batches=eval_max_batches)
        test_acc = evaluate(model, testloader, device, dtype, max_batches=eval_max_batches)
        history["train_loss"].append(last_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"[{run_name}] epoch {epoch+1:2d}/{nb_epochs}  loss={last_loss:.4f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
              f"w0_hidden(mean)={_w0_mean_str(model.hidden_neuron)}  "
              f"w0_output(mean)={_w0_mean_str(model.output_neuron)}  "
              f"({time.time()-epoch_start:.1f}s this epoch, "
              f"{(time.time()-run_start)/60:.1f} min elapsed)")

    return history


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser):
    parser.add_argument("--nb_epochs", type=int, default=DEFAULT_NB_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                         help="cap batches/epoch for a quick run; default None = full epoch")
    parser.add_argument("--eval_max_batches", type=int, default=5,
                         help="batches used to estimate train/test accuracy each epoch")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    return parser


def print_timescale_check():
    cycles_stim = NUM_STEPS * DT_EFF * EPS
    cycles_wait = T_WAIT * DT_EFF * EPS
    print(f"dt_eff={DT_EFF:.5f}  cycles_stim={cycles_stim:.3f}  cycles_wait={cycles_wait:.3f}  "
          f"(MNIST reference: 1.25 / 2.5)")
