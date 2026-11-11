"""
mnist_lib.py

Shared library for the "biologically timed" plain-MNIST rebuild, following
the exact same recipe as N_MNIST/nmnist_w0_lib.py: reuse the generic
`neuron_model.BistableNeuron` and `network_architecture.SNN` unchanged,
train in epochs over a real DataLoader, and drop the old MNIST-root
codebase's `steps=5` RK4-substep collapse in favour of ONE RK4 integration
step per network call -- exactly like N_MNIST already does.

Two deliberate conventions:

1. "1 network step = 1 ms" is a RELABELING, not a re-derivation. dt_eff
   below is numerically IDENTICAL to N_MNIST's DT_EFF (0.025) -- the value
   that was already tuned (via nmnist_w0_lib.py's own
   print_timescale_check()) to reproduce network_previous.py's working
   bistable dynamics. Nothing about v_th/eps/a/d/c/dt_eff changes here;
   only the unit label on the time axis changes, from "network steps" /
   "model-time units" to "ms". This keeps every run in this folder
   numerically comparable to N_MNIST.

2. NUM_STEPS / T_WAIT / T_EVAL mirror N_MNIST EXACTLY (100 / 100 / 100),
   not the MNIST-root defaults (T=50, T_wait=100, both at a coarser
   dt_eff=0.05). This makes cycles_stim = NUM_STEPS*DT_EFF*EPS = 1.25,
   matching the original MNIST-root tuning target exactly -- but
   cycles_wait = T_WAIT*DT_EFF*EPS = 1.25 as well, NOT the 2.5 cycles the
   MNIST-root code's T_wait=100 (at its own, coarser dt_eff=0.05) actually
   covers. This is a real, known gap between the two codebases (N_MNIST's
   own print_timescale_check() prints "(MNIST reference: 1.25 / 2.5)" and
   never actually hits the second number) -- we chose to
   mirror N_MNIST's T_WAIT=100 here for exact structural comparability
   between the two folders, accepting that the silent/wait phase in THIS
   folder is dynamically shorter (in relaxation-cycle terms) than the one
   the original MNIST-root persistent-memory results were tuned against.
   If that gap ever matters for a specific experiment, T_WAIT=200 would
   restore the full 2.5 cycles.

A note on a separate, pre-existing quirk inherited unchanged from
neuron_model.py (flagged here, not silently fixed, since it's shared code
copied byte-for-byte from N_MNIST): `BistableNeuron.step()`'s `dw()`
hardcodes the recovery variable's rate as `0.5 * (...)` rather than using
`self.eps * (...)` -- so the `eps=` constructor argument is stored but
never actually consulted by the dynamics. It happens to be numerically a
no-op today because EPS=0.5 everywhere in this codebase, but changing EPS
alone would silently do nothing. Worth knowing if you ever sweep eps.

Neuron/network variants (added after the initial const-w0-only version):
every `make_*_neurons()` function below returns (hidden_neuron,
output_neuron) ready to pass to build_model(), and every one of them sets
`.w0_initial` on any neuron whose w0 is per-neuron (fixed-distribution or
learnable) so save_experiment.py's save_run() can record both the initial
and final w0 -- ported unchanged from nmnist_w0_lib.py, same convention.
"""

import math
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see MNIST-root params.py for why

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuron_model import BistableNeuron, spike_fn          # noqa: E402  (this folder's own copy)
from network_architecture import SNN                        # noqa: E402  (this folder's own copy)


# ---------------------------------------------------------------------------
# N_MNIST-matched constants -- see module docstring for the two conventions
# these encode (ms-as-relabeling, T_WAIT mirroring N_MNIST exactly).
# ---------------------------------------------------------------------------
NUM_STEPS = 100                 # stimulus duration, in "ms" (1 network step = 1 ms, see above)
T_WAIT = 100                    # silent "ms" appended after the stimulus
T_EVAL = 100                    # vote read from the entire wait period

N_IN = 28 * 28                  # plain MNIST pixels -- the only architecture difference from N_MNIST
N_HIDDEN = 100
N_OUT = 10

V_TH, EPS, A, D, C = 1.0, 0.5, 0.1, 1.0, 0.0
TAU = 1.0                       # neutralizes BistableNeuron's /tau division (unused in this class's dv, see neuron_model.py)
DT_EFF = 2.5 / NUM_STEPS         # = 0.025, identical to N_MNIST's DT_EFF -- NOT re-derived, see docstring point 1
W0_DEFAULT = 1.0                 # matches the MNIST-root params.py default (w_0 = 1)

# LIF timescale, for the hidden=LIF/output=bistable variant (and its reverse) --
# tau_mem chosen so that, at time_step=DT_EFF (same integration step as the
# bistable neurons here), beta = exp(-DT_EFF/tau_mem) = 0.5, matching
# params.py's lif_beta=0.5 used for the equivalent hidden-LIF comparison on
# plain MNIST (network_with_lif.py) -- ported unchanged from nmnist_w0_lib.py.
LIF_TAU_MEM = DT_EFF / math.log(2)
LIF_THRESHOLD = 1.0

DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 128
DEFAULT_NB_EPOCHS = 50
DEFAULT_SEED = 42
DEFAULT_OUT_DIR = "results/mnist_ms_experiments"


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

def get_dataloaders(batch_size=DEFAULT_BATCH_SIZE, data_dir="./data"):
    """Plain MNIST, pixel intensities normalised to [0, 1] (NOT the usual
    mean/std normalisation -- these values are used directly as Bernoulli
    spike probabilities in encode_batch(), so they must stay in [0, 1])."""
    transform = transforms.ToTensor()  # already maps uint8 [0,255] -> float [0,1], no further scaling
    trainset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    testset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
    return trainloader, testloader


def encode_batch(imgs, device, dtype, num_steps=NUM_STEPS):
    """Turn a batch of static pixel-intensity images into a dense Poisson
    (Bernoulli-per-step) spike train, shaped (batch, num_steps, N_IN) --
    the same (batch, time, features) convention N_MNIST's tonic ToFrame
    output already has, so network_architecture.SNN.forward() (which does
    ONE big `torch.einsum` over the whole time axis, not a per-step
    Linear() call) needs no changes at all.

    This is also the one genuine, unavoidable difference in "input
    conversion" between the two datasets: N-MNIST's events are already
    binary/spiking straight off the DVS camera, so N_MNIST's
    `flatten_batch()` only reshapes and moves already-spiking data to
    `device`. Plain MNIST pixels are static intensities, not spikes, so
    they have to be converted into a spike train first -- that's what this
    function does (rate-code each pixel's intensity as a per-step spike
    probability, then sample). See the notebook's "input encoding" section
    for raster plots of exactly what this produces.

    imgs : (batch, 1, 28, 28) or (batch, 784), values in [0, 1].
    """
    imgs = imgs.reshape(imgs.shape[0], -1).to(device=device, dtype=dtype)   # (batch, N_IN)
    rates = imgs.unsqueeze(1).expand(-1, num_steps, -1)                     # (batch, num_steps, N_IN)
    return torch.bernoulli(rates)


# ---------------------------------------------------------------------------
# Neurons -- constant scalar w0, per-neuron distribution w0 (fixed or
# learnable), and LIF, all ported unchanged from nmnist_w0_lib.py, plus one
# new addition (make_bistable_hidden_lif_output_neurons, the reverse of
# N_MNIST's own make_lif_hidden_bistable_output_neurons).
# ---------------------------------------------------------------------------

class BistableNeuronVarW0(BistableNeuron):
    """See nmnist_w0_lib.py's module docstring, point 1: identical to
    BistableNeuron for a constant scalar w0, but recomputes I_TC live from
    self.w0 every step instead of using the parent's cached self.I_TC --
    required whenever w0 is per-neuron and/or learnable."""

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
    """Ported from nmnist_w0_lib.py (itself ported from HSD's neuron_model.py):
    current-based LIF, no synaptic filter, direct current injection -- same
    convention as BistableNeuron's I_eff = I. Uses this folder's spike_fn
    (ATan surrogate, shared with the bistable neurons)."""

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
    """Ported unchanged from netowork_mix_w0.py / nmnist_w0_lib.py -- draw n
    per-neuron w0 values instead of using one shared scalar.

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


def make_const_w0_neurons(w0_value=W0_DEFAULT, device="cpu", dtype=torch.float):
    """One shared scalar w0 for both layers."""
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
    instead -- not mixed twice. Ported unchanged from nmnist_w0_lib.py."""
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
    """Hidden layer: LIF_simple (no w0 at all). Output layer:
    BistableNeuronVarW0 with a constant scalar w0. Isolates whether the
    bistable, persistent-memory-holding dynamics matter specifically at
    the READOUT layer, decoupled from the hidden layer -- ported unchanged
    from nmnist_w0_lib.py."""
    hidden_neuron = LIF_simple(tau_mem=LIF_TAU_MEM, time_step=DT_EFF, threshold=LIF_THRESHOLD,
                                device=device, dtype=dtype)
    output_neuron = BistableNeuronVarW0(w0=w0_value, device=device, dtype=dtype, **bistable_kwargs_common())
    output_neuron.w0_initial = output_neuron.w0
    return hidden_neuron, output_neuron


def make_bistable_hidden_lif_output_neurons(w0_value, device, dtype=torch.float):
    """NEW -- the reverse of make_lif_hidden_bistable_output_neurons, not
    present in N_MNIST. Hidden layer: BistableNeuronVarW0 with a constant
    scalar w0. Output layer: LIF_simple (no w0 at all). Isolates whether
    the bistable dynamics matter specifically at the HIDDEN layer,
    decoupled from the readout -- the mirror-image question to
    run_lif_hidden.py's."""
    hidden_neuron = BistableNeuronVarW0(w0=w0_value, device=device, dtype=dtype, **bistable_kwargs_common())
    hidden_neuron.w0_initial = hidden_neuron.w0
    output_neuron = LIF_simple(tau_mem=LIF_TAU_MEM, time_step=DT_EFF, threshold=LIF_THRESHOLD,
                                device=device, dtype=dtype)
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
    optimizer.step() that updates w1/w2 also updates w0. For a constant
    scalar w0 (a plain float), a fixed (non-learnable) per-neuron tensor,
    or a layer with no w0 at all (LIF), this is identical to the parent
    class. Ported unchanged from nmnist_w0_lib.py."""

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
# Train / eval -- epoch-based, mirroring nmnist_w0_lib.train_one_run exactly
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, dtype, max_batches=5):
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = encode_batch(x, device, dtype)
            y = y.to(device)
            spike_count, _ = model.forward(x)
            correct += (spike_count.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train_one_run(model, trainloader, testloader, run_name, device, dtype,
                   lr=DEFAULT_LR, nb_epochs=DEFAULT_NB_EPOCHS, max_batches_per_epoch=None,
                   print_every=20, eval_max_batches=5):
    """Train for `nb_epochs` full passes over trainloader -- epochs, not a
    fixed TOTAL_STEPS gradient-step budget like the MNIST-root
    train_fixed_steps(). Structurally identical to
    nmnist_w0_lib.train_one_run(); only encode_batch()/N_IN differ.

    This is a single, straight-through training run -- there is no
    resume/continue-training path in this folder. N_MNIST's own
    continue_training.py, used to resume `dist_two_tailed_mean-1.25_std0.50_learnable`
    partway through, is what silently overwrote that run's saved "initial"
    w0 with its already-partially-trained value (the root cause tracked
    down earlier in this project) -- deliberately not ported here so that
    mistake can't recur."""
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
            x = encode_batch(x, device, dtype)
            y = y.to(device)
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
    """Same diagnostic as nmnist_w0_lib.py's own function -- run both
    side by side and they now print identical numbers, since this folder
    mirrors N_MNIST's NUM_STEPS/T_WAIT/DT_EFF exactly (see module
    docstring, point 2, for the one caveat: the wait phase's cycle count
    doesn't match the ORIGINAL MNIST-root tuning target, by design)."""
    cycles_stim = NUM_STEPS * DT_EFF * EPS
    cycles_wait = T_WAIT * DT_EFF * EPS
    print(f"dt_eff={DT_EFF:.5f}  cycles_stim={cycles_stim:.3f}  cycles_wait={cycles_wait:.3f}  "
          f"(MNIST-root reference: 1.25 / 2.5 -- wait phase intentionally mirrors "
          f"N_MNIST's T_WAIT=100 instead, see module docstring)")
