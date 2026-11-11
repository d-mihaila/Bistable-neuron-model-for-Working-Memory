"""
neuron_model.py (ODR)

Exact copy of the two neuron classes from HSD_take_2/neuron_model.py --
BistableNeuron (2D FitzHugh-Nagumo-like, RK4-integrated) and LIF_syn
(current-based LIF with a synaptic filter, ported from hsd_lib.py) -- kept
here so this folder is self-contained, same convention MNIST/N_MNIST/
HSD_take_2 already use ("exact copy ... kept here so this folder's
notebooks can import it locally").

2026-10-22: added LIF_simple, the LIF of the MNIST / N-MNIST experiments
(beta = 0.5 per ms, threshold 1, no synaptic filter). It is now the LIF of
the whole project: the ODR I population uses it by default
(network_architecture.RingSNN(i_model="lif")). LIF_syn stays for the
earlier results (i_model="lif_syn").

Both expose the same interface used throughout this codebase:
    state          = neuron.init_state(batch_size, n_units)
    state, spk     = neuron.step(state, I)
    v              = neuron.voltage(state)
    w              = neuron.recovery(state)   # bistable only, via hasattr check

For the ODR network we are NOT training these connections (see
network_architecture.py -- the recurrent weights are the fixed,
hand-built Mexican-hat kernel from Compte et al. 2000, exactly like the
original paper: no learning rule touches them). The ATan surrogate
gradient is kept on the spike function anyway, for two reasons: (1) it
costs nothing in a no-grad forward simulation, and (2) it means these
exact same neuron classes can be dropped into a trainable network later
(phase 2 of the experiment plan) with zero changes.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Surrogate spike function -- identical to HSD_take_2 / MNIST convention
# ---------------------------------------------------------------------------

class SpikeFunction(torch.autograd.Function):
    """Forward: Heaviside step. Backward: ATan surrogate gradient."""

    @staticmethod
    def forward(ctx, v, v_th):
        spk = (v >= v_th).float()
        ctx.save_for_backward(v - v_th)
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        (shifted_v,) = ctx.saved_tensors
        surrogate = 1 / (1 + (np.pi * shifted_v).pow(2)) / np.pi
        return grad_output * surrogate, None


spike_fn = SpikeFunction.apply


# ---------------------------------------------------------------------------
# Bistable neuron -- unchanged from HSD_take_2/neuron_model.py
# ---------------------------------------------------------------------------

class BistableNeuron:
    """
    2D bistable neuron (FitzHugh-Nagumo-like). No leak term in dv -- the
    v**2 component is regenerative. Recovery comes entirely from w.
    RK4-integrated, hard reset on both v and w.
    """

    def __init__(self, v_th=1.0, eps=0.5, a=0.1, d=1.0, c=0.0, w0=-0.5,
                 dt_eff=0.025, device="cpu", dtype=torch.float):
        self.v_th = v_th
        self.threshold = v_th  # alias, so plotting code can treat both models uniformly
        self.eps = eps
        self.a = a
        self.d = d
        self.c = c
        self.w0 = w0
        self.dt_eff = dt_eff
        self.I_TC = w0 ** 2 / (1 - a ** 2)
        self.device = device
        self.dtype = dtype

    def init_state(self, batch_size, n_units):
        shape = (batch_size, n_units) if isinstance(n_units, int) else (batch_size, *n_units)
        v = torch.zeros(shape, device=self.device, dtype=self.dtype)
        w = torch.zeros(shape, device=self.device, dtype=self.dtype)
        return {"v": v, "w": w}

    def step(self, state, I):
        v, w = state["v"], state["w"]
        I_eff = I

        def dv(v_, w_):
            return v_ ** 2 - w_ ** 2 + I_eff + self.I_TC

        def dw(v_, w_):
            return self.eps * (self.a * v_ - w_ + self.w0)

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

        spk = spike_fn(v, self.v_th)
        fired = spk.detach() > 0.5

        v = torch.where(fired, torch.full_like(v, self.c), v)
        w = torch.where(fired, torch.full_like(w, self.d), w)

        return {"v": v, "w": w}, spk

    def voltage(self, state):
        return state["v"]

    def recovery(self, state):
        return state["w"]


# ---------------------------------------------------------------------------
# LIF with synaptic filter -- unchanged from HSD_take_2/hsd_lib.py (LIF_syn),
# "the recipe that already worked ... every bistable variant is compared
# against". This is what "the LIF model we used for the HSD test" means.
# Since 2026-10-22 NOT the default any more (see LIF_simple below); kept
# so that the earlier results can be reproduced (RingSNN(i_model="lif_syn")).
# ---------------------------------------------------------------------------

class LIF_syn:
    """Current-based LIF with a synaptic filter."""

    def __init__(self, tau_mem=10e-3, tau_syn=5e-3, time_step=1e-3,
                 threshold=1.0, device="cpu", dtype=torch.float):
        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))
        self.threshold = threshold
        self.time_step = time_step
        self.device = device
        self.dtype = dtype

    def init_state(self, batch_size, n_units):
        syn = torch.zeros((batch_size, n_units), device=self.device, dtype=self.dtype)
        mem = torch.zeros((batch_size, n_units), device=self.device, dtype=self.dtype)
        return {"syn": syn, "mem": mem}

    def step(self, state, I):
        syn, mem = state["syn"], state["mem"]
        spk = spike_fn(mem, self.threshold)
        rst = spk.detach()
        new_syn = self.alpha * syn + I
        new_mem = (self.beta * mem + syn) * (1.0 - rst)
        return {"syn": new_syn, "mem": new_mem}, spk

    def voltage(self, state):
        return state["mem"]


# ---------------------------------------------------------------------------
# LIF of the MNIST / N-MNIST experiments (2026-10-22) -- THE LIF of this
# project. Same step as LIF_simple in MNIST/mnist_lib.py and
# N_MNIST/nmnist_w0_lib.py: current-based, no synaptic filter,
#     v <- beta * v + I ;  spike when v > threshold ;  then v = 0
# (the input of the spiking step is lost, as in MNIST). beta = 0.5 per ms
# (tau_mem = 1 ms / ln 2 = 1.44 ms), threshold 1. In MNIST tau_mem is
# written as DT_EFF / ln 2 with time_step = DT_EFF: the same beta per step,
# and 1 step = 1 ms in every experiment.
# ---------------------------------------------------------------------------

class _StrictSpike(torch.autograd.Function):
    """Forward: v > v_th (strict, as MNIST / N_MNIST spike_fn). Backward: ATan surrogate."""

    @staticmethod
    def forward(ctx, v, v_th):
        ctx.save_for_backward(v - v_th)
        return (v > v_th).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (shifted_v,) = ctx.saved_tensors
        surrogate = 1 / (1 + (np.pi * shifted_v).pow(2)) / np.pi
        return grad_output * surrogate, None


class LIF_simple:
    """The MNIST / N-MNIST LIF: current-based, no synaptic filter, beta = 0.5 per ms."""

    def __init__(self, tau_mem=1e-3 / math.log(2), time_step=1e-3, threshold=1.0,
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
        spk = _StrictSpike.apply(mem, self.threshold)
        rst = spk.detach()
        new_mem = (self.beta * mem + I) * (1.0 - rst)
        return {"mem": new_mem}, spk

    def voltage(self, state):
        return state["mem"]


AVAILABLE_MODELS = ("lif", "lif_syn", "bistable")


def make_neuron(model, device="cpu", dtype=torch.float, **kwargs):
    """Single switch point -- build a neuron model by name. kwargs are passed
    straight to the chosen class (e.g. w0=... for 'bistable').
    'lif' = LIF_simple (the MNIST / N-MNIST LIF, the default LIF everywhere
    since 2026-10-22); 'lif_syn' = the earlier LIF with a synaptic filter."""
    model = model.lower()
    if model == "lif":
        return LIF_simple(device=device, dtype=dtype, **kwargs)
    elif model == "lif_syn":
        return LIF_syn(device=device, dtype=dtype, **kwargs)
    elif model == "bistable":
        return BistableNeuron(device=device, dtype=dtype, **kwargs)
    else:
        raise ValueError(f"Unknown model '{model}': expected one of {AVAILABLE_MODELS}")
