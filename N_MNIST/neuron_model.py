"""
neuron_model.py

The bistable (FitzHugh-Nagumo-like) spiking neuron, wrapped so it can drop
into the same nn.Sequential pipeline as snnTorch's snn.Leaky -- same call
signature, same hidden-state reset pattern, same (spk) / (spk, mem) return
convention. The notebook picks between the two with a single switch:

    model = "lif"        # or "bistable"
    net = nn.Sequential(
        nn.Conv2d(2, 12, 5),
        make_spiking_layer(model, lif_kwargs=lif_kwargs, bistable_kwargs=bistable_kwargs, device=device),
        ...
    )

Nothing in plots.py needs to know which one was chosen: get_weight_layers()
only ever looks at Conv2d/Linear (neither neuron model has weights of its
own), and SpikeRecorder finds spiking layers structurally -- any module with
a `reset_hidden()` method counts, which both snn.Leaky (via a light check)
and BistableSpikingLayer satisfy.
"""

import math

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils as snn_utils


# ---------------------------------------------------------------------------
# Surrogate spike function
# ---------------------------------------------------------------------------
# BistableNeuron.step() calls a module-level `spike_fn(v - v_th)` -- a hard
# threshold on the forward pass, with a smooth surrogate gradient on the
# backward pass so the network stays trainable. This uses the same
# arctangent surrogate as `surrogate.atan()` from snnTorch (used for the LIF
# model elsewhere in this notebook), so switching `model` doesn't also
# quietly switch to an easier or harder gradient to train through.

DEFAULT_SPIKE_ALPHA = 2.0


class _ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2 * (1 + (math.pi / 2 * alpha * x) ** 2))
        return grad_output * grad, None


def spike_fn(x, alpha=DEFAULT_SPIKE_ALPHA):
    """Heaviside step on the forward pass, arctangent surrogate gradient on
    the backward pass. `BistableNeuron.step()` calls this as `spike_fn(v -
    self.v_th)` with a single positional argument, so the default alpha is
    what's actually used unless you edit DEFAULT_SPIKE_ALPHA above."""
    return _ATanSpike.apply(x, alpha)


# ---------------------------------------------------------------------------
# The bistable neuron itself -- unmodified from what you handed over
# ---------------------------------------------------------------------------

class BistableNeuron:
    """
    2D bistable neuron (FitzHugh-Nagumo-like). No leak term in dv -- the
    `v**2` component is regenerative. Recovery comes entirely from `w`.
    RK4-integrated, hard reset on both v and w.
    """

    def __init__(self, v_th=1.0, eps=0.5, a=0.1, d=1.0, c=0.0, w0=-0.5, tau=0.1,
                 dt_eff=0.01, device="cpu", dtype=torch.float):
        self.v_th = v_th
        self.threshold = v_th  # alias, so plotting code can treat both models uniformly
        self.eps = eps
        self.a = a
        self.d = d
        self.c = c
        self.w0 = w0
        self.tau = tau
        self.dt_eff = dt_eff  # seconds per network step -- for plotting/time axes
        self.I_TC = w0 ** 2 / (1 - a ** 2)
        self.device = device
        self.dtype = dtype
        self.bias = 0.1

        # --- optional synaptic current filter (disabled for now) ----------
        # Uncomment to add a Zenke-style leaky synaptic filter ahead of the
        # bistable dv equation. Kept here as a one-line re-enable once we
        # want to compare "with vs. without synaptic filtering" directly.
        # self.tau_syn = 5e-3
        # self.alpha_syn = float(np.exp(-self.dt_eff / self.tau_syn))

    def init_state(self, batch_size, n_units):
        shape = (batch_size, n_units) if isinstance(n_units, int) else (batch_size, *n_units)
        # v = torch.rand(shape, device=self.device, dtype=self.dtype)
        # w = torch.rand(shape, device=self.device, dtype=self.dtype)
        v = torch.zeros(shape, device=self.device, dtype=self.dtype)
        w = torch.zeros(shape, device=self.device, dtype=self.dtype)
        return {"v": v, "w": w}

    def step(self, state, I):
        v, w = state["v"], state["w"]

        # --- optional synaptic filtering (disabled) ------------------------
        # syn = state["syn"]
        # new_syn = self.alpha_syn * syn + I
        # I_eff = syn   # feed the filtered current into dv instead of raw I
        I_eff = I  # direct injection -- no synaptic leakage for now

        def dv(v_, w_):
            # return (v_ ** 2 - w_ ** 2 + I_eff + self.I_TC - self.bias) / self.tau
            return v_ ** 2 - w_**2 + I_eff + self.I_TC  # - self.bias

        def dw(v_, w_):
            # return self.eps * (self.a * v_ - w_ + self.w0)
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

        # new_state = {"v": v, "w": w, "syn": new_syn}  # if synaptic filter enabled
        new_state = {"v": v, "w": w}
        return new_state, spk

    def voltage(self, state):
        return state["v"]

    def recovery(self, state):
        """The slow recovery variable w -- only bistable models have one.
        Its presence (via hasattr) is how SNN.forward decides whether to
        record a w-trace at all."""
        return state["w"]


# ---------------------------------------------------------------------------
# nn.Module wrapper -- what actually goes inside nn.Sequential
# ---------------------------------------------------------------------------

class BistableSpikingLayer(nn.Module):
    """
    Drop-in analog of `snn.Leaky(init_hidden=True, ...)`, backed by
    BistableNeuron instead of a leaky-integrate-and-fire model.

    - Holds its own (v, w) state internally, like init_hidden=True does for
      snn.Leaky -- so the same forward_pass() loop (`spk_out, mem_out =
      net(data[step])`) works unchanged regardless of which model is used.
    - State is created lazily on first use (`reset_hidden()` just clears it
      to None), matching whatever shape the input has -- (batch, C, H, W)
      after a conv layer, or (batch, features) after the linear layer.
      BistableNeuron.step() is purely elementwise, so this works without
      touching BistableNeuron.init_state() (which assumes a flat
      (batch, n_units) shape and isn't used here).
    - Returns just `spk` when output=False (so it can feed straight into
      the next Conv2d/Linear layer, same as an intermediate snn.Leaky);
      returns `(spk, mem)` when output=True, where `mem` is the neuron's
      voltage `v` -- the direct analog of membrane potential.
    """

    def __init__(self, output=False, **neuron_kwargs):
        super().__init__()
        self.neuron = BistableNeuron(**neuron_kwargs)
        self.output = output
        self.state = None

    def reset_hidden(self):
        self.state = None

    def forward(self, I):
        if self.state is None or self.state["v"].shape != I.shape:
            self.state = self.neuron.init_state(I.shape[0], I.shape[1:])

        self.state, spk = self.neuron.step(self.state, I)

        if self.output:
            mem = self.neuron.voltage(self.state)
            return spk, mem
        return spk

# ---------------------------------------------------------------------------
# Network-building helpers
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = ("lif", "bistable")


def make_spiking_layer(model, output=False, lif_kwargs=None, bistable_kwargs=None,
                        device="cpu", dtype=torch.float):
    """Build one spiking layer -- an snn.Leaky or a BistableSpikingLayer --
    picked by `model`. This is the single switch point: change `model` in
    the notebook and every call site that goes through this function
    builds the other kind of network, with no other code changes needed.

    lif_kwargs       : passed to snn.Leaky (e.g. {"beta": 0.5, "spike_grad": surrogate.atan()}).
                        `init_hidden=True` is set automatically if not given.
    bistable_kwargs  : passed to BistableNeuron (e.g. {"v_th": 1.0, "eps": 0.5, ...}).
                        `device`/`dtype` are filled in automatically if not given.
    """
    model = model.lower()

    if model == "lif":
        kwargs = dict(lif_kwargs or {})
        kwargs.setdefault("init_hidden", True)
        return snn.Leaky(output=output, **kwargs)

    elif model == "bistable":
        kwargs = dict(bistable_kwargs or {})
        kwargs.setdefault("device", device)
        kwargs.setdefault("dtype", dtype)
        return BistableSpikingLayer(output=output, **kwargs)

    else:
        raise ValueError(f"Unknown model '{model}': expected one of {AVAILABLE_MODELS}")


def reset_net(net):
    """Resets hidden state for every spiking layer in `net`, whether it's
    an snn.Leaky (LIF) or a BistableSpikingLayer (bistable) -- or a mix of
    both. Use this everywhere forward_pass() used to call
    `snntorch.utils.reset(net)`, so the training loop and the before/after
    raster captures work unchanged no matter which `model` was selected.
    """
    snn_utils.reset(net)  # resets every snn.Leaky; harmless no-op on everything else
    for module in net.modules():
        if isinstance(module, BistableSpikingLayer):
            module.reset_hidden()




def init_weights_normal(net, mean=0.0, std=0.5, seed=None):
    """Re-initialize every Conv2d/Linear weight in `net` from N(mean, std),
    replacing PyTorch's default Kaiming-uniform init. Biases are zeroed.
    Call this right after building `net`."""
    if seed is not None:
        torch.manual_seed(seed)
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)