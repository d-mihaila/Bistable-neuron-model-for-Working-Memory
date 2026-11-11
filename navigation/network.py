"""
network.py (navigation)

The network of the final snake task. Four parts, nothing else:

  1. RING: n_excit excitatory neurons on a circle (16 by default), Mexican-hat
     E -> E weights (Compte et al. 2000, as in ODR). No inhibitory neurons.
     Bistable neurons (w0 sets the regime) or LIF neurons (the control).
  2. INPUT: the apple cue, a Poisson spike barrage centred on the apple's
     compass angle, plus the same background Poisson input as in ODR.
  3. DECISION: four neurons (E, N, W, S). Each reads the ring within 90 deg of
     its direction (cos^2 weights). They inhibit each other. The one with most
     spikes in a 200 ms slot is the step.
  4. COUNTERS (distance meters): one neuron per axis. Each is a LIF neuron
     without leak (it adds up, it never forgets) with self-excitation (once it
     fires it keeps firing). Its spikes inhibit the two decision neurons of its
     axis (E and W, or N and S) through a slow synapse (20 ms).

One look (look_begin):
  forget  300 ms : w0 = +1 on the ring and a -3 current on every neuron
  cue     100 ms : w0 = -1 everywhere, the apple cue is on
  RF             : the ring's own activity in 300-375 ms gives the RF centre;
                   from now on w0 = -1 within +-42.5 deg of it, w0_out outside
Then look_slot runs the walk in slots, and look_end closes the look.

The counters are set by the task at every look (enc_set: start voltage from the
number of steps along the axis) and get one kick per step along their axis
(enc_kick). This is the explicit, non-neural part of the model.

neuron_model.py is the same file as ../ODR/neuron_model.py.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from neuron_model import BistableNeuron, LIF_syn, LIF_simple

DIR_NAMES = ["E", "N", "W", "S"]                     # decision neuron k = compass direction k
DIR_ANGLES = np.array([0.0, 90.0, 180.0, 270.0])
HEADING_STEPS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

# ODR ring settings (run_odr_demo.DEFAULT_KWARGS): same E->E scale, background and per-spike weights
RING = dict(sigma_deg=20.0, j_pos=1.6, g_ee_scale=1.6, n_extern_poisson=1000, bg_rate_hz=1.4, bg_weight_e=0.0105)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preferred_angles(n):
    return np.linspace(0.0, 360.0, n, endpoint=False)


def mexican_hat_kernel(n_excit, sigma_deg=20.0, j_pos=1.6):
    """Compte et al. (2000) E->E kernel, mean 1 over the ring."""
    tmp = math.sqrt(2.0 * math.pi) * sigma_deg * math.erf(180.0 / math.sqrt(2.0) / sigma_deg) / 360.0
    j_neg = (1.0 - j_pos * tmp) / (1.0 - tmp)
    d = np.arange(n_excit)
    d = 360.0 * np.minimum(d, n_excit - d) / n_excit
    return j_neg + (j_pos - j_neg) * np.exp(-0.5 * d ** 2 / sigma_deg ** 2)


def ring_distance(a_deg, b_deg):
    """Absolute angular distance, broadcast, in [0, 180]."""
    return np.abs((np.asarray(a_deg) - np.asarray(b_deg) + 180.0) % 360.0 - 180.0)


def signed_angle(theta_deg):
    """Angle in (-180, 180]."""
    a = (np.asarray(theta_deg, dtype=float) + 180.0) % 360.0 - 180.0
    return np.where(a == -180.0, 180.0, a)


def cue_profile(angles_deg, center_deg, width_deg=40.0, peak_rate_hz=750.0):
    """Gaussian rate (Hz) over the ring, FWHM = width_deg, cut at 2.5 sigma (ODR task._angle_profile)."""
    sigma = max(width_deg / 2.355, 1e-3)
    diff = (angles_deg - center_deg + 180.0) % 360.0 - 180.0
    prof = peak_rate_hz * np.exp(-0.5 * (diff / sigma) ** 2)
    return np.where(np.abs(diff) <= 2.5 * sigma, prof, 0.0)


def decode_population_vector(act, angles_deg):
    """Population vector over the last axis. Returns (angle_deg, resultant length)."""
    act = np.asarray(act, dtype=float)
    th = np.deg2rad(angles_deg)
    x, y = (act * np.cos(th)).sum(-1), (act * np.sin(th)).sum(-1)
    tot = act.sum(-1)
    res = np.where(tot > 0, np.sqrt(x ** 2 + y ** 2) / np.where(tot > 0, tot, 1.0), 0.0)
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0, res


def weighted_circular_readout(decoded_deg, resultant, t0, t1):
    """Resultant-weighted circular mean over [t0, t1) (ODR w0_gating)."""
    d = np.deg2rad(decoded_deg[:, t0:t1])
    r = resultant[:, t0:t1]
    x, y = (r * np.cos(d)).sum(-1), (r * np.sin(d)).sum(-1)
    s = np.clip(r.sum(-1), 1e-9, None)
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0, np.sqrt(x ** 2 + y ** 2) / s


def decision_profile(dist_deg, fade_deg=90.0):
    """Ring -> decision weight by angular distance: cos^2(pi/2 d / fade), 0 from fade_deg on."""
    d = np.asarray(dist_deg, dtype=float)
    f = np.cos(0.5 * np.pi * np.clip(d / fade_deg, 0.0, 1.0)) ** 2
    return np.where(d < fade_deg, f, 0.0)


@dataclass
class Timing:
    t_pre: int = 300        # forget window (w0_pre, pre_amp)
    t_cue: int = 100        # cue on, w0 = -1 everywhere

    @property
    def cue_on(self):
        return self.t_pre

    @property
    def cue_off(self):
        return self.t_pre + self.t_cue


def make_lif(beta=0.5, threshold=1.0, device="cpu", dtype=torch.float):
    """The MNIST / N-MNIST LIF (neuron_model.LIF_simple): v <- beta v + I, spike when
    v > threshold, then v = 0. beta = voltage kept per ms (1 step = 1 ms)."""
    return LIF_simple(tau_mem=-1e-3 / math.log(beta), time_step=1e-3, threshold=threshold, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class SnakeNet:
    """Ring + cue + 4 decision neurons + 2 counters. See the module docstring.

    e_model, dec_model : "bistable" or "lif" (the MNIST LIF, lif_beta / lif_threshold).
    stim_weight_e      : current per cue spike onto a ring neuron.
    ee_gain            : factor on the E -> E weights (default 1).
    w0_mem, w0_out     : w0 inside / outside the RF after the cue. w0_out = -1: no RF.
    w0_pre, pre_amp    : the forget window. (-1, 0): no forget, a plain wait.
    g_de, g_dd         : ring -> decision gain (weights g_de / n_excit), decision <-> decision
                         inhibition (one spike = -g_dd).
    g_enc, enc_self, enc_tau_syn, enc_map : counters (inhibition onto the decision
                         neurons, self-excitation, synaptic time constant, which decision
                         neurons each counter blocks).
    """

    def __init__(self, n_excit=16, *, e_model="bistable", dec_model="bistable", lif_beta=0.5, lif_threshold=1.0,
                 stim_weight_e=0.084, ee_gain=1.0, w0_mem=-1.0, w0_out=0.0, rf_radius_deg=2.5 * 40.0 / 2.355,
                 rf_decode=(300, 375), stim_width_deg=40.0, w0_pre=1.0, pre_amp=-3.0, dec_w0=-1.0,
                 dec_angles=DIR_ANGLES, dec_fade_deg=90.0, g_de=None, g_dd=None,
                 g_enc=10.0, enc_self=None, enc_tau_mem=1e6, enc_tau_syn=20.0, enc_map=None,
                 timing=None, dt_eff=0.025, seed=None, device="cpu", dtype=torch.float):
        if g_de is None:
            g_de = 6.0 if dec_model == "lif" else 100.0
        if g_dd is None:
            g_dd = 3.0 if dec_model == "lif" else 50.0
        self.n_excit = n_excit
        self.device, self.dtype = device, dtype
        self.dt_s = 1e-3
        self.angles_deg = preferred_angles(n_excit)
        self.timing = timing or Timing()
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        # ring
        self.e_model, self.dec_model = e_model, dec_model
        self.w0_mem, self.w0_out, self.w0_pre, self.pre_amp = w0_mem, w0_out, w0_pre, pre_amp
        self.rf_radius_deg, self.rf_decode = rf_radius_deg, rf_decode
        if e_model == "bistable":
            self.e_neuron = BistableNeuron(w0=w0_mem, dt_eff=dt_eff, device=device, dtype=dtype)
        elif e_model == "lif":                                   # no w0: no RF gate, no bistable memory
            self.e_neuron = make_lif(lif_beta, lif_threshold, device, dtype)
        else:
            raise ValueError("e_model must be 'bistable' or 'lif'")
        kern = mexican_hat_kernel(n_excit, RING["sigma_deg"], RING["j_pos"])
        idx = np.arange(n_excit)
        dist = np.abs(idx[:, None] - idx[None, :])
        dist = np.minimum(dist, n_excit - dist)
        W = kern[dist] * RING["g_ee_scale"] / n_excit
        self.W_EE = torch.tensor(W if ee_gain == 1.0 else W * ee_gain, device=device, dtype=dtype)
        self._bg_rate_hz = RING["n_extern_poisson"] * RING["bg_rate_hz"]
        self.bg_weight_e = RING["bg_weight_e"]
        self.stim_weight_e = stim_weight_e
        self.stim_width_deg = stim_width_deg

        # decision neurons
        self.dec_angles = np.asarray(dec_angles, dtype=float).copy()
        self.n_dec = n_dec = len(self.dec_angles)
        self.dec_w0, self.dec_fade_deg, self.g_de, self.g_dd = dec_w0, dec_fade_deg, g_de, g_dd
        prof = np.stack([decision_profile(ring_distance(self.dec_angles[k], self.angles_deg), dec_fade_deg)
                         for k in range(n_dec)])                 # (n_dec, n_excit)
        self.dec_profile = prof
        self.W_DE = torch.tensor(prof * g_de / n_excit, device=device, dtype=dtype)
        self.W_DD = torch.tensor(-g_dd * (1.0 - np.eye(n_dec)), device=device, dtype=dtype)
        if dec_model == "lif":
            self.d_neuron = make_lif(lif_beta, lif_threshold, device, dtype)
        elif dec_model == "bistable":
            self.d_neuron = BistableNeuron(w0=dec_w0, dt_eff=dt_eff, device=device, dtype=dtype)
        else:
            raise ValueError("dec_model must be 'lif' or 'bistable'")

        # counters: LIF without leak (tau_mem 1e6 ms) + self-excitation; inhibit their axis
        self.enc_neuron = LIF_syn(tau_mem=enc_tau_mem, device=device, dtype=dtype)
        self.enc_self = 1.02 * (1.0 - self.enc_neuron.alpha) if enc_self is None else enc_self
        enc_map = np.eye(n_dec) if enc_map is None else np.asarray(enc_map, dtype=float)
        self.n_enc = enc_map.shape[1]
        self.g_enc = g_enc
        self.W_ZD = torch.tensor(-g_enc * enc_map, device=device, dtype=dtype)
        self.enc_decay = float(np.exp(-1.0 / enc_tau_syn))
        self.weight_cv = 0.0

    # ----- weight noise ------------------------------------------------------

    def add_weight_noise(self, cv, seed):
        """Every synapse of the ring and the decision layer times (1 + cv xi), xi ~ N(0, 1),
        drawn once (as the MNIST / N-MNIST weight-noise test; signs may flip). Synapses:
        cue -> E (one per ring neuron), E -> E, E -> decision, decision -> decision. The
        counters (the explicit distance meter) and the background input are left as they are."""
        if cv <= 0:
            return
        rng = np.random.default_rng(seed)
        f = lambda shape: torch.tensor(1.0 + cv * rng.standard_normal(shape), device=self.device, dtype=self.dtype)
        self.W_EE = self.W_EE * f(tuple(self.W_EE.shape))
        self.W_DE = self.W_DE * f(tuple(self.W_DE.shape))
        self.W_DD = self.W_DD * f(tuple(self.W_DD.shape))
        self.stim_weight_e = float(self.stim_weight_e) * f((self.n_excit,))
        self.weight_cv = float(cv)

    # ----- helpers --------------------------------------------------------

    def _poisson(self, rate_hz, weight):
        lam = torch.clamp(torch.as_tensor(rate_hz, device=self.device, dtype=self.dtype) * self.dt_s, min=0)
        counts = torch.poisson(lam)
        return counts * weight, counts

    def _set_w0(self, w0_np):
        if self.e_model != "bistable":
            return
        w0 = torch.tensor(w0_np, device=self.device, dtype=self.dtype)
        self.e_neuron.w0 = w0
        self.e_neuron.I_TC = w0 ** 2 / (1 - self.e_neuron.a ** 2)

    def _set_dec_w0(self, w0):
        """Bistable decision neurons follow the forget window's w0 too."""
        if self.dec_model == "bistable":
            self.d_neuron.w0 = w0
            self.d_neuron.I_TC = w0 ** 2 / (1 - self.d_neuron.a ** 2)

    def _rest_state(self, neuron, B, n, w0):
        """Bistable rest point at input 0 (the saddle-node set by I_TC)."""
        a = neuron.a
        st = neuron.init_state(B, n)
        st["v"] = torch.full_like(st["v"], a * w0 / (1 - a ** 2))
        st["w"] = torch.full_like(st["w"], w0 / (1 - a ** 2))
        return st

    def init_state(self, B=1):
        """Fresh state: ring at v = w = 0 (as in ODR), bistable decision neurons at rest."""
        se = self.e_neuron.init_state(B, self.n_excit)
        sd = (self._rest_state(self.d_neuron, B, self.n_dec, self.dec_w0) if self.dec_model == "bistable"
              else self.d_neuron.init_state(B, self.n_dec))
        z = lambda n: torch.zeros(B, n, device=self.device, dtype=self.dtype)
        return dict(e=se, d=sd, spk_e=z(self.n_excit), spk_d=z(self.n_dec),
                    z=self.enc_neuron.init_state(B, self.n_enc), spk_z=z(self.n_enc), z_syn=z(self.n_enc))

    # ----- simulation core ------------------------------------------------

    @staticmethod
    def _new_rec():
        return dict(spk_e=[], spk_d=[], stim=[], dec=[], res=[], spk_z=[], v_z=[], count={})

    def _run(self, st, T, rec, *, pulse=0.0, cue_rate=None, dec_bias=None, record=True, decode=True):
        """Advance T ms with the ring's w0 map st["w0"] ((B, N)). pulse: extra current on
        every ring, decision and counter neuron. cue_rate: (B, N) Hz or None. dec_bias:
        (B, n_dec) extra current on the decision neurons (the tonic drive)."""
        B, N = st["w0"].shape
        dev, dt = self.device, self.dtype
        se, sd, sz = st["e"], st["d"], st["z"]
        spk_e_prev, spk_d_prev, spk_z_prev, z_syn = st["spk_e"], st["spk_d"], st["spk_z"], st["z_syn"]
        rate = torch.zeros(B, N, device=dev, dtype=dt) if cue_rate is None else cue_rate
        bg_rate = torch.full((B, N), self._bg_rate_hz, device=dev, dtype=dt)
        bias = None if dec_bias is None else torch.as_tensor(dec_bias, device=dev, dtype=dt)
        old_w0, old_itc = getattr(self.e_neuron, "w0", None), getattr(self.e_neuron, "I_TC", None)
        cnt = {k: torch.zeros(B, device=dev, dtype=dt) for k in ("e", "d", "stim", "enc")}
        try:
            self._set_w0(st["w0"])
            self._set_dec_w0(st.get("dec_w0", self.dec_w0))
            for _ in range(int(T)):
                # ring: recurrent E -> E (last ms's spikes) + background + cue + pulse
                in_e = spk_e_prev @ self.W_EE.T
                bg_e, _ = self._poisson(bg_rate, self.bg_weight_e)
                stim_e, stim_spk = self._poisson(rate, self.stim_weight_e)
                se, spk_e = self.e_neuron.step(se, in_e + stim_e + bg_e + pulse)

                # decision: ring -> decision, decision <-> decision inhibition
                in_d = spk_e_prev @ self.W_DE.T + spk_d_prev @ self.W_DD.T
                # counters: self-excitation (latch) + pulse; slow inhibition onto their axis
                sz, spk_z = self.enc_neuron.step(sz, spk_z_prev * self.enc_self + pulse)
                z_syn = z_syn * self.enc_decay + spk_z
                in_d = in_d + z_syn @ self.W_ZD.T
                in_d = in_d + pulse
                if bias is not None:
                    in_d = in_d + bias
                sd, spk_d = self.d_neuron.step(sd, in_d)

                if decode:
                    d_, r_ = decode_population_vector(spk_e.cpu().numpy(), self.angles_deg)
                else:                                   # not needed (walk, nothing recorded): skip for speed
                    d_ = r_ = np.zeros(B)
                rec["dec"].append(d_)
                rec["res"].append(r_)
                rec["spk_e"].append(spk_e.to(torch.uint8))
                rec["spk_d"].append(spk_d.to(torch.uint8))
                rec["spk_z"].append(spk_z.to(torch.uint8))
                cnt["e"] += spk_e.sum(1)
                cnt["d"] += spk_d.sum(1)
                cnt["enc"] += spk_z.sum(1)
                cnt["stim"] += stim_spk.sum(1)
                if record:
                    rec["stim"].append(stim_spk.clamp(max=255).to(torch.uint8))
                    rec["v_z"].append(sz["mem"].clone())
                spk_e_prev, spk_d_prev, spk_z_prev = spk_e, spk_d, spk_z
        finally:
            if self.e_model == "bistable":
                self.e_neuron.w0, self.e_neuron.I_TC = old_w0, old_itc
            self._set_dec_w0(self.dec_w0)
            for k, v in cnt.items():
                rec["count"][k] = rec["count"].get(k, 0) + v.cpu().numpy()
        st.update(e=se, d=sd, z=sz, spk_e=spk_e_prev, spk_d=spk_d_prev, spk_z=spk_z_prev, z_syn=z_syn)
        return st

    @staticmethod
    def _stack(rec, record=True):
        out = dict(decoded_deg=np.stack(rec["dec"], 1), resultant=np.stack(rec["res"], 1),
                   spk_e=torch.stack(rec["spk_e"], 1).cpu().numpy(),
                   spk_d=torch.stack(rec["spk_d"], 1).cpu().numpy(),
                   spk_z=torch.stack(rec["spk_z"], 1).cpu().numpy())
        out["stim_spk"] = torch.stack(rec["stim"], 1).cpu().numpy() if record and rec["stim"] else None
        out["v_z"] = torch.stack(rec["v_z"], 1).cpu().numpy() if record and rec["v_z"] else None
        out["counts"] = dict(rec["count"])
        return out

    @staticmethod
    def winner(counts, first, rng):
        """Most spikes, then first spike, then random. -1 if no spike."""
        if counts.max() <= 0:
            return -1
        best = np.flatnonzero(counts == counts.max())
        best = best[first[best] == first[best].min()]
        return int(best[0]) if len(best) == 1 else int(rng.choice(best))

    # ----- one look ------------------------------------------------------------

    def look_begin(self, cue_rate, state=None, record=True):
        """Forget window, cue (cue_rate: (B, N) Hz), RF placement. Returns a look dict."""
        tm = self.timing
        cue_rate = np.atleast_2d(np.asarray(cue_rate, dtype=np.float32))
        B, N = cue_rate.shape[0], self.n_excit
        st = self.init_state(B) if state is None else dict(state)
        rec = self._new_rec()
        st["w0"] = np.full((B, N), self.w0_pre, dtype=np.float32)
        st["dec_w0"] = self.w0_pre
        self._run(st, tm.t_pre, rec, pulse=self.pre_amp, record=record)
        st["w0"] = np.full((B, N), self.w0_mem, dtype=np.float32)
        st["dec_w0"] = self.dec_w0
        rate = torch.as_tensor(cue_rate, device=self.device, dtype=self.dtype)
        self._run(st, tm.t_cue, rec, cue_rate=rate, record=record)
        t0, t1 = self.rf_decode
        rf_center, _ = weighted_circular_readout(np.stack(rec["dec"], 1), np.stack(rec["res"], 1), t0, t1)
        inside = ring_distance(self.angles_deg[None, :], rf_center[:, None]) <= self.rf_radius_deg
        st["w0"] = np.where(inside, self.w0_mem, self.w0_out).astype(np.float32)
        return dict(st=st, rec=rec, B=B, rf_center=rf_center, w0_map=st["w0"].copy(), record=record,
                    t=tm.cue_off, slots=[])

    def look_slot(self, look, ms, dec_bias=None):
        """Run ms more of the walk. Returns (winner (B,), counts (B, n_dec)) of this slot."""
        rec, t0, B = look["rec"], look["t"], look["B"]
        if dec_bias is not None:
            dec_bias = np.broadcast_to(np.asarray(dec_bias, dtype=float), (B, self.n_dec)).copy()
        self._run(look["st"], ms, rec, dec_bias=dec_bias, record=look["record"], decode=look["record"])
        spk_d = torch.stack(rec["spk_d"][t0:t0 + ms], 1).cpu().numpy()          # (B, ms, n_dec)
        counts = spk_d.sum(1).astype(int)
        first = np.full((B, self.n_dec), np.inf)
        for b in range(B):
            for k in range(self.n_dec):
                ts = np.flatnonzero(spk_d[b, :, k])
                if ts.size:
                    first[b, k] = ts[0]
        win = np.array([self.winner(counts[b], first[b], self.rng) for b in range(B)])
        look["slots"].append(dict(t0=t0, t1=t0 + ms, counts=counts, winner=win))
        look["t"] = t0 + ms
        return win, counts

    def look_end(self, look):
        """Close a look. Returns (rec, state); rec has every spike of the look."""
        out = self._stack(look["rec"], look["record"])
        out.update(rf_center=look["rf_center"], w0_map=look["w0_map"], slots=look["slots"], T=look["t"],
                   timing=self.timing, enc_t=look.get("enc_t", self.timing.cue_off))
        state = {k: v for k, v in look["st"].items() if k not in ("w0", "dec_w0")}
        return out, state

    # ----- counters -------------------------------------------------------------

    def enc_set(self, look, v_start):
        """Set the counters' voltage at a look ((B, n_enc)); their synaptic current and last
        spikes are cleared. A voltage at or above 1 means: closed from the start."""
        st = look["st"]
        v = torch.as_tensor(np.asarray(v_start, dtype=np.float32), device=self.device, dtype=self.dtype)
        st["z"] = {"syn": torch.zeros_like(v), "mem": v.reshape(st["spk_z"].shape).clone()}
        st["spk_z"] = torch.zeros_like(st["spk_z"])
        st["z_syn"] = torch.zeros_like(st["spk_z"])
        look["enc_t"] = len(look["rec"]["spk_z"])          # enc_state only looks at spikes after this

    def enc_kick(self, look, k, amount):
        """One step along counter k's axis: add amount to its voltage (a copy of the step)."""
        st = look["st"]
        k = np.broadcast_to(np.asarray(k), (st["spk_z"].shape[0],))
        for b, kk in enumerate(k):
            if kk >= 0:
                st["z"]["mem"][b, int(kk)] += float(amount)

    def enc_state(self, look):
        """(voltage (B, n_enc), fired in the last 30 ms (B, n_enc) bool)."""
        st = look["st"]
        mem = st["z"]["mem"].cpu().numpy().copy()
        n = len(look["rec"]["spk_z"])
        spk = look["rec"]["spk_z"][max(look.get("enc_t", 0), n - 30):]
        recent = torch.stack(spk, 1).sum(1).cpu().numpy() > 0 if spk else np.zeros_like(mem, bool)
        return mem, recent
