"""
w0_gating.py (ODR)

Selective persistent memory by gating w0 per neuron and over time.

Idea
----
w0 sets whether a bistable neuron can hold persistent activity. In the
single-w0 sweeps, w0 = -1 / -0.5 hold a clean bump; w0 = 0 is the
"forgetting" regime (the bump drifts, goes silent, and is fragile in small
networks). So w0 can act as a top-down gate:

  1. Cue on, w0 = -1 everywhere: the whole ring can encode.
  2. At cue offset, READ the angle the network itself encodes (population
     vector over the cue window -- not the true cue angle).
  3. Keep w0 = -1 only inside a "receptive field" around that decoded angle.
     Set every other neuron to w0_out (default 0). Only the cued location can
     now persist; the rest of the ring can still fire when driven, but
     cannot hold it.
  4. A distractor arrives mid-delay outside the receptive field.
  5. Read the remembered angle at the end of the delay.
  6. Forget: w0 = 0 everywhere for t_forget ms.
  7. Back to w0 = -1 everywhere, present a new cue at a new angle, and gate
     again around the new decoded angle. Read it out.

Nothing in network_architecture.py / neuron_model.py is modified. This file
reuses RingSNN for the weights and the Poisson drive, and re-implements only
the time loop so w0 can change every step.

About I_TC
----------
In BistableNeuron, I_TC = w0^2 / (1 - a^2) is derived from w0. Changing w0
therefore also changes the tonic current (w0 = -1 -> I_TC ~ 1.01; w0 = 0 ->
I_TC = 0). With couple_itc=True (default, model-consistent) both change
together. With couple_itc=False, I_TC is frozen at the value for
w0_reference, so only w0 itself (the recovery fixed point) changes. That
switch is the control for "is it w0, or just less tonic current?".
"""

from dataclasses import dataclass, field

import numpy as np
import torch

import task
from network_architecture import RingSNN, decode_population_vector, circular_error_deg


# ---------------------------------------------------------------------------
# Protocol description
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """One time span with one w0 rule.

    kind = "uniform": every E neuron gets w0.
    kind = "rf":      neurons within radius_deg of the receptive-field centre
                      get w0_in, all others get w0_out. The centre is decoded
                      from E spikes over decode_window (t0, t1) when
                      center_deg is None, otherwise it is fixed.
    """
    name: str
    t_start: int
    t_end: int
    kind: str = "uniform"
    w0: float = -1.0
    w0_in: float = -1.0
    w0_out: float = 0.0
    radius_deg: float = 42.5
    decode_window: tuple = None
    center_deg: float = None


@dataclass
class Pipeline:
    T: int
    stim_rate: torch.Tensor          # (batch, T, n_excit), Hz
    phases: list
    events: dict                     # named time markers, for plotting
    cues: list                       # [(name, angle, t0, t1)]
    distractors: list                # [(name, angle, t0, t1)]
    readouts: list                   # [(name, true_angle, t0, t1)]
    meta: dict = field(default_factory=dict)
    pulses: list = field(default_factory=list)   # [(name, t0, t1, amp)] current added to every E neuron


def build_pipeline(n_excit, angles_deg, *,
                   cue1_deg=90.0, cue2_deg=270.0,
                   distr1_deg=210.0, distr2_deg=30.0,
                   t_fix=100, t_cue=250, t_delay=1500,
                   t_distr=200, distr_onset_after_cue=400,
                   t_readout=400, t_forget=100,
                   stim_rate_hz=6000.0, stim_width_deg=40.0,
                   gate=True, rf_radius_deg=None,
                   w0_mem=-1.0, w0_out=0.0, w0_forget=0.0,
                   distractor_in_second=True,
                   neg_pulse=None, neg_amp=-3.0, t_neg=100,
                   batch_size=1, device="cpu", dtype=torch.float):
    """Two-memory pipeline: cue1 -> gate -> distractor1 -> readout1 ->
    forget -> cue2 -> gate -> distractor2 -> readout2.

    gate=False keeps w0 = w0_mem on every neuron for the whole trial except
    the forget window (the control: same inputs, no receptive-field gating).
    neg_pulse adds a NEGATIVE input pulse (current neg_amp, every E neuron) :
        None      no pulse
        "forget"  during the forget window (f_on .. f_off)
        "start"   at the very start of the trial, t = 0 .. t_neg (before cue 1)
    rf_radius_deg defaults to the cue's own stimulation range (the 2.5 sigma
    cutoff in task._angle_profile), i.e. "everything the cue did not drive".
    """
    if rf_radius_deg is None:
        rf_radius_deg = 2.5 * stim_width_deg / 2.355

    # ---- timeline -------------------------------------------------------
    c1_on = t_fix
    c1_off = c1_on + t_cue
    d1_on = c1_off + distr_onset_after_cue
    d1_off = d1_on + t_distr
    r1_end = c1_off + t_delay
    r1_on = r1_end - t_readout
    f_on, f_off = r1_end, r1_end + t_forget
    c2_on = f_off
    c2_off = c2_on + t_cue
    d2_on = c2_off + distr_onset_after_cue
    d2_off = d2_on + t_distr
    r2_end = c2_off + t_delay
    r2_on = r2_end - t_readout
    T = r2_end

    # ---- input rates (reuse task.py's spatial profile) -------------------
    rate = np.zeros((T, n_excit), dtype=np.float32)
    prof = lambda a: task._angle_profile(angles_deg, a, stim_width_deg, stim_rate_hz)
    rate[c1_on:c1_off] += prof(cue1_deg)
    rate[d1_on:d1_off] += prof(distr1_deg)
    rate[c2_on:c2_off] += prof(cue2_deg)
    distractors = [("distractor 1", distr1_deg, d1_on, d1_off)]
    if distractor_in_second:
        rate[d2_on:d2_off] += prof(distr2_deg)
        distractors.append(("distractor 2", distr2_deg, d2_on, d2_off))
    stim_rate = torch.tensor(rate, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

    # ---- w0 phases ------------------------------------------------------
    def mem_phase(name, t0, t1, cue_on, cue_off):
        if gate:
            return Phase(name, t0, t1, kind="rf", w0_in=w0_mem, w0_out=w0_out,
                         radius_deg=rf_radius_deg, decode_window=(cue_on, cue_off))
        return Phase(name, t0, t1, kind="uniform", w0=w0_mem)

    phases = [
        Phase("encode 1", 0, c1_off, kind="uniform", w0=w0_mem),
        mem_phase("hold 1", c1_off, f_on, c1_on, c1_off),
        Phase("forget", f_on, f_off, kind="uniform", w0=w0_forget),
        Phase("encode 2", c2_on, c2_off, kind="uniform", w0=w0_mem),
        mem_phase("hold 2", c2_off, T, c2_on, c2_off),
    ]

    pulses = []
    if neg_pulse == "forget":
        pulses.append(("neg. pulse", f_on, f_off, neg_amp))
    elif neg_pulse == "start":
        pulses.append(("neg. pulse", 0, min(t_neg, c1_on), neg_amp))
    elif neg_pulse is not None:
        raise ValueError("neg_pulse must be None, 'forget' or 'start'")

    events = dict(c1_on=c1_on, c1_off=c1_off, d1_on=d1_on, d1_off=d1_off,
                  r1_on=r1_on, r1_end=r1_end, f_on=f_on, f_off=f_off,
                  c2_on=c2_on, c2_off=c2_off, d2_on=d2_on, d2_off=d2_off,
                  r2_on=r2_on, r2_end=r2_end)
    return Pipeline(
        T=T, stim_rate=stim_rate, phases=phases, events=events,
        cues=[("cue 1", cue1_deg, c1_on, c1_off), ("cue 2", cue2_deg, c2_on, c2_off)],
        distractors=distractors,
        readouts=[("readout 1", cue1_deg, r1_on, r1_end), ("readout 2", cue2_deg, r2_on, r2_end)],
        meta=dict(gate=gate, rf_radius_deg=rf_radius_deg, w0_mem=w0_mem,
                  w0_out=w0_out, w0_forget=w0_forget, neg_pulse=neg_pulse, neg_amp=neg_amp),
        pulses=pulses)


# ---------------------------------------------------------------------------
# Long trials: many cue -> distractor -> readout -> forget loops, per-trial
# angles, gaps between phases, and delay jitter (used by
# ODR_w0_gating_experiments.ipynb)
# ---------------------------------------------------------------------------

class StimSchedule:
    """Cue / distractor rates built on the fly, one profile per trial.

    A full (batch, T, n) rate array is too big for trials of 20 s, so this
    stores only the input windows. run_gated only needs .shape and [:, t]."""

    def __init__(self, batch_size, T, n, device="cpu", dtype=torch.float):
        self.shape = (batch_size, T, n)
        self.events = []                              # (t0, t1, (batch, n) tensor)
        self._zero = torch.zeros(batch_size, n, device=device, dtype=dtype)
        self.device, self.dtype = device, dtype

    def add(self, t0, t1, profile):
        self.events.append((int(t0), int(t1), torch.as_tensor(profile, device=self.device, dtype=self.dtype)))

    def __getitem__(self, idx):
        _, t = idx
        out = None
        for t0, t1, p in self.events:
            if t0 <= t < t1:
                out = p if out is None else out + p
        return self._zero if out is None else out


def build_sequence(n_excit, angles_deg, cue_deg, distr_deg=None, *, batch_size=1,
                   t_fix=200, t_cue=250, t_delay=1500, t_distr=200,
                   distr_onset_after_cue=400, t_readout=400, t_forget=100, gap=100,
                   delay_shift=0, stim_rate_hz=6000.0, stim_width_deg=40.0,
                   gate=True, rf_radius_deg=None, w0_mem=-1.0, w0_out=0.0, w0_forget=1.0,
                   neg_pulse=None, neg_amp=-3.0, t_neg=100, start_amp=None, forget_keep_rf=False,
                   device="cpu", dtype=torch.float):
    """L memories in one continuous trial. For memory k:

        [gap, w0_mem]  cue k  -> hold k (gated)  ... distractor k ... readout k  [gap, still gated]
        -> forget k (w0_forget, + negative pulse if neg_pulse="forget")  -> next memory

    cue_deg, distr_deg : (L,) or (batch, L) angles in deg. distr_deg NaN (or None) = no distractor.
    gap                : ms of nothing between readout and forget, and between forget and the next cue.
    delay_shift        : extra delay (ms), scalar or (L,) -- shifts readout and forget relative to the
                         cue, which randomises the phase of the bump's volleys at forget onset.
    neg_pulse          : None, "forget" (in every forget window), "start" (t = 0 .. t_neg, before
                         cue 1), or both, e.g. ("start", "forget"). The forget pulse has current
                         neg_amp, the start pulse start_amp (default: neg_amp).
    forget_keep_rf     : True = the forget window keeps the hold's RF map (w0_mem inside, w0_out
                         outside) instead of a uniform w0_forget.
    There is no forget window after the last memory.
    """
    if rf_radius_deg is None:
        rf_radius_deg = 2.5 * stim_width_deg / 2.355
    kinds = set() if neg_pulse is None else ({neg_pulse} if isinstance(neg_pulse, str) else set(neg_pulse))
    if not kinds <= {"start", "forget"}:
        raise ValueError("neg_pulse must be None, 'start', 'forget' or both")
    start_amp = neg_amp if start_amp is None else start_amp
    cue = np.broadcast_to(np.atleast_2d(np.asarray(cue_deg, dtype=float)), (batch_size, np.size(cue_deg) if np.ndim(cue_deg) < 2 else np.shape(cue_deg)[1])).copy()
    L = cue.shape[1]
    if distr_deg is None:
        distr = np.full_like(cue, np.nan)
    else:
        distr = np.broadcast_to(np.atleast_2d(np.asarray(distr_deg, dtype=float)), cue.shape).copy()
    shift = np.broadcast_to(np.asarray(delay_shift, dtype=int), (L,))

    prof = lambda a: task._angle_profile(angles_deg, a, stim_width_deg, stim_rate_hz)
    profiles = lambda angs: np.stack([prof(a) for a in angs])          # (batch, n)

    phases, cues, distractors, readouts, forgets, pulses = [], [], [], [], [], []
    windows = []                                                        # (t0, t1, profile)
    t = 0
    for k in range(L):
        c_on = t + (t_fix if k == 0 else gap)
        c_off = c_on + t_cue
        d_on = c_off + distr_onset_after_cue
        d_off = d_on + t_distr
        r_end = c_off + t_delay + int(shift[k])
        r_on = r_end - t_readout
        last = k == L - 1
        hold_end = r_end + (0 if last else gap)

        phases.append(Phase(f"encode {k + 1}", t, c_off, kind="uniform", w0=w0_mem))
        if gate:
            phases.append(Phase(f"hold {k + 1}", c_off, hold_end, kind="rf", w0_in=w0_mem, w0_out=w0_out,
                                radius_deg=rf_radius_deg, decode_window=(c_on, c_off)))
        else:
            phases.append(Phase(f"hold {k + 1}", c_off, hold_end, kind="uniform", w0=w0_mem))
        windows.append((c_on, c_off, profiles(cue[:, k])))
        cues.append((f"cue {k + 1}", cue[:, k], c_on, c_off))
        if not np.all(np.isnan(distr[:, k])):
            dk = np.where(np.isnan(distr[:, k]), 0.0, distr[:, k])
            p = profiles(dk) * (~np.isnan(distr[:, k]))[:, None]
            windows.append((d_on, d_off, p))
            distractors.append((f"distractor {k + 1}", distr[:, k], d_on, d_off))
        readouts.append((f"readout {k + 1}", cue[:, k], r_on, r_end))
        if last:
            t = r_end
            break
        f_on, f_off = hold_end, hold_end + t_forget
        if forget_keep_rf and gate:
            phases.append(Phase(f"forget {k + 1}", f_on, f_off, kind="rf", w0_in=w0_mem, w0_out=w0_out,
                                radius_deg=rf_radius_deg, decode_window=(c_on, c_off)))
        else:
            phases.append(Phase(f"forget {k + 1}", f_on, f_off, kind="uniform", w0=w0_forget))
        forgets.append((f_on, f_off))
        if "forget" in kinds:
            pulses.append(("neg. pulse", f_on, f_off, neg_amp))
        t = f_off
    T = t
    if "start" in kinds:
        pulses.append(("neg. pulse", 0, min(t_neg, cues[0][2]), start_amp))

    stim = StimSchedule(batch_size, T, len(angles_deg), device=device, dtype=dtype)
    for t0, t1, p in windows:
        stim.add(t0, t1, p)
    f0 = forgets[0] if forgets else (T, T)
    events = dict(f_on=f0[0], f_off=f0[1])
    return Pipeline(T=T, stim_rate=stim, phases=phases, events=events, cues=cues,
                    distractors=distractors, readouts=readouts, pulses=pulses,
                    meta=dict(gate=gate, rf_radius_deg=rf_radius_deg, w0_mem=w0_mem, w0_out=w0_out,
                              w0_forget=w0_forget, neg_pulse=neg_pulse, neg_amp=neg_amp,
                              forget_keep_rf=bool(forget_keep_rf and gate), forgets=forgets, cue_deg=cue, distr_deg=distr, delay_shift=shift))


def signed_error_deg(decoded_deg, true_deg):
    """decoded - true, wrapped to [-180, 180)."""
    return (np.asarray(decoded_deg) - np.asarray(true_deg) + 180.0) % 360.0 - 180.0


def sequence_table(r, pipe):
    """One row per (trial, memory): angles, decoded angle, error, signed error,
    pull toward the distractor of the same memory, and offset to the previous cue."""
    cue, distr = pipe.meta["cue_deg"], pipe.meta["distr_deg"]
    B, L = cue.shape
    rows = []
    for k in range(L):
        ro = r["readouts"][f"readout {k + 1}"]
        se = signed_error_deg(ro["decoded"], cue[:, k])
        to_d = signed_error_deg(distr[:, k], cue[:, k])               # where the distractor sits
        to_prev = signed_error_deg(cue[:, k - 1], cue[:, k]) if k > 0 else np.full(B, np.nan)
        rf = r.get("rf_centers", {}).get(f"hold {k + 1}")      # self-placed RF centre (None without RF)
        rf = np.full(B, np.nan) if rf is None else np.broadcast_to(np.asarray(rf, dtype=float), (B,))
        rf_err = np.abs(signed_error_deg(rf, cue[:, k]))
        pop, N = r.get("pop_spikes"), r.get("n_excit", 1)
        c_off = pipe.cues[k][3]
        r_end = pipe.readouts[k][3]
        hold_rate = (pop[:, c_off:r_end].mean(1) / N * 1000) if pop is not None else np.full(B, np.nan)
        trial_rate = (pop.mean(1) / N * 1000) if pop is not None else np.full(B, np.nan)
        # one cue presentation (loop k) = from the start of the forget window before cue k (t = 0 for
        # cue 1) to the start of the next forget window (the end of the trial for the last cue).
        # Same boundaries in every network: networks without a forget signal still have the window.
        forgets = pipe.meta.get("forgets", [])
        lp0 = 0 if k == 0 else forgets[k - 1][0]
        lp1 = forgets[k][0] if k < len(forgets) else pop.shape[1] if pop is not None else 0
        pop_i, N_i = r.get("pop_spikes_i"), r.get("n_inhib", 1)
        if pop is not None:
            loop_e = pop[:, lp0:lp1].mean(1) / N * 1000
            loop_i = pop_i[:, lp0:lp1].mean(1) / N_i * 1000 if pop_i is not None else np.full(B, np.nan)
            loop_all = (pop[:, lp0:lp1].sum(1) + (pop_i[:, lp0:lp1].sum(1) if pop_i is not None else np.nan)) \
                / (N + N_i) / max(lp1 - lp0, 1) * 1000
        else:
            loop_e = loop_i = loop_all = np.full(B, np.nan)
        ro_frac = ro.get("frac_active", np.full(B, np.nan))
        ro_rate = ro.get("rate_hz", np.full(B, np.nan))
        ro_on = ro.get("on_cue", np.full(B, np.nan))
        for b in range(B):
            rows.append(dict(trial=b, memory=k + 1, cue_deg=cue[b, k], distr_deg=distr[b, k],
                             separation_deg=abs(to_d[b]) if np.isfinite(to_d[b]) else np.nan,
                             decoded_deg=float(ro["decoded"][b]), error_deg=float(abs(se[b])),
                             signed_error_deg=float(se[b]), strength=float(ro["strength"][b]),
                             pull_to_distractor_deg=float(se[b] * np.sign(to_d[b])) if np.isfinite(to_d[b]) else np.nan,
                             prev_cue_offset_deg=float(to_prev[b]),
                             pull_to_prev_cue_deg=float(se[b] * np.sign(to_prev[b])) if k > 0 else np.nan,
                             delay_shift_ms=int(pipe.meta["delay_shift"][k]),
                             rf_center_deg=float(rf[b]), rf_error_deg=float(rf_err[b]),
                             frac_active_readout=float(ro_frac[b]), rate_readout_hz=float(ro_rate[b]),
                             spikes_on_cue_readout=float(ro_on[b]),
                             rate_hold_hz=float(hold_rate[b]), rate_trial_hz=float(trial_rate[b]),
                             loop_t0_ms=int(lp0), loop_t1_ms=int(lp1), rate_loop_e_hz=float(loop_e[b]),
                             rate_loop_i_hz=float(loop_i[b]), rate_loop_hz=float(loop_all[b])))
    return rows


# ---------------------------------------------------------------------------
# Readout helpers
# ---------------------------------------------------------------------------

def weighted_circular_readout(decoded_deg, resultant, t0, t1):
    """Confidence-weighted circular mean of the per-step decode over [t0, t1)
    -- the same readout RingSNN.run uses for its final angle.
    decoded_deg, resultant: (batch, T). Returns (angle_deg, strength)."""
    d = np.deg2rad(decoded_deg[:, t0:t1])
    r = resultant[:, t0:t1]
    x = (r * np.cos(d)).sum(-1)
    y = (r * np.sin(d)).sum(-1)
    denom = np.clip(r.sum(-1), 1e-9, None)
    return (np.rad2deg(np.arctan2(y, x)) + 360) % 360, np.sqrt(x ** 2 + y ** 2) / denom


def _on_cue_share(counts, angles_deg, true_deg, radius_deg=2.5 * 40.0 / 2.355):
    """counts: (batch, n) spikes per neuron. Fraction of them from neurons within radius of the cue."""
    near = _ring_distance(angles_deg, np.broadcast_to(np.asarray(true_deg, dtype=float), (counts.shape[0],))) <= radius_deg
    tot = counts.sum(1)
    return np.where(tot > 0, (counts * near).sum(1) / np.clip(tot, 1e-9, None), np.nan)


def _ring_distance(angles_deg, center_deg):
    """(batch, n) absolute angular distance from each neuron to a per-trial centre."""
    c = np.asarray(center_deg).reshape(-1, 1)
    return np.abs((angles_deg[None, :] - c + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------
# Simulation with a time-varying, per-neuron w0
# ---------------------------------------------------------------------------

def run_gated(net: RingSNN, pipe: Pipeline, couple_itc=True, w0_reference=-1.0, verbose=False, record=True):
    """Same dynamics as RingSNN.run, plus a per-neuron w0 that follows
    pipe.phases. Returns recs (same keys as RingSNN.run where meaningful)
    plus:
        w0_map     : (batch, T, n_excit) the w0 every neuron had at every step
        rf_centers : {phase name: decoded RF centre per trial}
    record=False skips the spike / w0 recordings (needed for long trials:
    they grow as batch x T x n). Readouts and the decoded angle are kept.
        readouts   : {readout name: dict(decoded, strength, error, true)}
    """
    stim_rate_e = pipe.stim_rate
    B, T, N = stim_rate_e.shape
    dev, dt = net.device, net.dtype
    neuron = net.e_neuron
    a = neuron.a
    itc_ref = w0_reference ** 2 / (1 - a ** 2)

    state_e = neuron.init_state(B, N)
    state_i = net.i_neuron.init_state(B, net.n_inhib)
    spk_e_prev = torch.zeros(B, N, device=dev, dtype=dt)
    spk_i_prev = torch.zeros(B, net.n_inhib, device=dev, dtype=dt)
    refrac_i = torch.zeros(B, net.n_inhib, device=dev, dtype=dt)

    spk_e_rec, spk_i_rec, stim_rec, noise_rec = [], [], [], []
    pulse_amp = np.zeros(T, dtype=np.float32)            # extra current on every E neuron
    for _, p0, p1, amp in pipe.pulses:
        pulse_amp[p0:p1] += amp
    w0_rec = np.zeros((B, T, N), dtype=np.float16) if record else None
    rf_centers = {}

    # per-step decode so far (needed to decode the RF centre online)
    dec_so_far = np.zeros((B, T))
    res_so_far = np.zeros((B, T))

    phase_at = {}
    for ph in pipe.phases:
        for t in range(ph.t_start, min(ph.t_end, T)):
            phase_at[t] = ph

    # activity statistics (cheap, kept even with record=False): population E spikes per step,
    # and per-neuron E spike counts inside every readout window
    pop_spikes = np.zeros((B, T), dtype=np.float32)
    pop_spikes_i = np.zeros((B, T), dtype=np.float32)    # population I spikes per step (for the total rate)
    ro_counts = {name: np.zeros((B, N), dtype=np.float32) for name, _, _, _ in pipe.readouts}
    ro_at = {}
    for name, _, t0, t1 in pipe.readouts:
        for tt in range(t0, min(t1, T)):
            ro_at.setdefault(tt, []).append(name)

    current_phase, w0_vec = None, None
    old_w0, old_itc = neuron.w0, neuron.I_TC
    try:
        for t in range(T):
            ph = phase_at[t]
            if ph is not current_phase:          # phase change -> new w0 map
                current_phase = ph
                if ph.kind == "uniform":
                    w0_np = np.full((B, N), ph.w0, dtype=np.float32)
                else:
                    if ph.center_deg is not None:
                        center = np.full(B, ph.center_deg)
                    else:
                        t0, t1 = ph.decode_window
                        center, _ = weighted_circular_readout(dec_so_far, res_so_far, t0, t1)
                    rf_centers[ph.name] = center
                    inside = _ring_distance(net.angles_deg, center) <= ph.radius_deg
                    w0_np = np.where(inside, ph.w0_in, ph.w0_out).astype(np.float32)
                    if verbose:
                        print(f"t={t}: '{ph.name}' RF centre = {np.round(center, 1)} deg, "
                              f"{inside.sum(1)} of {N} neurons inside")
                w0_vec = torch.tensor(w0_np, device=dev, dtype=dt)
                neuron.w0 = w0_vec
                neuron.I_TC = (w0_vec ** 2 / (1 - a ** 2)) if couple_itc else itc_ref
            if record:
                w0_rec[:, t] = w0_np

            rec_e = spk_e_prev @ net.W_EE.T + spk_i_prev.sum(1, keepdim=True) * net.w_ie
            rec_i = spk_e_prev.sum(1, keepdim=True) * net.w_ei + spk_i_prev.sum(1, keepdim=True) * net.w_ii
            bg_e, noise_e = net._poisson_drive(torch.full((B, N), net._bg_rate_hz, device=dev, dtype=dt), net.bg_weight_e)
            bg_i, _ = net._poisson_drive(torch.full((B, net.n_inhib), net._bg_rate_hz, device=dev, dtype=dt), net.bg_weight_i)
            stim_e, stim_spk = net._poisson_drive(stim_rate_e[:, t], net.stim_weight_e)

            state_e, spk_e = neuron.step(state_e, rec_e + stim_e + bg_e + float(pulse_amp[t]))
            state_i, spk_i = net.i_neuron.step(state_i, rec_i.expand(-1, net.n_inhib) + bg_i)
            state_i, spk_i, refrac_i = net._apply_refractory(state_i, spk_i, refrac_i, net.refrac_steps_i)

            spk_np = spk_e.cpu().numpy()
            pop_spikes[:, t] = spk_np.sum(1)
            pop_spikes_i[:, t] = spk_i.sum(1).cpu().numpy()
            for name in ro_at.get(t, ()):
                ro_counts[name] += spk_np
            d, r = decode_population_vector(spk_np, net.angles_deg)
            dec_so_far[:, t], res_so_far[:, t] = d, r

            if record:
                spk_e_rec.append(spk_e.to(torch.uint8)); spk_i_rec.append(spk_i.to(torch.uint8))
                stim_rec.append(stim_spk.clamp(max=255).to(torch.uint8))
                noise_rec.append(noise_e.clamp(max=255).to(torch.uint8))
            spk_e_prev, spk_i_prev = spk_e, spk_i
    finally:
        neuron.w0, neuron.I_TC = old_w0, old_itc   # leave the net as we found it

    readouts = {}
    for name, true_deg, t0, t1 in pipe.readouts:
        ang, strength = weighted_circular_readout(dec_so_far, res_so_far, t0, t1)
        readouts[name] = dict(decoded=ang, strength=strength, true=true_deg,
                              error=circular_error_deg(ang, true_deg),
                              frac_active=(ro_counts[name] > 0).mean(1),          # fraction of E neurons that fire
                              rate_hz=ro_counts[name].mean(1) / max(t1 - t0, 1) * 1000,  # mean E rate
                              # share of readout spikes fired by neurons within ±42.5 deg of the true cue
                              on_cue=_on_cue_share(ro_counts[name], net.angles_deg, true_deg))

    stack = lambda lst: torch.stack(lst, 1) if record else None
    return dict(
        spk_e=stack(spk_e_rec), spk_i=stack(spk_i_rec),
        stim_spk_e=stack(stim_rec),
        noise_spk_e=stack(noise_rec),
        decoded_deg=dec_so_far, resultant_length=res_so_far,
        w0_map=w0_rec, rf_centers=rf_centers, readouts=readouts, pop_spikes=pop_spikes, n_excit=N,
        pop_spikes_i=pop_spikes_i, n_inhib=net.n_inhib,
        angles_deg=net.angles_deg, T=T, couple_itc=couple_itc)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CUE_COLOR = "#0072B2"         # Okabe-Ito blue (same as plots.py)
DISTRACTOR_COLOR = "#D55E00"  # Okabe-Ito vermillion (same as plots.py)
DECODED_COLOR = "#D1172B"     # red, dashed: the angle the network reports at readout
INK = "#222222"
MUTED = "#8c8c8c"


def _w0_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # diverging: negative (bistable, can hold) = blue, 0 = neutral gray, positive = orange
    return LinearSegmentedColormap.from_list("w0", ["#0072B2", "#d9d9d9", "#E69F00"])


READOUT_COLOR = "#111111"
CUE_GREEN = "#1A9641"         # compact (abstract) figure: cue
DISTRACTOR_RED = "#D62728"    # compact (abstract) figure: distractor
ERROR_RED = "#B2182B"         # compact (abstract) figure: "3° error"
FORGET_COLOR = "#6b6b6b"
PULSE_COLOR = "#7b3fa0"


def plot_pipeline(r, pipe, trial=0, title="", w0_lim=1.0, show_inhib=False,
                  n_noise_show=80, fs=16, figsize=(15, 5.0), fig=None, rect=None, compact_labels=False, marker_scale=1.0, label_fs=None,
                  w0_range=None, time_in_s=False):
    """Full-pipeline figure for one trial (top to bottom):
        1. ALL input to the ring as spikes: background noise (gray, a random
           subset of neurons), cue (blue) and distractor (vermillion), by preferred angle
        2. E raster by preferred angle, drawn on top of a faded heat map of the
           w0 every neuron has at every moment (blue = -1, gray = 0, orange = +1)
        3. decoded angle over time (opacity = resultant length)
    Cue / distractor / readout / forget / negative-pulse timings are vertical
    lines through every panel, labelled above the top panel; negative pulses are hatched
    purple areas on the input and E panels.
    fig / rect: draw into an existing figure, panels inside rect = (left, bottom, right, top)
    in figure fractions (the timing labels go above `top`). compact_labels: shorter labels
    (no angles on cue / distractor, no readout / forget names) for small figures: no
    background-noise note, "distractor", errors written "3° error", the start pulse labelled
    "negative pulse" (purple) and forget windows "forget" (black, like readout).
    w0_range = (lo, hi): range of the w0 heat map and its colour bar (default: -w0_lim .. +w0_lim).
    compact_labels also writes "noise" (gray) on the input panel with smaller, dimmer noise dots, and
    draws the forget windows in purple like the start pulse. time_in_s: time axis in seconds."""
    import matplotlib.pyplot as plt
    ev, T, ang = pipe.events, r["T"], r["angles_deg"]
    cue_c = CUE_GREEN if compact_labels else CUE_COLOR               # abstract figure: green cue, red distractor
    dis_c = DISTRACTOR_RED if compact_labels else DISTRACTOR_COLOR
    A = lambda a: float(np.atleast_1d(a)[trial]) if np.ndim(a) else float(a)   # per-trial angle
    n_rows = 4 if show_inhib else 3
    ratios = [1.25, 1.45, 1.5] + ([0.8] if show_inhib else [])
    with plt.rc_context({"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs,
                         "ytick.labelsize": fs}):
        if fig is None:
            fig = plt.figure(figsize=figsize)
            rect = rect or (0.075, 0.15, 0.925, 0.80)
        left, bottom, right, top = rect
        gs = fig.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.10,
                              left=left, right=right, bottom=bottom, top=top)
        axes = [fig.add_subplot(gs[0])]
        axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in range(1, n_rows)]
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        ax_in, ax_e, ax_d = axes[:3]
        for ax in (ax_in, ax_e):                              # negative pulses: hatched purple
            for _, p0_, p1_, _ in pipe.pulses:
                ax.axvspan(p0_, p1_, facecolor="none", hatch="////", edgecolor=PULSE_COLOR, lw=0,
                           alpha=0.55, zorder=4)

        # ---- 1. input spikes: noise + cue + distractor --------------------
        if "noise_spk_e" in r:
            rng = np.random.default_rng(0)
            n_show = min(40 if compact_labels else n_noise_show, len(ang))      # compact figure: fewer noise rows
            show = np.sort(rng.choice(len(ang), size=n_show, replace=False))
            nz = r["noise_spk_e"][trial].cpu().numpy()[:, show]
            tn, nn = np.nonzero(nz)
            n_size, n_alpha = (0.5, 0.08) if compact_labels else (1.2, 0.25)   # compact figure: smaller, dimmer
            ax_in.scatter(tn, ang[show][nn], s=n_size * marker_scale, c=MUTED, alpha=n_alpha, linewidths=0, rasterized=True)
        stim = r["stim_spk_e"][trial].cpu().numpy()
        t_idx, n_idx = np.nonzero(stim)
        is_d = np.zeros_like(t_idx, dtype=bool)
        for _, _, t0, t1 in pipe.distractors:
            is_d |= (t_idx >= t0) & (t_idx < t1)
        ax_in.scatter(t_idx[~is_d], ang[n_idx[~is_d]], s=2.5 * marker_scale, c=cue_c, linewidths=0, rasterized=True)
        ax_in.scatter(t_idx[is_d], ang[n_idx[is_d]], s=2.5 * marker_scale, c=dis_c, linewidths=0, rasterized=True)
        lfs = label_fs or fs
        ax_in.set_ylabel("input", fontsize=lfs)
        if not compact_labels:
            ax_in.text(0.005, 0.97, "gray: background noise",
                       transform=ax_in.transAxes, ha="left", va="top", fontsize=fs - 2, color="#555555",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=6)

        # ---- 2. E raster on the faded w0 heat map -------------------------
        lo, hi = w0_range if w0_range is not None else (-w0_lim, w0_lim)
        w0_cm = _w0_cmap()
        if w0_range is not None:                      # keep the same colours, but only the lo..hi part of the scale
            from matplotlib.colors import LinearSegmentedColormap
            w0_cm = LinearSegmentedColormap.from_list(
                "w0_part", [w0_cm((v + w0_lim) / (2 * w0_lim)) for v in np.linspace(lo, hi, 64)])
        im = ax_e.imshow(r["w0_map"][trial].astype(np.float32).T, aspect="auto", origin="lower",
                         extent=[0, T, 0, 360], cmap=w0_cm, vmin=lo, vmax=hi,
                         interpolation="nearest", alpha=0.55, zorder=0)
        spk = r["spk_e"][trial].cpu().numpy()
        t_idx, n_idx = np.nonzero(spk)
        ax_e.scatter(t_idx, ang[n_idx], s=0.9 * marker_scale, c=INK, linewidths=0, rasterized=True, zorder=2)
        for name, a, t0, t1 in ([] if compact_labels else pipe.readouts):   # true cue (blue), decoded (red, dashed)
            ax_e.hlines(A(a), t0, t1, color=CUE_COLOR, lw=2.0 * marker_scale ** 0.5, zorder=3)
            ax_e.hlines(float(r["readouts"][name]["decoded"][trial]), t0, t1, color=DECODED_COLOR,
                        lw=2.0 * marker_scale ** 0.5, ls=(0, (4, 2)), zorder=3)
        ax_e.set_ylabel("E\nneurons" if compact_labels else "E neurons", fontsize=lfs,
                        linespacing=0.95)                    # compact figure: two lines, panels are short
        cax = ax_e.inset_axes([1.012, 0.0, 0.012, 1.0])
        ticks = [lo, hi] if w0_range is not None else [lo, 0, hi]
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        if compact_labels:      # abstract figure: horizontal w_0 right next to the bar, between the two tick labels
            cb.ax.annotate("$w_0$", xy=(1.0, 0.5), xycoords="axes fraction", xytext=(6, 0),
                           textcoords="offset points", ha="left", va="center", fontsize=fs)
        else:
            cb.set_label("w0", fontsize=fs)
        cb.ax.tick_params(labelsize=fs - 1)
        cb.ax.set_yticklabels([(f"{t:+g}" if t > 0 else f"{t:g}").replace("-", "−") for t in ticks])

        # ---- 3. decoded angle (unchanged) ---------------------------------
        dec, res = r["decoded_deg"][trial], r["resultant_length"][trial]
        keep = res > 0
        tt = np.arange(T)[keep]
        ax_d.scatter(tt, dec[keep], s=4 * marker_scale, c=INK, alpha=np.clip(res[keep], 0.05, 1), linewidths=0, rasterized=True)
        for name, a, t0, t1 in pipe.readouts:
            ro = r["readouts"][name]
            true_a, dec_a = A(a), float(ro["decoded"][trial])
            if compact_labels:                               # one green dotted line at the cue angle
                ax_d.hlines(true_a, t0, t1, color=cue_c, lw=3.0 * marker_scale ** 0.5, ls=(0, (1, 1)), zorder=4)
            else:
                ax_d.hlines(true_a, t0, t1, color=CUE_COLOR, lw=2.4 * marker_scale ** 0.5, zorder=4)
                ax_d.hlines(dec_a, t0, t1, color=DECODED_COLOR, lw=2.4 * marker_scale ** 0.5, ls=(0, (4, 2)), zorder=4)
            xm = 0.5 * (t0 + t1)
            if abs(dec_a - true_a) > 15:                     # double arrow from cue to decoded angle
                ax_d.annotate("", xy=(xm, dec_a), xytext=(xm, true_a), zorder=5,
                              arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4, shrinkA=0, shrinkB=0))
            if compact_labels:   # "3°" over a smaller "error", in red, in the empty half of the panel
                box_ = dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85)
                if true_a < 180:     # dots low: number at the top, "error" under it
                    ax_d.text(xm, 352, f"{ro['error'][trial]:.0f}°", ha="center", va="top", fontsize=fs - 1,
                              color=ERROR_RED, bbox=box_, zorder=6)
                    ax_d.annotate("error", (xm, 352), xytext=(0, -(fs + 0.5)), textcoords="offset points",
                                  ha="center", va="top", fontsize=fs - 3, color=ERROR_RED, zorder=6)
                else:                # dots high: "error" at the bottom, number above it
                    ax_d.annotate(f"{ro['error'][trial]:.0f}°", (xm, 8), xytext=(0, fs - 2.5),
                                  textcoords="offset points", ha="center", va="bottom", fontsize=fs - 1,
                                  color=ERROR_RED, bbox=box_, zorder=6)
                    ax_d.text(xm, 8, "error", ha="center", va="bottom", fontsize=fs - 3, color=ERROR_RED, zorder=6)
            else:
                ax_d.text(t1 - 12, 352, f"error {ro['error'][trial]:.0f}°",
                          ha="right", va="top", fontsize=fs - 1, color=INK,
                          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax_d.set_ylabel("decoded", fontsize=lfs)

        for ax in axes[:3]:
            ax.set_ylim(0, 360); ax.set_yticks([90, 270]); ax.set_yticklabels(["90°", "270°"])
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        if show_inhib:
            ax_i = axes[3]
            si = r["spk_i"][trial].cpu().numpy()
            ti, ni = np.nonzero(si[:, :100])
            ax_i.scatter(ti, ni, s=0.5, c=MUTED, linewidths=0, rasterized=True)
            ax_i.set_ylabel("I neurons\n(first 100)")
            for s_ in ("top", "right"): ax_i.spines[s_].set_visible(False)
        if time_in_s:
            from matplotlib.ticker import FuncFormatter, MultipleLocator
            axes[-1].xaxis.set_major_locator(MultipleLocator(1000))
            axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
            axes[-1].set_xlabel("time (s)")
        else:
            axes[-1].set_xlabel("time (ms)")
        axes[-1].set_xlim(0, T)

        # ---- timing lines through every panel, labels on top --------------
        p0, pN = axes[0].get_position(), axes[-1].get_position()
        H_in = (p0.y1 - pN.y0) * fig.get_figheight()             # overlay height in inches
        pt = lambda x: x / 72.0 / H_in                           # points -> overlay axes fraction
        ov = fig.add_axes([p0.x0, pN.y0, p0.width, p0.y1 - pN.y0], facecolor="none")
        ov.set_xlim(0, T); ov.set_ylim(0, 1); ov.axis("off")
        row1, row2 = pt(3), pt(fs * (1.5 if compact_labels else 1.75))   # label baselines (pts above the top panel)

        def vline(x, color, ls, top, lw=1.3 * marker_scale ** 0.5, z=5):
            ov.plot([x, x], [0, 1 + top], color=color, ls=ls, lw=lw, alpha=0.9, clip_on=False, zorder=z)

        lab_size = fs - 1 if compact_labels else None                   # compact figure: smaller, not bold
        lab_weight = "normal" if compact_labels else "bold"

        def label(x0, x1, text, color, base, size, edge=True):
            xm = (x0 + x1) / 2
            ha = "center"
            if xm < 0.04 * T: ha, xm = "left", x0
            ov.text(xm, 1 + base, text, ha=ha, va="bottom", color=color, fontsize=lab_size or size,
                    fontweight=lab_weight, clip_on=False, linespacing=1.15, zorder=8,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))

        forgets = pipe.meta.get("forgets", [(ev["f_on"], ev["f_off"])])
        in_forget = lambda p0_, p1_: any(abs(p0_ - f0) < 1 and abs(p1_ - f1) < 1 for f0, f1 in forgets)
        if compact_labels:                                   # start pulse (not a forget window): purple label + lines,
            for _, p0_, p1_, _ in pipe.pulses:               # drawn under the cue lines so a cue that follows directly shows
                if in_forget(p0_, p1_):
                    continue
                vline(p0_, PULSE_COLOR, "-", row2, z=4.5); vline(p1_, PULSE_COLOR, "-", row2, z=4.5)
                label(p0_, p1_, "negative pulse", PULSE_COLOR, row2, fs)
        for name, a, t0, t1 in pipe.cues:
            vline(t0, cue_c, ":", row1); vline(t1, cue_c, ":", row1)
            label(t0, t1, ("cue" if compact_labels else f"{name}: {A(a):.0f}°"), cue_c, row1, fs)
        for name, a, t0, t1 in pipe.distractors:
            vline(t0, dis_c, ":", row1); vline(t1, dis_c, ":", row1)
            label(t0, t1, ("distractor" if compact_labels else f"{name}: {A(a):.0f}°"), dis_c, row1, fs)
        for name, a, t0, t1 in pipe.readouts:
            vline(t0, READOUT_COLOR, "-", row2); vline(t1, READOUT_COLOR, "-", row2)
            label(t0, t1, "readout" if compact_labels else name, READOUT_COLOR, row2, fs)
        # forget windows (w0 and / or negative pulse): their own label row on top
        w0f = pipe.meta.get("w0_forget")
        fcol = PULSE_COLOR if compact_labels else FORGET_COLOR       # compact figure: purple, like the start pulse
        row3 = row2                                             # forget windows never overlap a readout
        for f0, f1 in forgets:
            what = []
            if pipe.meta.get("forget_keep_rf"):
                what.append("RF kept")
            elif w0f is not None and abs(w0f - pipe.meta.get("w0_mem", -1.0)) > 1e-9:
                what.append("w0 = " + (f"{w0f:+g}" if w0f else "0"))
            if any(abs(p0_ - f0) < 1 and abs(p1_ - f1) < 1 for _, p0_, p1_, _ in pipe.pulses):
                what.append("neg. pulse")
            vline(f0, fcol, "-", row3); vline(f1, fcol, "-", row3)
            what = [w for w in what if w != "neg. pulse"]     # the pulse is the hatched area
            label(f0, f1, "forget" + ("" if compact_labels or not what else " (" + ", ".join(what) + ")"),
                  fcol, row3, fs)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=fs + 2, y=0.995)
    return fig


def pulse_report(r, pipe, margin=100):
    """Mean E firing rate (Hz) in the `margin` ms before, during and after each negative pulse."""
    out = {}
    for name, t0, t1, amp in pipe.pulses:
        rate = lambda a, b: float(r["spk_e"][:, max(a, 0):b].float().mean()) * 1000 if b > a else float("nan")
        out[f"{name} [{t0}-{t1} ms, I={amp:g}]"] = dict(
            before=rate(t0 - margin, t0), during=rate(t0, t1), after=rate(t1, t1 + margin))
    return out


def summarize(results):
    """results: {condition label: recs}. Returns a pandas-free text table of
    mean / median readout error per memory."""
    rows = ["condition".ljust(42) + "memory 1 err (mean / median)   memory 2 err (mean / median)"]
    for label, r in results.items():
        e1 = r["readouts"]["readout 1"]["error"]; e2 = r["readouts"]["readout 2"]["error"]
        rows.append(f"{label:<42}{e1.mean():6.1f}° / {np.median(e1):5.1f}°" + " " * 15 +
                    f"{e2.mean():6.1f}° / {np.median(e2):5.1f}°")
    return "\n".join(rows)
