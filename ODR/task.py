"""
task.py (ODR)

Builds oculomotor delayed-response trials: fixation -> cue -> delay
(-> optional distractor) -> response/readout window. Output is the external
RATE tensor (Hz, not current -- see network_architecture.py's module
docstring, "INPUT IS NOW SPIKING") fed to RingSNN.run(); everything about
WHEN things happen is tracked in a small TrialTiming object so plots.py can
draw the right markers without every caller re-deriving step indices.

Durations are given in ms and converted to steps 1:1 (1 network step =
1 ms, same convention as the rest of this codebase -- see
network_architecture.py's dt_eff note).
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TrialTiming:
    t_fix: int
    t_cue_start: int
    t_cue_end: int
    t_delay_end: int
    t_distractor_start: int = None
    t_distractor_end: int = None
    T: int = None  # total steps

    @property
    def t_response_start(self):
        return self.t_delay_end  # readout window = last part of the delay; no separate input epoch


def _angle_profile(angles_deg, center_deg, width_deg, peak_rate_hz):
    """The spatial shape of the cue/distractor drive: every E neuron gets
    SOME target rate, scaled by how close its preferred angle is to
    center_deg (a Gaussian falloff -- Compte's own code uses a flat boxcar
    window of stim_width_deg instead, full strength inside the window and
    exactly zero outside; the Gaussian is a deliberate, minor deviation for
    smoothness, not a fidelity target). sigma is set so ~width_deg is the
    effective FWHM. Returns a RATE in Hz (peak_rate_hz at center_deg, falling
    off with distance) -- see network_architecture.py's _poisson_drive for
    how a rate becomes actual spikes."""
    sigma = max(width_deg / 2.355, 1e-3)  # width_deg ~ FWHM
    diff = (angles_deg - center_deg + 180.0) % 360.0 - 180.0
    profile = peak_rate_hz * np.exp(-0.5 * (diff / sigma) ** 2)
    # Hard cutoff at 2.5 sigma: with a genuinely Poisson-driven rate (rather
    # than a smooth current), a Gaussian's long tail means every neuron on
    # the ring gets a small but nonzero chance of a spurious "cue" spike
    # every step -- negligible individually, but summed over hundreds of
    # steps and hundreds of far-away neurons it adds real, spatially
    # nonspecific drive. Truncating keeps the cue/distractor genuinely
    # localized, matching Compte's own boxcar window in spirit (a hard
    # edge), just still smooth-shouldered instead of flat-topped.
    profile = np.where(np.abs(diff) <= 2.5 * sigma, profile, 0.0)
    return profile


def build_odr_trial(n_excit, angles_deg, cue_angle_deg,
                     t_fix=100, t_cue=250, t_delay=1500,
                     stim_width_deg=40.0, stim_rate_hz=6000.0,
                     distractor_angle_deg=None, distractor_onset=None, t_distractor=200,
                     distractor_width_deg=40.0, distractor_rate_hz=6000.0,
                     batch_size=1, device="cpu", dtype=torch.float):
    """
    Returns (stim_rate, timing).

    stim_rate : (batch_size, T, n_excit) tensor, in Hz -- the cue AND the
        distractor (if any) already combined into one rate profile (they
        never overlap in time, so summing is safe). Feed straight to
        RingSNN.run(stim_rate_e=...) -- it draws the actual Poisson spikes.
        Same cue_angle_deg repeated across the batch dimension (pass an
        array of cue_angle_deg, one per batch element, for a batch of
        DIFFERENT trials -- see build_odr_batch below).
    timing : TrialTiming, so plots.py can draw t_cue_start/end,
             t_delay_end etc. without recomputing them.
    """
    T = t_fix + t_cue + t_delay
    stim_rate = np.zeros((T, n_excit), dtype=np.float32)

    cue_profile = _angle_profile(angles_deg, cue_angle_deg, stim_width_deg, stim_rate_hz)
    stim_rate[t_fix:t_fix + t_cue, :] = cue_profile[None, :]

    t_distractor_start = t_distractor_end = None
    if distractor_angle_deg is not None:
        if distractor_onset is None:
            distractor_onset = t_fix + t_cue + t_delay // 2  # midway through the delay, by default
        t_distractor_start = distractor_onset
        t_distractor_end = min(distractor_onset + t_distractor, T)
        distr_profile = _angle_profile(angles_deg, distractor_angle_deg, distractor_width_deg, distractor_rate_hz)
        stim_rate[t_distractor_start:t_distractor_end, :] = distr_profile[None, :]

    stim_rate = torch.tensor(stim_rate, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

    timing = TrialTiming(t_fix=t_fix, t_cue_start=t_fix, t_cue_end=t_fix + t_cue,
                          t_delay_end=T, t_distractor_start=t_distractor_start,
                          t_distractor_end=t_distractor_end, T=T)
    return stim_rate, timing


def build_odr_batch(n_excit, angles_deg, cue_angles_deg, **kwargs):
    """Same as build_odr_trial but one DIFFERENT cue angle per batch
    element (cue_angles_deg: array-like of length batch_size). Useful for
    e.g. testing all 8 standard target locations in one batched run.
    NOTE: this repeats the SAME distractor_angle_deg (if given) across all
    batch elements -- pass distractor angles per-trial by looping
    build_odr_trial yourself if you need those to vary too (the size/w0
    sweeps in run_odr_sweep.py do this, since each trial there needs its
    own random cue AND random distractor angle)."""
    cue_angles_deg = np.asarray(cue_angles_deg)
    batch_size = len(cue_angles_deg)
    kwargs = dict(kwargs)
    device = kwargs.pop("device", "cpu")
    dtype = kwargs.pop("dtype", torch.float)
    kwargs.pop("batch_size", None)

    trials, timing = [], None
    for angle in cue_angles_deg:
        stim_rate_1, timing = build_odr_trial(n_excit, angles_deg, float(angle),
                                               batch_size=1, device=device, dtype=dtype, **kwargs)
        trials.append(stim_rate_1)
    stim_rate = torch.cat(trials, dim=0)
    return stim_rate, timing


EIGHT_TARGETS_DEG = np.arange(0, 360, 45)  # the standard 8-location ODR set, 0/45/.../315
