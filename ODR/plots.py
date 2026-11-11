"""
plots.py (ODR)

Raster plots and summary figures, following the same visual conventions as
HSD_take_2/plots.py and MNIST/plots.py already in this codebase: crimson
dotted line = cue boundary, steelblue dotted line = readout, scatter
rasters sorted so structure is visible, fig.tight_layout() throughout.

Category colors (noise / cue / distractor, wherever a plot needs to tell
them apart) use the Okabe-Ito colorblind-safe set: noise = neutral gray
(it's the "background" category), cue = blue (#0072B2), distractor =
vermillion (#D55E00) -- a validated colorblind-safe pair, not just "looks
different to me".

The one ODR-specific raster addition is sorting the E-population raster by
PREFERRED ANGLE (not neuron index -- they're the same here since the ring
is built in angle order, but keeping this explicit protects against ever
reordering the population later) so the bump is visible as a horizontal
band, not a diffuse scatter.
"""

import numpy as np
import matplotlib.pyplot as plt

NOISE_COLOR = "#8c8c8c"       # neutral gray -- the "background" category
CUE_COLOR = "#0072B2"         # Okabe-Ito blue
DISTRACTOR_COLOR = "#D55E00"  # Okabe-Ito vermillion


def _mark_timing(ax, timing):
    ax.axvline(timing.t_cue_start, color="crimson", ls=":", lw=1.5, label="cue on")
    ax.axvline(timing.t_cue_end, color="darkorange", ls=":", lw=1.5, label="cue off")
    ax.axvline(timing.t_delay_end, color="steelblue", ls=":", lw=1.5, label="readout")
    if timing.t_distractor_start is not None:
        ax.axvline(timing.t_distractor_start, color="purple", ls="--", lw=1.2, label="distractor on")
        ax.axvline(timing.t_distractor_end, color="purple", ls=":", lw=1.0, label="distractor off")


def plot_input_raster(recs, timing, run_name, cue_angle_deg, trial_idx=0, n_noise_show=150):
    """Everything entering the E population from outside, as actual spikes,
    on one raster -- background noise (gray), cue spikes (blue), distractor
    spikes (vermillion). All three are now literally the same kind of
    event (a Poisson-drawn unitary spike converted to a current kick -- see
    network_architecture.py's module docstring, "INPUT IS NOW SPIKING") so
    they belong on one plot, distinguished only by rate/timing/color, not
    by mechanism.

    Background noise is shown for a random SUBSET of n_noise_show E
    neurons (and a further n_noise_show//3 I neurons on a second axis) --
    showing all of them would be close to solid, because the aggregate
    background rate is genuinely very high (n_extern_poisson * bg_rate_hz
    ~= 1.4kHz per neuron, see network_architecture.py's module docstring).
    Cue/distractor spikes are NOT subsampled -- they're localized to a
    neighborhood of the ring, so they're naturally much sparser, and they're
    the actual signal of interest here."""
    stim_spk = recs["stim_spk_e"][trial_idx].detach().cpu().numpy()   # (T, n_excit) counts
    noise_e = recs["noise_spk_e"][trial_idx].detach().cpu().numpy()   # (T, n_excit)
    noise_i = recs["noise_spk_i"][trial_idx].detach().cpu().numpy()   # (T, n_inhib)
    T = recs["T"]
    n_excit = stim_spk.shape[1]
    n_inhib = noise_i.shape[1]

    rng = np.random.default_rng(0)
    show_e = np.sort(rng.choice(n_excit, size=min(n_noise_show, n_excit), replace=False))
    show_i = np.sort(rng.choice(n_inhib, size=min(max(n_noise_show // 3, 1), n_inhib), replace=False))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5]})

    # --- row 1: E population -- background noise (gray, subsampled) + ------
    # cue/distractor spikes (colored, full resolution), split by epoch using
    # the timing markers (cue window vs. distractor window -- they never
    # overlap in time, so this coloring is unambiguous even though
    # stim_spk_e itself doesn't tag which event produced each spike).
    times_e, neurons_e = np.nonzero(noise_e[:, show_e])
    axes[0].scatter(times_e, show_e[neurons_e], s=1.5, c=NOISE_COLOR, alpha=0.25,
                     label=f"background noise ({len(show_e)} of {n_excit} E shown)")

    t_idx, n_idx = np.nonzero(stim_spk)
    if timing.t_distractor_start is not None:
        is_distr = (t_idx >= timing.t_distractor_start) & (t_idx < timing.t_distractor_end)
    else:
        is_distr = np.zeros_like(t_idx, dtype=bool)
    axes[0].scatter(t_idx[~is_distr], n_idx[~is_distr], s=5, c=CUE_COLOR, label="cue spikes")
    if is_distr.any():
        axes[0].scatter(t_idx[is_distr], n_idx[is_distr], s=5, c=DISTRACTOR_COLOR, label="distractor spikes")

    axes[0].axhline(cue_angle_deg / 360.0 * n_excit, color="black", ls="-", lw=0.7, alpha=0.5,
                     label="true cue angle")
    axes[0].set_ylabel("E neuron (sorted by\npreferred angle)")
    axes[0].set_yticks(np.linspace(0, n_excit, 5))
    axes[0].set_yticklabels([f"{d:.0f}°" for d in np.linspace(0, 360, 5)])
    axes[0].set_title(f"{run_name} -- all external input as spikes, cue at {cue_angle_deg:.0f}°")

    # --- row 2: I population background noise, subsampled -----------------
    times_i, neurons_i = np.nonzero(noise_i[:, show_i])
    axes[1].scatter(times_i, show_i[neurons_i], s=1.5, c=NOISE_COLOR, alpha=0.25,
                     label=f"background noise ({len(show_i)} of {n_inhib} I shown)")
    axes[1].set_ylabel(f"I neuron")
    axes[1].set_xlabel("step (ms)")

    for ax in axes:
        _mark_timing(ax, timing)
        ax.set_xlim(0, T)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    axes[1].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    return fig


def plot_odr_raster(recs, timing, run_name, cue_angle_deg, trial_idx=0, max_inhib=100):
    """3-row figure for ONE trial: E-population raster (sorted by preferred
    angle, so the bump is a visible band), I-population raster, and the
    decoded population-vector angle over time with the true cue angle
    marked -- the single most important figure per trial: does the decoded
    line sit on the true angle and STAY there through the delay."""
    spk_e = recs["spk_e"][trial_idx].detach().cpu().numpy()   # (T, n_excit)
    spk_i = recs["spk_i"][trial_idx].detach().cpu().numpy()   # (T, n_inhib)
    decoded = recs["decoded_deg"][trial_idx]                   # (T,)
    resultant = recs["resultant_length"][trial_idx]            # (T,)
    T = recs["T"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 2]})

    # --- row 1: E population raster, sorted by preferred angle --------
    times_e, neurons_e = np.nonzero(spk_e)
    axes[0].scatter(times_e, neurons_e, s=2, c="black", alpha=0.5)
    axes[0].axhline(cue_angle_deg / 360.0 * spk_e.shape[1], color="crimson", ls="-", lw=0.8, alpha=0.6,
                     label="true cue angle")
    axes[0].set_ylabel("E neuron (sorted by preferred angle)")
    axes[0].set_yticks(np.linspace(0, spk_e.shape[1], 5))
    axes[0].set_yticklabels([f"{d:.0f}°" for d in np.linspace(0, 360, 5)])
    axes[0].set_title(f"{run_name} -- cue at {cue_angle_deg:.0f}° (trial {trial_idx})")

    # --- row 2: I population raster (subset) ----------------------------
    n_show = min(max_inhib, spk_i.shape[1])
    times_i, neurons_i = np.nonzero(spk_i[:, :n_show])
    axes[1].scatter(times_i, neurons_i, s=2, c="dimgray", alpha=0.5)
    axes[1].set_ylabel(f"I neuron\n(first {n_show})")

    # --- row 3: decoded angle over time ----------------------------------
    axes[2].scatter(np.arange(T), decoded, s=4, c=plt.cm.viridis(resultant), alpha=0.8)
    axes[2].axhline(cue_angle_deg, color="crimson", ls="-", lw=1.2, label="true cue angle")
    axes[2].set_ylim(0, 360)
    axes[2].set_ylabel("decoded angle (°)")
    axes[2].set_xlabel("step (ms)")
    axes[2].set_title("population-vector readout (color = resultant length, i.e. how peaked the bump is)")

    for ax in axes:
        _mark_timing(ax, timing)
        ax.set_xlim(0, T)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    axes[2].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    return fig


def plot_bump_profile_snapshots(recs, timing, run_name, cue_angle_deg, trial_idx=0,
                                 snapshot_steps=None, window=20):
    """Firing-rate-vs-preferred-angle 'bump shape' at a few chosen moments
    (default: just after cue onset, just before cue offset, mid-delay, and
    at readout) -- averaged over a small window of `window` steps around
    each snapshot, since single-step spike counts are too sparse to show a
    clean bump shape. This is the figure that answers "is there actually a
    localized bump, or just diffuse activity" at a glance."""
    spk_e = recs["spk_e"][trial_idx].detach().cpu().numpy()
    angles = recs["angles_deg"]
    T = recs["T"]

    if snapshot_steps is None:
        snapshot_steps = [
            min(timing.t_cue_start + window, T - 1),
            max(timing.t_cue_end - window, 0),
            (timing.t_cue_end + timing.t_delay_end) // 2,
            max(timing.t_delay_end - window, 0),
        ]
        labels = ["cue onset", "cue offset", "mid-delay", "readout"]
    else:
        labels = [f"t={s}" for s in snapshot_steps]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for step, label in zip(snapshot_steps, labels):
        lo, hi = max(0, step - window // 2), min(T, step + window // 2)
        rate = spk_e[lo:hi].mean(axis=0) * 1000.0  # spikes/window -> approx Hz (1 step = 1 ms)
        ax.plot(angles, rate, marker=".", ms=3, label=f"{label} (t={step})")
    ax.axvline(cue_angle_deg, color="crimson", ls=":", lw=1.2, label="true cue angle")
    ax.set_xlabel("preferred angle (°)")
    ax.set_ylabel("firing rate (Hz, approx.)")
    ax.set_title(f"{run_name} -- E-population bump shape over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_decode_timeline(recs, timing, run_name, cue_angle_deg, trial_idx=0):
    """Single-trace decoded-angle-over-time plot -- lighter-weight than the
    full 3-row plot_odr_raster, useful when you just want the readout
    trace for one trial without the two spike rasters (e.g. quick sanity
    checks inside a sweep)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    decoded = recs["decoded_deg"][trial_idx]
    resultant = recs["resultant_length"][trial_idx]
    T = recs["T"]
    ax.plot(np.arange(T), decoded, lw=1.0, color=CUE_COLOR, alpha=0.35)
    ax.scatter(np.arange(T), decoded, s=6, c=CUE_COLOR, alpha=np.clip(resultant, 0.05, 1.0))
    ax.axhline(cue_angle_deg, color="black", ls="--", lw=1.2, label="true cue angle")
    _mark_timing(ax, timing)
    ax.set_ylim(0, 360)
    ax.set_xlim(0, timing.T)
    ax.set_xlabel("step (ms)")
    ax.set_ylabel("decoded angle (°)")
    ax.set_title(f"{run_name} -- cue at {cue_angle_deg:.0f}°")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


W0_LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
# Fixed categorical order (dataviz skill's validated 8-hue theme, first 7 slots --
# blue/orange/aqua/yellow/magenta/green/violet, in that order, never cycled or
# reassigned by rank), one color per w0 value in run_odr_cross_sweep.py's
# W0_VALUES order. Validated for this exact 7-color subset via
# scripts/validate_palette.js (adjacent-pair CVD Delta E and normal-vision
# floor both pass; the contrast WARN on 3 of the 7 is why the legend always
# carries visible w0 = ... text labels, never color alone).


def plot_accuracy_by_w0_lines(sizes, w0_values, mean_err_by_w0, title, n_trials=None):
    """One line per w0 value, network size (log x) vs. mean |decode error| --
    the cross-product view of the size sweep and w0 sweep (run
    separately elsewhere): does the working w0 range found at full size
    (see run_odr_sweep.py / the project notes) hold up as the network also
    shrinks, or do some w0 values only work at large N? mean_err_by_w0 is a
    dict {w0: [mean error per size, in `sizes` order]} -- see
    run_odr_cross_sweep.py."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.asarray(sizes, dtype=float)
    for i, w0 in enumerate(w0_values):
        color = W0_LINE_COLORS[i % len(W0_LINE_COLORS)]
        series = mean_err_by_w0[w0] if w0 in mean_err_by_w0 else mean_err_by_w0[str(w0)]
        y = np.asarray(series, dtype=float)
        ax.plot(x, y, marker="o", ms=6, lw=2, color=color, label=f"w0 = {w0:g}")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlabel("network size (N excitatory)")
    ax.set_ylabel("mean |decode error| at readout (°)")
    n_note = f" (n={n_trials} trials/point)" if n_trials else ""
    ax.set_title(title + n_note)
    ax.legend(fontsize=8, loc="upper left", ncol=2, title="w0", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_accuracy_sweep(x_values, mean_err_deg, std_err_deg, xlabel, title,
                         log_x=False, n_trials=None):
    """Generic sweep summary: mean +/- std of |decode error| at readout,
    one point per x_values entry (a network size, or a w0 value), each
    averaged over n_trials independent trials (random cue angle +
    random distractor angle each trial -- see run_odr_sweep.py). Used for
    BOTH the size sweep and the w0 sweep -- same shape of result, same
    honest way to show it: a line with error bars, not a bar chart (the
    x-axis is ordered/continuous in both cases, so a line reads the trend
    correctly; a bar chart would imply unordered categories)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.asarray(x_values, dtype=float)
    mean_err = np.asarray(mean_err_deg, dtype=float)
    std_err = np.asarray(std_err_deg, dtype=float)
    ax.errorbar(x, mean_err, yerr=std_err, marker="o", ms=6, lw=1.5, capsize=4,
                color=CUE_COLOR, ecolor=CUE_COLOR, elinewidth=1, alpha=0.9)
    ax.axhline(0, color="#8c8c8c", ls="-", lw=0.8, zorder=0)
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("|decode error| at readout (°), mean ± std")
    n_note = f" (n={n_trials} trials/point)" if n_trials else ""
    ax.set_title(title + n_note)
    fig.tight_layout()
    return fig
