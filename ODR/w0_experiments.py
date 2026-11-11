"""
w0_experiments.py (ODR)

Batch experiments on top of w0_gating.py, used by ODR_w0_gating_experiments.ipynb:

  1. phase sweep   -- gaps between all phases, delay shifted by 0..100 ms:
                      does forgetting still depend on the volley phase?
  2. overlap       -- cue-distractor separation from 180 deg down to 10 deg,
                      per network setup (no RF / RF gate / RF gate with w0_out > 0)
  3. repeats       -- many cue -> distractor -> readout -> forget loops in one
                      trial: does the error grow, and is each memory independent
                      of the previous one?

Every experiment returns a pandas DataFrame with one row per (trial, memory);
the notebook saves them as CSV.
"""

import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import w0_gating as g
from network_architecture import RingSNN
from run_odr_demo import DEFAULT_KWARGS, DEFAULT_STIM_RATE_HZ

N_EXCIT, N_INHIB = 1024, 256
I_MODEL = "lif"            # I neurons: "lif" = the MNIST / N-MNIST LIF (default since 2026-10-22);
                           # "lif_syn" = the LIF of every ODR result before 2026-10-22
RF_RADIUS = 2.5 * 40.0 / 2.355          # 42.5 deg: RF half-width = cue / distractor half-width

# one fixed style per case, used by every plot (colour, marker, line style).
# Okabe-Ito hues for the RF cases, grays + dashed lines for the no-RF cases.
STYLES = {
    "RF out 0, forget w0 0":            ("#009E73", "^", "-"),
    "RF out 0, forget w0 +1":           ("#0072B2", "o", "-"),
    "RF out +1, forget w0 0":           ("#E69F00", "v", "-"),
    "RF out +1, forget w0 +1":          ("#56B4E9", "s", "-"),
    "RF out 0, pulses":                 ("#D55E00", "D", "-"),
    "RF out 0, pulses + forget w0 0":   ("#882255", "h", "-"),
    "RF out 0, pulses + forget w0 +1":  ("#CC79A7", "P", "-"),
    "no RF, forget w0 +1":              ("#333333", "X", "--"),
    "no RF, pulses":                    ("#9a9a9a", "*", "--"),
}
_FALLBACK = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def style(name, i=0):
    """(colour, marker, line style) for a case name."""
    if name in STYLES:
        return STYLES[name]
    return (_FALLBACK[i % len(_FALLBACK)], MARKERS[i % len(MARKERS)], "--" if name.startswith("no RF") else "-")


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def run_sequence(cue, distr, n_trials, *, seed=0, record=False, **kw):
    """Build and run one batch. cue / distr: (L,) or (n_trials, L) in deg.
    Extra keywords go to w0_gating.build_sequence. Returns (pipe, recs, DataFrame)."""
    kw = dict(kw)
    w0_mem = kw.get("w0_mem", -1.0)
    kw.setdefault("stim_rate_hz", DEFAULT_STIM_RATE_HZ)
    net = RingSNN(N_EXCIT, N_INHIB, e_kwargs={"w0": w0_mem}, seed=seed, i_model=I_MODEL, **DEFAULT_KWARGS)
    pipe = g.build_sequence(N_EXCIT, net.angles_deg, cue, distr, batch_size=n_trials, **kw)
    with torch.no_grad():
        r = g.run_gated(net, pipe, couple_itc=True, w0_reference=w0_mem, record=record)
    return pipe, r, pd.DataFrame(g.sequence_table(r, pipe))


def _tag(df, **cols):
    for k, v in cols.items():
        df[k] = v
    return df


def phase_sweep(setups, shifts=range(0, 101, 10), n_trials=4, cue=(90.0, 270.0),
                distr=(210.0, 30.0), gap=100, seed=0):
    """Experiment 1. Same two-memory trial for every setup; the delay of BOTH
    memories is shifted by `shift` ms, which moves the forget window relative to
    the bump's volleys."""
    out = []
    for name, kw in setups.items():
        t = time.time()
        for s in shifts:
            _, _, df = run_sequence(cue, distr, n_trials, seed=seed, gap=gap, delay_shift=int(s), **kw)
            out.append(_tag(df, setup=name, shift_ms=int(s), gap_ms=gap))
        print(f"{name:40s} done in {time.time() - t:5.0f} s")
    return pd.concat(out, ignore_index=True)


def overlap_sweep(setups, separations=(180, 150, 120, 90, 60, 45, 30, 20, 10), n_batches=2,
                  n_trials=8, gap=100, max_shift=100, rng_seed=0, seed=0):
    """Experiment 2. Two memories per trial, random cue angles, the distractor at
    +-separation from its cue (random side). Every setup sees exactly the same
    angles and delay shifts (paired design)."""
    rng = np.random.default_rng(rng_seed)
    plan = []
    for sep in separations:
        for b in range(n_batches):
            cue = rng.uniform(0, 360, (n_trials, 2))
            side = rng.choice([-1.0, 1.0], (n_trials, 2))
            plan.append((sep, b, cue, (cue + side * sep) % 360, rng.integers(0, max_shift + 1, 2)))
    out = []
    for name, kw in setups.items():
        t = time.time()
        for sep, b, cue, distr, shift in plan:
            _, _, df = run_sequence(cue, distr, n_trials, seed=seed + b, gap=gap, delay_shift=shift, **kw)
            out.append(_tag(df, setup=name, separation_set_deg=sep, batch=b, gap_ms=gap))
        print(f"{name:40s} done in {time.time() - t:5.0f} s")
    return pd.concat(out, ignore_index=True)


def repeat_sweep(setups, n_loops=10, n_batches=2, n_trials=8, gap=100, max_shift=100,
                 distr_sep=(90.0, 180.0), rng_seed=1, seed=0):
    """Experiment 3. n_loops memories in one continuous trial, each with a random
    cue, a distractor 90-180 deg away (far, so the distractor is not the issue),
    and a random delay shift. Setups may override `gap` (the pause after each
    readout and after each forget)."""
    rng = np.random.default_rng(rng_seed)
    plan = []
    for b in range(n_batches):
        cue = rng.uniform(0, 360, (n_trials, n_loops))
        side = rng.choice([-1.0, 1.0], (n_trials, n_loops))
        distr = (cue + side * rng.uniform(*distr_sep, (n_trials, n_loops))) % 360
        plan.append((b, cue, distr, rng.integers(0, max_shift + 1, n_loops)))
    out = []
    for name, kw in setups.items():
        kw = dict(kw)
        gap_k = kw.pop("gap", gap)
        t = time.time()
        for b, cue, distr, shift in plan:
            _, _, df = run_sequence(cue, distr, n_trials, seed=seed + b, gap=gap_k, delay_shift=shift, **kw)
            out.append(_tag(df, setup=name, batch=b, gap_ms=gap_k))
        print(f"{name:40s} done in {time.time() - t:5.0f} s")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

FS = 13


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=FS - 1)


def _median_iqr(df, by, col):
    gb = df.groupby(by)[col]
    return gb.median(), gb.quantile(0.25), gb.quantile(0.75)


def _legend_right(ax):
    ax.legend(fontsize=FS - 2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _chance(ax):
    ax.axhline(90, color="#8c8c8c", ls=":", lw=1.2)
    ax.text(1.005, 90, "chance", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=FS - 2, color="#555555")


def plot_phase_sweep(df, memory=2):
    """Median error of one memory vs the delay shift, one line per case."""
    d = df[df.memory == memory]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, name in enumerate(d.setup.unique()):
        col, mk, ls = style(name, i)
        m, lo, hi = _median_iqr(d[d.setup == name], "shift_ms", "error_deg")
        ax.fill_between(m.index, lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(m.index, m.values, marker=mk, ms=8, lw=2.2, ls=ls, color=col, label=name)
    _chance(ax)
    ax.set_xlabel("delay shifted by (ms)", fontsize=FS)
    ax.set_ylabel(f"memory {memory} error (deg)\nmedian, band = IQR", fontsize=FS)
    ax.set_ylim(0, 185)
    _legend_right(ax)
    _style(ax)
    fig.tight_layout()
    return fig


def _mark_rf(ax, label=True):
    """Receptive field around the cue: gray band + gray edge bar, dotted black line at its centre."""
    ax.axvspan(0, RF_RADIUS, color="#bdbdbd", alpha=0.25, lw=0, zorder=0)
    ax.axvline(RF_RADIUS, color="#8c8c8c", lw=2.5, zorder=1)
    ax.axvline(0, color="black", ls=":", lw=1.6, zorder=1)
    if label:
        y = ax.get_ylim()[1]
        ax.text(RF_RADIUS / 2, y * 0.98, "receptive\nfield", ha="center", va="top", fontsize=FS - 2, color="#555555")
        ax.text(-1, y * 0.80, "RF centre", ha="left", va="top", fontsize=FS - 2, color="black", rotation=90)


def plot_overlap(df, memories=(1, 2), show_pull=True, dodge=0.0, merge_identical=False):
    """Error (and pull toward the distractor) vs cue-distractor separation.
    Separation 0 = the distractor sits on the cue, i.e. on the RF centre.
    dodge (deg) shifts each case slightly along x, so cases with identical values stay visible.
    merge_identical=True draws cases whose results for these memories are identical
    (e.g. cases that differ only after memory 1) as one line, with all their names in the legend."""
    if merge_identical:
        sel = df[df.memory.isin(memories)].sort_values(["batch", "separation_set_deg", "memory", "trial"])
        groups = {}
        for name in df.setup.unique():
            key = tuple(np.round(sel[sel.setup == name].decoded_deg.values, 6))
            groups.setdefault(key, []).append(name)
        keep = {names[0]: "\n".join(names) for names in groups.values()}
        df = df[df.setup.isin(keep)].copy()
        label_of = keep
    else:
        label_of = {n: n for n in df.setup.unique()}
    panels = [f"memory {m} error" + (" (after forgetting)" if m > 1 else "") for m in memories]
    n = len(panels) + (1 if show_pull else 0)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 2.5, 5.4))
    axes = np.atleast_1d(axes)
    names = list(df.setup.unique())
    for i, name in enumerate(names):
        d = df[df.setup == name]
        col, mk, ls = style(name, i)
        dx = (i - (len(names) - 1) / 2) * dodge
        for ax, mem in zip(axes, memories):
            m, lo, hi = _median_iqr(d[d.memory == mem], "separation_set_deg", "error_deg")
            ax.fill_between(m.index + dx, lo, hi, color=col, alpha=0.08, lw=0)
            ax.plot(m.index + dx, m.values, marker=mk, ms=8, lw=2.2, ls=ls, color=col, label=label_of[name])
        if show_pull:
            p = d[d.memory == memories[0]].groupby("separation_set_deg")["pull_to_distractor_deg"]
            mean, sem = p.mean(), p.std() / np.sqrt(p.count())
            axes[-1].errorbar(mean.index + dx, mean.values, yerr=sem.values, marker=mk, ms=8, lw=2.2, ls=ls,
                              capsize=3, color=col, label=label_of[name])
    for ax, title in zip(axes, panels + (["pull toward the distractor (memory 1)"] if show_pull else [])):
        ax.set_xlim(185, -6)                                   # far -> on top of the cue
        ax.set_xlabel("cue-distractor separation (deg)", fontsize=FS)
        ax.set_title(title, fontsize=FS, loc="left")
        _style(ax)
    for ax in axes[:len(memories)]:
        ax.set_ylabel("error (deg), median, band = IQR", fontsize=FS)
        ax.set_ylim(0, max(60, ax.get_ylim()[1]))
    if show_pull:
        axes[-1].axhline(0, color="#8c8c8c", lw=1)
        axes[-1].set_ylabel("signed error toward distractor (deg)\nmean +- s.e.m.", fontsize=FS)
    for ax in axes:
        _mark_rf(ax)
    _legend_right(axes[-1])
    fig.tight_layout()
    return fig


def plot_repeats(df, bins=(0, 45, 90, 135, 180)):
    """Left: error per memory in the sequence. Right: bias toward the previous cue."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
    centers = 0.5 * (np.asarray(bins[:-1]) + np.asarray(bins[1:]))
    names = list(df.setup.unique())
    for i, name in enumerate(names):
        d = df[df.setup == name]
        col, mk, ls = style(name, i)
        m, lo, hi = _median_iqr(d, "memory", "error_deg")
        axes[0].fill_between(m.index, lo, hi, color=col, alpha=0.06, lw=0)
        axes[0].plot(m.index, m.values, marker=mk, ms=8, lw=2.2, ls=ls, color=col, label=name)
        dd = d[d.memory > 1].copy()
        dd["bin"] = pd.cut(dd.prev_cue_offset_deg.abs(), bins, include_lowest=True)
        p = dd.groupby("bin", observed=False)["pull_to_prev_cue_deg"]
        mean, sem = p.mean(), p.std() / np.sqrt(p.count())
        off = (i - (len(names) - 1) / 2) * 2.2
        axes[1].errorbar(centers + off, mean.values, yerr=sem.values, marker=mk, ms=7, lw=1.8,
                         ls=ls, capsize=3, color=col, label=name)
    _chance(axes[0])
    axes[0].set_xlabel("memory in the sequence", fontsize=FS)
    axes[0].set_ylabel("error (deg), median, band = IQR", fontsize=FS)
    axes[0].set_xticks(sorted(df.memory.unique()))
    axes[0].set_ylim(0, 185)
    axes[0].set_title("does the error grow over repeats?", fontsize=FS, loc="left")
    axes[1].axhline(0, color="#8c8c8c", lw=1)
    axes[1].set_xticks(centers)
    axes[1].set_xticklabels([f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])])
    axes[1].set_xlabel("distance to the previous cue (deg)", fontsize=FS)
    axes[1].set_ylabel("signed error toward previous cue (deg)\nmean +- s.e.m.", fontsize=FS)
    axes[1].set_title("is each memory independent of the previous one?", fontsize=FS, loc="left")
    _legend_right(axes[1])
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    return fig


def plot_case_summary(df):
    """One row per case. Left: memory 1 (only the distractor can hurt it).
    Right: memories 2+ (forgetting has to work). Dots = single memories, bar = median."""
    names = list(df.setup.unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.62 * len(names) + 1.6), sharey=True)
    rng = np.random.default_rng(0)
    y = np.arange(len(names))[::-1]
    for ax, sel, title in zip(axes, (df.memory == 1, df.memory > 1),
                              ("memory 1 (distractor only)", "memories 2 and later (after forgetting)")):
        for yi, name in zip(y, names):
            col, mk, _ = style(name, names.index(name))
            e = df[sel & (df.setup == name)].error_deg.values
            ax.scatter(e, yi + rng.uniform(-0.18, 0.18, len(e)), s=14, color=col, alpha=0.45, linewidths=0)
            ax.plot([np.median(e)] * 2, [yi - 0.32, yi + 0.32], color="#222222", lw=2.6)
            ax.text(np.median(e) + 2, yi + 0.22, f"{np.median(e):.0f}\u00b0", va="bottom", ha="left",
                    fontsize=FS - 2, color="#222222")
        ax.axvline(90, color="#8c8c8c", ls=":", lw=1.2)
        ax.set_xlim(0, 185)
        ax.set_title(title, fontsize=FS, loc="left")
        ax.set_xlabel("readout error (deg)   |   bar = median, dotted = chance", fontsize=FS - 1)
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=FS - 1)
    fig.tight_layout()
    return fig


def repeat_table(df):
    """Per case: median error of memory 1 and of memories 2+, the mean pull toward
    the previous cue, and the fraction of memories pulled toward it."""
    rows = []
    for name in df.setup.unique():
        d = df[df.setup == name]
        later = d[d.memory > 1]
        rows.append(dict(setup=name,
                         median_err_memory1=d[d.memory == 1].error_deg.median(),
                         median_err_later=later.error_deg.median(),
                         mean_pull_prev_cue=later.pull_to_prev_cue_deg.mean(),
                         sem_pull_prev_cue=later.pull_to_prev_cue_deg.std() / np.sqrt(len(later)),
                         frac_toward_prev=(later.pull_to_prev_cue_deg > 0).mean(),
                         n_memories=len(d)))
    return pd.DataFrame(rows).round(2)


# ===========================================================================
# Final cases (ODR_w0_gating_final.ipynb)
# ===========================================================================
# "pulse" = negative input pulse to every E neuron before cue 1 (-2, 100 ms) AND in every
# forget window before the next cue (-3, 200 ms). w0 = -1 everywhere unless stated:
# "RF out X" = w0 = -1 inside the RF, X outside it during the hold; "forget w0 +1" = w0 = +1
# everywhere during the forget pulse (otherwise w0 stays at -1 there).
FINAL_PULSES = dict(neg_pulse=("start", "forget"), neg_amp=-3.0, start_amp=-2.0, t_neg=100)
FINAL_CASES = {
    "no RF, no pulse":                 dict(gate=False, w0_forget=-1.0),
    "no RF, pulse":                    dict(gate=False, w0_forget=-1.0, **FINAL_PULSES),
    "RF out 0, no pulse":              dict(gate=True, w0_out=0.0, w0_forget=-1.0),
    "RF out 0, pulse":                 dict(gate=True, w0_out=0.0, w0_forget=-1.0, **FINAL_PULSES),
    "RF out +1, no pulse":             dict(gate=True, w0_out=1.0, w0_forget=-1.0),
    "RF out +1, pulse":                dict(gate=True, w0_out=1.0, w0_forget=-1.0, **FINAL_PULSES),
    "RF out 0, pulse + forget w0 +1":  dict(gate=True, w0_out=0.0, w0_forget=1.0, **FINAL_PULSES),
    "RF out +1, pulse + forget w0 +1": dict(gate=True, w0_out=1.0, w0_forget=1.0, **FINAL_PULSES),
}
# added later: the RF map kept through the forget pulse (w0 -1 inside, 0 outside), and longer cues
FINAL_CASES["RF out 0, pulse, forget keeps RF"] = dict(gate=True, w0_out=0.0, forget_keep_rf=True, **FINAL_PULSES)
FINAL_CASES["RF out 0, pulse, cue 500 ms"] = dict(gate=True, w0_out=0.0, w0_forget=-1.0, t_cue=500, **FINAL_PULSES)
FINAL_CASES["RF out 0, pulse, cue 1000 ms"] = dict(gate=True, w0_out=0.0, w0_forget=-1.0, t_cue=1000, **FINAL_PULSES)
for _kw in FINAL_CASES.values():
    _kw["t_forget"] = 200
# cases whose memory 1 is identical to another case (they differ only after memory 1)
SAME_MEMORY1 = {"RF out 0, pulse + forget w0 +1", "RF out +1, pulse + forget w0 +1", "RF out 0, pulse, forget keeps RF"}
# explicit (colour, line style, marker) for the added cases; the rest follow w0 outside / pulse
EXTRA_STYLES = {
    "RF out 0, pulse, forget keeps RF": ("#1B9E77", "-", "s"),
    "RF out 0, pulse, cue 500 ms":      ("#9970AB", "-", "^"),
    "RF out 0, pulse, cue 1000 ms":     ("#5E3C99", "-", "v"),
}

# colour = w0 outside the RF during the hold (no RF = -1 everywhere), line = pulse or not
W0_OUT_COLOURS = {-1: "#2166AC", 0: "#4D4D4D", 1: "#D6452B"}
W0_OUT_LABELS = {-1: "no RF ($w_0=-1$ everywhere)", 0: "RF, $w_0=0$ outside", 1: "RF, $w_0=+1$ outside"}
PULSE_STYLES = {False: (0, (1.2, 1.6)), True: "-"}          # no pulse = dotted, pulse = solid


def final_case_marker(name):
    if name in EXTRA_STYLES:
        return EXTRA_STYLES[name][2]
    return "D" if "forget w0 +1" in name else "o"


def final_case_style(name):
    kw = FINAL_CASES[name]
    if name in EXTRA_STYLES:
        col, ls, _ = EXTRA_STYLES[name]
        return col, ls, (-1 if not kw["gate"] else int(kw["w0_out"])), "neg_pulse" in kw
    w0o = -1 if not kw["gate"] else int(kw["w0_out"])
    pulse = "neg_pulse" in kw
    return W0_OUT_COLOURS[w0o], PULSE_STYLES[pulse], w0o, pulse


def plot_separation_final(df, ax=None, memory=1, fs=None, legend=True):
    """Median memory-1 error vs cue-distractor separation. Colour = w0 outside the RF,
    solid = pulse, dotted = no pulse. Cases that differ only in the forget signal have the same
    memory 1, so only one of them is drawn."""
    fs = fs or FS
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5))
    d = df[df.memory == memory]
    for name in FINAL_CASES:
        if name not in set(d.setup) or (memory == 1 and name in SAME_MEMORY1):
            continue
        col, ls, w0o, pulse = final_case_style(name)
        m = d[d.setup == name].groupby("separation_set_deg").error_deg.median()
        ax.plot(m.index, m.values, color=col, ls=ls, lw=2.4, marker=final_case_marker(name) if pulse else None, ms=5,
                mec="white", mew=0.5, label=SHORT_LABELS[name])
    ax.axvspan(0, RF_RADIUS, color="#bdbdbd", alpha=0.25, lw=0, zorder=0)
    ax.axvline(RF_RADIUS, color="#8c8c8c", lw=2.0, zorder=1)
    ax.set_xlim(185, -5)
    ax.set_xticks([180, 135, 90, 45, 0])
    ax.set_ylim(0, None)
    ax.text(RF_RADIUS / 2, ax.get_ylim()[1] * 0.98, "receptive field", rotation=90, ha="center", va="top",
            fontsize=fs - 2, color="#555555")
    ax.set_xlabel("cue–distractor separation (deg)", fontsize=fs)
    ax.set_ylabel(f"memory {memory} error (deg)", fontsize=fs)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=fs - 1)
    if legend:   # one entry per network; "pulse" = negative input before each cue (for memory 1: before cue 1)
        ax.legend(fontsize=fs - 2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  title="one line per network\n(pulse = negative input before each cue)", title_fontsize=fs - 3,
                  handlelength=2.6)
    return ax


def final_legend_handles():
    from matplotlib.lines import Line2D
    blank = Line2D([], [], color="none")
    h = [blank] + [Line2D([], [], color=W0_OUT_COLOURS[k], lw=3) for k in (-1, 0, 1)]
    l = [r"$\bf{hold}$"] + [W0_OUT_LABELS[k] for k in (-1, 0, 1)]
    h += [blank, Line2D([], [], color="#333333", lw=2.4, ls=PULSE_STYLES[True], marker="o", ms=4.5, mec="white"),
          Line2D([], [], color="#333333", lw=2.4, ls=PULSE_STYLES[False])]
    l += [r"$\bf{negative\ pulse}$", "pulse before each cue", "no pulse"]
    return h, l


def add_final_legend(ax, fs=10, **kw):
    h, l = final_legend_handles()
    opts = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=fs,
                handlelength=2.6, labelspacing=0.35)
    opts.update(kw)
    return ax.legend(h, l, **opts)


def efficiency_table(df, first=2):
    """Per case, over memories `first`.. : readout error (median, mean), and coding cost:
    fraction of E neurons that fire during the readout window, mean E rate (Hz per neuron)
    during the readout, during the hold (cue offset -> readout end), and over the whole trial."""
    rows = []
    for name in FINAL_CASES:
        d = df[(df.setup == name) & (df.memory >= first)]
        if d.empty:
            continue
        rows.append({"case": name,
                     "error median (deg)": d.error_deg.median(),
                     "error mean (deg)": d.error_deg.mean(),
                     "active at readout (%)": 100 * d.frac_active_readout.mean(),
                     "readout spikes on the cue (%)": 100 * d.spikes_on_cue_readout.mean(),
                     "rate readout (Hz)": d.rate_readout_hz.mean(),
                     "rate hold (Hz)": d.rate_hold_hz.mean(),
                     "rate whole trial (Hz)": d.drop_duplicates(["batch", "trial"]).rate_trial_hz.mean(),
                     "RF misplaced (%)": 100 * (d.rf_error_deg > RF_RADIUS).mean() if d.rf_error_deg.notna().any() else np.nan})
    return pd.DataFrame(rows).set_index("case")


FINAL_TEX_LABELS = {
    "no RF, no pulse": r"no RF, no pulse",
    "no RF, pulse": r"no RF, pulse",
    "RF out 0, no pulse": r"RF ($w_0^{out}=0$), no pulse",
    "RF out 0, pulse": r"RF ($w_0^{out}=0$), pulse",
    "RF out +1, no pulse": r"RF ($w_0^{out}=+1$), no pulse",
    "RF out +1, pulse": r"RF ($w_0^{out}=+1$), pulse",
    "RF out 0, pulse + forget w0 +1": r"RF ($w_0^{out}=0$), pulse + $w_0=+1$",
    "RF out +1, pulse + forget w0 +1": r"RF ($w_0^{out}=+1$), pulse + $w_0=+1$",
    "RF out 0, pulse, forget keeps RF": r"RF ($w_0^{out}=0$), pulse, RF kept",
    "RF out 0, pulse, cue 500 ms": r"RF ($w_0^{out}=0$), pulse, cue 500\,ms",
    "RF out 0, pulse, cue 1000 ms": r"RF ($w_0^{out}=0$), pulse, cue 1000\,ms",
}


def efficiency_latex(df, path=None):
    t = efficiency_table(df)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Network & Error & Active at & Spikes on & Rate, & Rate, \\",
             r" & (deg) & readout (\%) & the cue (\%) & hold (Hz) & trial (Hz) \\",
             r"\midrule"]
    for name, x in t.iterrows():
        lines.append(rf"{FINAL_TEX_LABELS[name]} & {x['error median (deg)']:.0f} & "
                     rf"{x['active at readout (%)']:.0f} & {x['readout spikes on the cue (%)']:.0f} & "
                     rf"{x['rate hold (Hz)']:.1f} & {x['rate whole trial (Hz)']:.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    if path:
        with open(path, "w") as f:
            f.write(tex + "\n")
    return tex


SHORT_LABELS = {
    "no RF, no pulse": "no RF",
    "no RF, pulse": "no RF + pulse",
    "RF out 0, no pulse": "RF, $w_0^{out}=0$",
    "RF out 0, pulse": "RF, $w_0^{out}=0$ + pulse",
    "RF out +1, no pulse": "RF, $w_0^{out}=+1$",
    "RF out +1, pulse": "RF, $w_0^{out}=+1$ + pulse",
    "RF out 0, pulse + forget w0 +1": "RF, $w_0^{out}=0$ + pulse ($w_0=+1$)",
    "RF out +1, pulse + forget w0 +1": "RF, $w_0^{out}=+1$ + pulse ($w_0=+1$)",
    "RF out 0, pulse, forget keeps RF": "RF, $w_0^{out}=0$ + pulse (RF kept)",
    "RF out 0, pulse, cue 500 ms": "RF, $w_0^{out}=0$ + pulse, cue 500 ms",
    "RF out 0, pulse, cue 1000 ms": "RF, $w_0^{out}=0$ + pulse, cue 1000 ms",
}
OUT_OF_CUE_FRAC = 1 - 2 * RF_RADIUS / 360          # share of the ring farther than 42.5 deg from the cue


def error_vs_rate_outside(df, first=2, ax=None, fs=None):
    """Mean +- SD readout error (memories `first`..) against the mean +- SD E rate of the
    neurons farther than 42.5 deg from the cue during the readout window."""
    fs = fs or FS
    d = df[df.memory >= first].copy()
    d["rate_out"] = d.rate_readout_hz * (1 - d.spikes_on_cue_readout) / OUT_OF_CUE_FRAC
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.2))
    for name in FINAL_CASES:
        g_ = d[d.setup == name]
        if g_.empty:
            continue
        col, _, w0o, pulse = final_case_style(name)
        mk = final_case_marker(name)
        ax.errorbar(g_.rate_out.mean(), g_.error_deg.mean(), xerr=g_.rate_out.std(), yerr=g_.error_deg.std(),
                    fmt=mk, ms=9, color=col, mfc=col if pulse else "white", mew=2, capsize=3, lw=1.2,
                    label=SHORT_LABELS[name], alpha=0.9)
    ax.axhline(90, color="#8c8c8c", ls=":", lw=1)
    ax.text(0.5, 91, "chance", ha="left", va="bottom", fontsize=fs - 3, color="#555555")
    ax.legend(fontsize=fs - 3, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="filled = pulse, open = no pulse\n◆ = $w_0=+1$ during the pulse, ■ = RF kept,\n▲▼ = longer cue (500 / 1000 ms)", title_fontsize=fs - 3)
    ax.set_xlabel("E rate outside the cue region during readout (Hz)", fontsize=fs)
    ax.set_ylabel(f"error, memories {first}–10 (deg)\nmean ± SD", fontsize=fs)
    ax.set_ylim(-5, None)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=fs - 1)
    return ax


# ===========================================================================
# Cue-500 set: the six hold / pulse networks, all with a 500 ms cue.
# w0 = -1 inside the RF and in every forget window; w0_out = 0 or +1 outside the RF.
# ===========================================================================
CUE500_CASES = {
    "no RF, no pulse":     dict(gate=False, w0_forget=-1.0),
    "no RF, pulse":        dict(gate=False, w0_forget=-1.0, **FINAL_PULSES),
    "RF out 0, no pulse":  dict(gate=True, w0_out=0.0, w0_forget=-1.0),
    "RF out 0, pulse":     dict(gate=True, w0_out=0.0, w0_forget=-1.0, **FINAL_PULSES),
    "RF out +1, no pulse": dict(gate=True, w0_out=1.0, w0_forget=-1.0),
    "RF out +1, pulse":    dict(gate=True, w0_out=1.0, w0_forget=-1.0, **FINAL_PULSES),
}
for _kw in CUE500_CASES.values():
    _kw.update(t_forget=200, t_cue=500)
