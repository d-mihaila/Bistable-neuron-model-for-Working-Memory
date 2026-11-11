"""
shared_plots.py -- drawing helpers shared by the paper figures (imported by style.py as `P`).

They were first written for the Cosyne 2027 abstract figures and are kept here unchanged, only the
parts the paper figures use:

    W0_COLOURS, w0_colour           the w0 colour scale (blue = bistable, w0 < 0; red = IF-like, w0 > 0)
    _phase_pair                     two phase planes at I_ext = 0 with their key        (Fig. 1A)
    example_traces                  one MNIST test digit through each trained network   (Fig. 2B)
    make_figure3                    one ODR trial + error against cue-distractor separation (Fig. 3A, B)
    panel_odr_cue500, panel_odr_raster, panel_cue_scheme, ODR_CUE500_LINES ...   its panels and labels

Paths are relative to the repository root (the folder above NICE_figures/).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))       # .../NICE_figures
ROOT = os.path.dirname(HERE)                             # repository root
SINGLE_NEURON = os.path.join(ROOT, "single_neuron")      # neuron_figures.py, params.py

FS = 8   # default font size; make_figure3 sets it to its own fs while it draws


def _style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FS - 1, length=2.5, pad=1.5)


# ===========================================================================
# colours
# ===========================================================================

W0_COLOURS = {-2.0: "#1B1F6B", -1.0: "#2166AC", -0.5: "#4FB3D9",
              0.5: "#F2A33A", 1.0: "#D6452B", 2.0: "#7A0A1E"}


def w0_colour(w0):
    return W0_COLOURS[float(w0)]


# MNIST network colours of the Cosyne figure (style.py reads the LIF black from here)
FIG2_MNIST_LABELS = {
    "const_w0_-1.00": r"bistable, $w_0=-1$",
    "const_w0_+1.00": r"IF-like, $w_0=+1$",
    "all_lif_wait100": "LIF",
    "lif_hidden_bistable_output_w0_-1.00": r"LIF hidden, $w_0=-1$ output",
}

FIG2_MNIST_COLOURS = {
    "const_w0_-1.00": W0_COLOURS[-1.0],          # blue, as w0 = -1 everywhere else
    "const_w0_+1.00": W0_COLOURS[1.0],           # red, as w0 = +1 everywhere else
    "all_lif_wait100": "#111111",
    "lif_hidden_bistable_output_w0_-1.00": "#009E73",
}


# ===========================================================================
# Fig. 1A: phase planes at I_ext = 0 (drawn by single_neuron/neuron_figures._phase_panel)
# ===========================================================================

FIG4_STARTS = [(-3.0, -0.6), (-2.2, 1.8), (-0.3, -2.2), (0.6, -1.6), (0.8, 2.8),
               (0.0, 1.0),                                    # starts at the reset point (c, d)
               (0.2, 0.02)]                                   # just right of the w0 = 0 critical point: escapes and spikes; all v < v_th = 1


def _shrink_phase_panel(ax, k):
    """neuron_figures._phase_panel draws line widths, marker and arrow sizes in points for a
    1.3-inch panel; scale them by k for a smaller panel (neuron_figures.py itself is not changed)."""
    for ln in ax.lines:
        ln.set_linewidth(ln.get_linewidth() * k)
        ln.set_markersize(ln.get_markersize() * k)
        ln.set_markeredgewidth(ln.get_markeredgewidth() * k)
    for coll in ax.collections:                              # the streamplot flow lines
        coll.set_linewidth(np.asarray(coll.get_linewidth()) * k)
    for patch in ax.patches:                                 # streamplot arrowheads
        if hasattr(patch, "set_mutation_scale"):
            patch.set_mutation_scale(patch.get_mutation_scale() * k)
    for ann in ax.texts:                                     # trajectory / reset arrows
        ap = getattr(ann, "arrow_patch", None)
        if ap is not None:                                   # Annotation re-reads mutation_scale from arrowprops on every draw
            ann.arrowprops["mutation_scale"] = ann.arrowprops.get("mutation_scale", 10) * k * 0.85
            ap.set_linewidth(ap.get_linewidth() * k)


def _phase_pair(fig, W, H, left, bottom, side, gap, w0s, starts, fs, lim, key_ncol=4,
                box_labels=(("bistable", None), ("IF-like", None))):
    """Two I_ext = 0 phase planes (one per w0) drawn into `fig` (W x H in), lower-left corner at
    (left, bottom) in inches, key underneath. Each panel gets a thin dotted frame with a coloured
    name above it (box_labels = (text, colour) per panel; colour None -> the w0 colour of Figure 2)."""
    if SINGLE_NEURON not in sys.path:
        sys.path.insert(0, SINGLE_NEURON)
    import importlib
    import neuron_figures as nf
    importlib.reload(nf)
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    assert all(v0 < nf.v_th for v0, _ in starts), "initial v must be below the threshold"
    defaults = (W0_COLOURS[-1.0], W0_COLOURS[1.0])        # Figure 2: bistable blue, IF-like red
    for i, w0 in enumerate(w0s):
        x = left + i * (side + gap)
        ax = fig.add_axes([x / W, bottom / H, side / W, side / H])
        nf._phase_panel(ax, w0, 0.0, lim=lim, starts=starts)
        for art in list(ax.collections) + list(ax.patches):          # no vector field
            art.remove()
        _shrink_phase_panel(ax, 0.8)
        ax.set_title(rf"$w_0={w0:+g}$".replace("-", "\u2212") if w0 else r"$w_0=0$", fontsize=fs + 1, pad=3)
        ax.set_xlabel(r"$v$", fontsize=fs, labelpad=0)
        ax.tick_params(labelsize=fs - 1, length=2.0, pad=1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_ylabel(r"$w$", fontsize=fs, labelpad=0)
        # dotted frame (round title, ticks and v label) with the name above, in the Figure 2 colour
        txt, col = box_labels[i]
        col = col or defaults[i]
        m_l, m_r, m_b, m_t = (0.42 if i == 0 else 0.32), 0.05, 0.28, 0.20
        fig.add_artist(FancyBboxPatch(((x - m_l) / W, (bottom - m_b) / H), (side + m_l + m_r) / W,
                                      (side + m_b + m_t) / H, boxstyle="round,pad=0,rounding_size=0.008",
                                      mutation_aspect=W / H, fill=False, ec=col, lw=0.9,
                                      ls=(0, (1.2, 2.0)), transform=fig.transFigure))
        fig.text((x - m_l + (side + m_l + m_r) / 2) / W, (bottom + side + m_t + 0.03) / H, txt, color=col,
                 fontsize=fs, fontweight="bold", ha="center", va="bottom")
    key = [Line2D([], [], color=nf.C_VNULL, lw=2.0, label=r"$\dot v = 0$"),
           Line2D([], [], color=nf.C_WNULL, lw=2.0, ls=(0, (4, 2)), label=r"$\dot w = 0$"),
           Line2D([], [], color="#999999", lw=1.2, ls=":", label="threshold"),
           Line2D([], [], color=nf.C_CYCLE, lw=2.6, label="limit cycle"),
           Line2D([], [], color=nf.C_TRAJ, lw=1.8, label="spike"),
           Line2D([], [], color=nf.C_REST, lw=1.8, label="at rest"),
           Line2D([], [], marker="o", color="w", ms=5.5, fillstyle="left", mfc="black",
                  mfcalt="white", mec="black", mew=0.9, label="bifurcation point"),
           Line2D([], [], marker="s", color="w", mfc=nf.C_RESET, mec="black", ms=4.5, label="reset point")]
    fig.legend(handles=key, loc="upper center",
               bbox_to_anchor=((left + (len(w0s) * side + (len(w0s) - 1) * gap) / 2 - 0.1) / W, (bottom - 0.42) / H),
               ncol=key_ncol, frameon=False, fontsize=fs - 2, handlelength=1.2, columnspacing=0.8,
               handletextpad=0.35, labelspacing=0.3, borderpad=0, borderaxespad=0)


# ===========================================================================
# Fig. 2B: one MNIST test digit through each trained network
# ===========================================================================

def example_traces(runs, sample_index=0, t_wait=4900, seed=0, dataset="MNIST"):
    """Run ONE test digit through each network in `runs` with a long silent
    delay; returns {run: dict(hidden=(T, n_hidden), output=(T, n_out), label=int)}.
    Imports the dataset's own code, so call it once per dataset."""
    import torch
    folder = os.path.join(ROOT, "MNIST" if dataset == "MNIST" else "N_MNIST")
    sys.path.insert(0, folder)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        if dataset == "MNIST":
            import mnist_lib as lib
            from save_experiment import load_trained_model
            enc = lambda x: lib.encode_batch(x, "cpu", torch.float)
        else:
            import nmnist_w0_lib as lib
            from rebuild_network import load_trained_model
            enc = lambda x: lib.flatten_batch(x, "cpu", torch.float)
        _, testloader = lib.get_dataloaders(batch_size=64)
        torch.manual_seed(seed)
        x, y = next(iter(testloader))
        x, y = x[sample_index:sample_index + 1], int(y[sample_index])
        torch.manual_seed(seed)
        xe = enc(x)
        traces = {}
        for run in runs:
            model, _, _ = load_trained_model(lib.DEFAULT_OUT_DIR, run, device="cpu")
            model.T_wait = t_wait
            with torch.no_grad():
                _, recs = model.forward(xe)
            traces[run] = dict(hidden=recs["spk"][0].numpy(), output=recs["out_spk"][0].numpy(),
                               label=y, t_stim=lib.NUM_STEPS)
        return traces
    finally:
        os.chdir(cwd)
        sys.path.remove(folder)


# ===========================================================================
# Fig. 3: the ODR task (ODR/results/w0_gating_cue500_mnistLIF/)
# ===========================================================================

RF_RADIUS = 2.5 * 40.0 / 2.355          # 42.5 deg: receptive-field half-width = cue half-width

# networks: colour = w0 outside the RF during the hold. These are the Cosyne colours; the paper colours
# are set in style.ODR_W0_OUT_COLOURS and swapped in by odr_figures.make_figure_trial.
ODR_W0_OUT_COLOURS = {-1: "#2166AC", 0: "#4D4D4D", 1: "#D6452B"}
ODR_PULSE_LS = {False: (0, (1.2, 1.6)), True: "-"}
ODR_CUE_GREEN = "#1A9641"      # Figure 3: cue (same as ODR/w0_gating.CUE_GREEN)
ODR_DIS_RED = "#D62728"        # Figure 3: distractor (same as ODR/w0_gating.DISTRACTOR_RED)

# the six 500 ms-cue networks of the paper
ODR_CUE500_DIR = os.path.join(ROOT, "ODR", "results", "w0_gating_cue500_mnistLIF")

# Legend rule: "no RF" / "no pulse" are only written when the RF / pulse is absent. An RF is
# stated by its w0 outside the RF (w_0^out), a pulse by the w0 during the forget window
# (w_0^forget, always -1 here). Anything not marked "no ..." has it.
_WOUT = lambda v: r"$w_0^{\mathrm{out}}=%s$" % v
_WFORGET = r"$w_0^{\mathrm{forget}}=-1$"
ODR_CUE500_LINES = [  # (case, label, w0 outside, pulse); no pulse first: the legend's left column
    ("no RF, no pulse",     "no RF, no pulse",                -1, False),
    ("RF out 0, no pulse",  f"{_WOUT('0')}, no pulse",         0, False),
    ("RF out +1, no pulse", f"{_WOUT('+1')}, no pulse",        1, False),
    ("no RF, pulse",        f"no RF, {_WFORGET}",             -1, True),   # right column: with pulse
    ("RF out 0, pulse",     f"{_WOUT('0')}, {_WFORGET}",       0, True),
    ("RF out +1, pulse",    f"{_WOUT('+1')}, {_WFORGET}",      1, True),
]


def panel_odr_cue500(ax, overlap=None, lw=1.8, percent=False):
    """Fig. 3B: memory-1 error vs cue-distractor separation, 500 ms cue, six networks.
    Colour = w0 outside the RF (blue: no RF), solid with dots = pulse, dotted = no pulse.
    percent=True: error as % of the largest possible error, error / 180 deg * 100."""
    if overlap is None:
        overlap = pd.read_csv(os.path.join(ODR_CUE500_DIR, "cue500_overlap.csv"))
    d = overlap[overlap.memory == 1]
    for case, lab, w0o, pulse in ODR_CUE500_LINES:
        m = d[d.setup == case].groupby("separation_set_deg").error_deg.median()
        if percent:
            m = m / 180.0 * 100.0
        ax.plot(m.index, m.values, color=ODR_W0_OUT_COLOURS[w0o], ls=ODR_PULSE_LS[pulse], lw=lw,
                marker="o" if pulse else None, ms=3.2, mec="white", mew=0.4, label=lab,
                zorder=5 if w0o == 0 else 3)     # the black (RF out 0) lines, dotted and solid, in front
    ax.axvspan(0, RF_RADIUS, color="#bdbdbd", alpha=0.25, lw=0, zorder=0)
    ax.axvline(RF_RADIUS, color="#8c8c8c", lw=1.6, zorder=1)
    ax.set_xlim(185, -5)
    ax.set_ylim(0, None)
    ax.text(RF_RADIUS / 2, ax.get_ylim()[1] * 0.98, "RF", ha="center", va="top", fontsize=FS - 2,
            color="#555555")
    _style_ax(ax)


def panel_cue_scheme(ax, fs=10, cue_col="#111111", dis_col="#B03A2E", dis_label="distractor\n180°",
                     sep_label="180°", xlim=(-1.4, 2.4), side_labels=False, horizontal=False):
    """Mini scheme of the ODR ring: cue at 90 deg, distractor at -90 deg, 180 deg apart.
    dis_label: text under the distractor; sep_label: text next to the arc arrow (None = no text).
    horizontal=True (needs side_labels): the ring turned by 90 deg -- cue at 0 deg (right), distractor
    at 180 deg (left), arc arrow over the top from the distractor to the cue."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color="#555555", lw=1.6)
    if not side_labels:
        ax.text(0, 0, "360°\nring", ha="center", va="center", fontsize=fs - 2, color="#777777",
                linespacing=1.0)
    if horizontal:
        pts = [(0, cue_col, "cue\n0°", "bottom"), (180, dis_col, dis_label, "top")]
    else:
        pts = [(90, cue_col, "cue\n90°", "bottom"), (-90, dis_col, dis_label, "top")]
    for ang, col, lab, va in pts:
        x, y = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        ax.annotate("", xy=(x, y), xytext=(0.45 * x, 0.45 * y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=12))
        ax.plot(x, y, "o", color=col, ms=6, zorder=4)
        if side_labels and horizontal:      # one line, outside the dot (cue right, distractor left)
            ax.text(1.3 * x, 0, lab.replace("\n", " "), ha="left" if x > 0 else "right", va="center",
                    fontsize=fs - 1, color=col)
        elif side_labels:          # one line, left of the dot
            ax.text(-0.75, y, lab.replace("\n", " "), ha="right", va="center", fontsize=fs - 1, color=col)
        else:
            ax.text(0, 1.18 * y, lab, ha="center", va=va, fontsize=fs - 1, color=col, linespacing=1.0)
    if horizontal:
        a = np.radians(np.linspace(165, 15, 60))       # over the top, distractor (180) -> cue (0)
    else:
        a = np.radians(np.linspace(-80, 80, 60))       # arc from distractor to cue, outside the ring
    r = 1.28
    path = Path(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", color="#333333", lw=1.4,
                                 mutation_scale=11))
    if sep_label:
        ax.text(r + 0.2, 0, sep_label, ha="left", va="center", fontsize=fs - 1, color="#333333")
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*((-1.45, 1.45) if side_labels else (-1.95, 1.95)))
    ax.axis("off")


def panel_odr_raster(fig, rect, fs=9, case="RF out 0, pulse", start_pulse_ms=200, time_in_s=False,
                     marker_scale=0.22):
    """Fig. 3A: one ODR trial (cue 90 -> 270 deg, 500 ms cue) drawn by ODR/w0_gating.plot_pipeline into
    `rect` = (left, bottom, right, top) of `fig`. Same 8-trial batch as the ODR notebook; trial 0.
    The negative pulse at the start lasts `start_pulse_ms` (200 ms, as long as the forget window)
    and ends exactly where the first cue starts (the network has 200 ms of fixation before it)."""
    folder = os.path.join(ROOT, "ODR")
    shared = ["network_architecture", "neuron_model", "task", "plots", "run_odr_demo", "w0_gating", "w0_experiments"]
    saved = {m: sys.modules.pop(m) for m in shared if m in sys.modules}   # MNIST has modules with the same names
    sys.path.insert(0, folder)
    try:
        import w0_gating as g
        import w0_experiments as X
        kw = dict(X.CUE500_CASES[case])
        if kw.get("neg_pulse"):
            kw["t_neg"] = start_pulse_ms
        pipe, r, _ = X.run_sequence((90.0, 270.0), (210.0, 30.0), 8, record=True, gap=500, **kw)
        g.plot_pipeline(r, pipe, fig=fig, rect=rect, fs=fs, compact_labels=True, marker_scale=marker_scale,
                        label_fs=fs - 1, w0_range=(-1.0, 0.0), time_in_s=time_in_s)
    finally:
        sys.path.remove(folder)
        for m in shared:
            sys.modules.pop(m, None)
        sys.modules.update(saved)
    return r, pipe


def make_figure3(width=7.5, height=2.45, fs=9, raster_fs=9, case="RF out 0, pulse", start_pulse_ms=100,
                 raster_rect=(0.5, 0.36, 3.75, 1.76), plot_box=(4.88, 0.62, 1.04, 0.92),
                 scheme_box=(4.49, 1.56, 1.8, 0.7), legend_xy=(5.99, 1.08), letters=("A", "B"),
                 letter_y=2.23, letter_b_x=4.3):
    """Every position is in inches: raster_rect = (left, bottom, right, top) of the raster panels
    (event labels go above `top`); plot_box / scheme_box = (x, y, w, h) of the error-vs-separation plot
    and of the ring scheme above it; legend_xy = left edge and vertical centre of the legend (one column,
    right of the plot). Returns (fig, r, pipe) of the raster trial."""
    import matplotlib as mpl
    global FS
    old_fs, FS = FS, fs
    try:
        W, H = width, height
        fig = plt.figure(figsize=(W, H))
        box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])

        # ---- A: one ODR trial ------------------------------------------------------------
        l, b, rgt, t = raster_rect
        r, pipe = panel_odr_raster(fig, (l / W, b / H, rgt / W, t / H), fs=raster_fs, case=case,
                                   start_pulse_ms=start_pulse_ms, time_in_s=True)

        # ---- B: error vs separation, ring scheme on top, legend on the right ---------------
        ax_c = box(*plot_box)
        panel_odr_cue500(ax_c, percent=True)
        ax_c.set_xticks([180, 90, 0]); ax_c.set_xticklabels(["180°", "90°", "0°"])
        ax_c.set_yticks([0, 2, 4, 6])
        ax_c.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax_c.set_xlabel("cue–distractor\nseparation", fontsize=fs)
        ax_c.set_ylabel("decoding error", fontsize=fs)
        ax_s = box(*scheme_box)
        panel_cue_scheme(ax_s, fs=fs - 1, cue_col=ODR_CUE_GREEN, dis_col=ODR_DIS_RED,
                         dis_label="distractor\n180°", sep_label=None, xlim=(-4.5, 3.0), side_labels=True,
                         horizontal=True)
        h, lab = ax_c.get_legend_handles_labels()
        fig.legend(h, lab, loc="center left", bbox_to_anchor=(legend_xy[0] / W, legend_xy[1] / H), ncol=1,
                   fontsize=fs - 2, frameon=False, borderpad=0, borderaxespad=0,
                   handlelength=1.4, labelspacing=0.25, handletextpad=0.4)

        # ---- panel letters -----------------------------------------------------------------
        if letters:
            fig.text(0.02 / W, letter_y / H, letters[0], fontsize=fs + 2, fontweight="bold", va="bottom")
            fig.text(letter_b_x / W, letter_y / H, letters[1], fontsize=fs + 2, fontweight="bold", va="bottom")
        return fig, r, pipe
    finally:
        FS = old_fs
