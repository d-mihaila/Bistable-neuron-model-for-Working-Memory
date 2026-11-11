"""
odr_figures.py -- Fig. 3 of the paper (ODR task) and a supplementary network schematic.
Shown in fig3_odr.ipynb and supplementary.ipynb.

    make_figure_trial()       Fig. 3: A = one trial of the best network (RF, w0 = 0 outside,
                              negative pulse; cue 90 -> 270 deg), B = ring scheme + memory-1 error against
                              cue-distractor separation for the six networks, C = error against
                              firing rate per cue presentation
    make_figure_network()     schematic: ring of E neurons, I population inside, the cue as an arrow;
                              right: the E -> E connectivity profile (Mexican hat)
    rate_error_table()        the numbers of panel C (also E and I rates separately)

Results: ODR/results/w0_gating_cue500_mnistLIF/ (I neurons = the MNIST LIF):
    cue500_overlap.csv             memory-1 error vs separation       (ODR/ODR_w0_gating.ipynb, section 2)
    cue500_repeats_looprates.csv   ten memories per trial, with the rate per cue presentation
                                   (ODR/exp_loop_rates.py)
The raster trial of make_figure_trial is simulated with the ODR code (needs torch, ~30 s).
Colours (style.ODR_W0_OUT_COLOURS, as Figure 2): w0 outside the RF -- blue = no RF (w0 = -1 everywhere,
the bistable blue), purple = 0 (the heterogeneous purple), red = +1 (the w0 = +1 red); pulse = solid /
filled, no pulse = dotted / open; cue green, distractor red.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D

import style as S

P = S.P
ODR_DIR = os.path.join(S.ROOT, "ODR")
RESULTS = os.path.join(ODR_DIR, "results", "w0_gating_cue500_mnistLIF")
N_EXCIT, N_INHIB = 1024, 256                    # ODR/w0_experiments.py
RF_RADIUS = P.RF_RADIUS                          # 42.5 deg: RF half-width = cue half-width

# the six networks, in legend order: (case in the CSVs, label, w0 outside, pulse)
NETWORKS = P.ODR_CUE500_LINES
W0_OUT_COLOURS = S.ODR_W0_OUT_COLOURS           # -1 blue (no RF), 0 purple, +1 red (as Figure 2)
C_W0_ZERO = P.ODR_W0_OUT_COLOURS[0]              # grey: a neuron at w0 = 0 (ring schematic, show_rf)
C_E = S.C_BISTABLE                               # E neurons: bistable (w0 = -1 when not gated)
C_I = S.C_LIF                                    # I neurons: LIF


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 3: one trial + distractor immunity + error against firing rate
# ═════════════════════════════════════════════════════════════════════════════

def _network_legend_handles(compact=False):
    """One handle per network for panels B and C: line as in B (solid = pulse, dotted = none) and marker
    as in C (filled = pulse, open = none), in the w0-outside colour.
    compact=True: for a two-column legend (no pulse | pulse, each network's two lines side by side) with
    the pulse condition said once, as a column header ("no pulse" / w0_forget = -1), and only the RF
    condition per entry ("no RF", w0_out = 0, w0_out = +1). Order: header, 3 networks; header, 3 networks."""
    hs = []
    if compact:
        for pulse, header in ((False, "no pulse"), (True, P._WFORGET)):
            hs.append(Line2D([], [], color="none", label=header))
            for name, lab, w0o, p in NETWORKS:
                if p != pulse:
                    continue
                col = W0_OUT_COLOURS[w0o]
                short = "no RF" if w0o == -1 else P._WOUT("0" if w0o == 0 else "+1")
                hs.append(Line2D([], [], color=col, ls=P.ODR_PULSE_LS[pulse], lw=1.6, marker="o", ms=4,
                                 mfc=col if pulse else "white", mec=col, mew=1.0, label=short))
        return hs
    for name, lab, w0o, pulse in NETWORKS:
        col = W0_OUT_COLOURS[w0o]
        hs.append(Line2D([], [], color=col, ls=P.ODR_PULSE_LS[pulse], lw=1.6, marker="o", ms=4,
                         mfc=col if pulse else "white", mec=col, mew=1.0, label=lab))
    return hs


C_READOUT = "#B8860B"                           # dark yellow: cue angle at readout and its error (panel A)


def recolour_decoded(fig, colour=C_READOUT, old_text="#B2182B"):
    """Panel A, "decoded" raster (drawn by ODR/w0_gating.plot_pipeline, left unchanged): the line at the
    cue angle during each readout solid in `colour` (was dotted cue green), and the "x° / error" text in
    the same colour (was red, w0_gating.ERROR_RED = old_text)."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import same_color
    ax_d = next((a for a in fig.axes if a.get_ylabel() == "decoded"), None)
    if ax_d is None:
        return None
    for c in ax_d.collections:
        if isinstance(c, LineCollection):                  # the readout lines (hlines); dots are a scatter
            c.set_color(colour)
            c.set_linestyle("solid")
    for t in ax_d.texts:                                   # "2°" and "error" (Text and Annotation)
        if same_color(t.get_color(), old_text):
            t.set_color(colour)
    return ax_d


def make_figure_trial(results_dir=RESULTS, width=S.PAGE_W, height=3.0, fs=S.FS, rate_df=None, first=2,
                      rate="rate_loop_hz", raster_right=3.3, b_box=(4.3, 1.32, 0.88, 0.84),
                      c_box=(6.08, 1.95, 0.98, 0.75), axis_fs=None, legend_gap=0.1, frame=True,
                      readout_colour=C_READOUT, **kw):
    """A: one trial (RF out 0 + pulse, simulated with the current ODR code). B: ring scheme + memory-1
    error against cue-distractor separation (cue500_overlap.csv in `results_dir`). C: error against the
    firing rate per cue presentation (rate_error_table), short, top right. One legend for B and C under C,
    right of B: two columns (no pulse, dotted | pulse, solid: each network's two lines side by side), the
    pulse condition as column header (_network_legend_handles(compact=True)), in a thin frame (frame=True,
    as Figure 2), its right edge level with C's.
    raster_right: right edge of A's rasters (in; A's w0 colour bar sits right of it). b_box / c_box =
    (x, y, w, h) of B's and C's axes. axis_fs: tick and axis-label size of B and C; None = A's (fs - 1).
    legend_gap: from C's x label to the legend frame. readout_colour: the cue-angle line at each readout
    and its "x° error" in A's decoded raster (solid; None = as drawn by w0_gating, dotted green / red).
    Colours: style.ODR_W0_OUT_COLOURS (swapped into shared_plots while B is drawn).
    Built on shared_plots.make_figure3; positions in inches. Extra keywords go to make_figure3.
    Returns (fig, r, pipe); fig.table = the numbers of panel C."""
    W, H = width, height
    letter_y = H - 0.25
    axis_fs = axis_fs or fs - 1
    layout = dict(width=W, height=H, fs=fs, raster_fs=fs - 1,
                  raster_rect=(0.66, 0.9, raster_right, H - 0.7),
                  plot_box=b_box,
                  scheme_box=(3.86, 2.12, 1.55, 0.62),
                  legend_xy=(W + 5, 1.0),                  # B's own legend: off the page, replaced below
                  letter_y=letter_y, letter_b_x=3.78)
    layout.update(kw)
    old_dir, old_cols = P.ODR_CUE500_DIR, P.ODR_W0_OUT_COLOURS
    P.ODR_CUE500_DIR = results_dir                        # panel B reads its CSV from here
    P.ODR_W0_OUT_COLOURS = W0_OUT_COLOURS                 # and draws in the paper colours
    try:
        fig, r, pipe = P.make_figure3(**layout)
    finally:
        P.ODR_CUE500_DIR, P.ODR_W0_OUT_COLOURS = old_dir, old_cols
    for leg in list(fig.legends):                         # drop B's one-column legend
        leg.remove()
    if readout_colour:
        recolour_decoded(fig, readout_colour)
    ax_b = next(a for a in fig.axes if a.get_xlabel().startswith("cue"))

    # C: error against firing rate per cue presentation (short, top right)
    tab = rate_error_table(rate_df, first, rate)
    x, y, w, h = c_box
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    panel_rate_error(ax, tab, fs)
    ax.set_yticks([0, 25, 50])
    ax.set_xlabel("Average Firing\nRate (Hz)", labelpad=1)    # two lines: C sits at the page's right edge
    ax.set_ylabel("Average\nError (%)", labelpad=1)
    S.letter(fig, x - 0.62, letter_y, "C", W, H, fs)

    # B and C: ticks and axis labels at A's text size
    for a in (ax_b, ax):
        a.tick_params(labelsize=axis_fs)
        a.xaxis.label.set_fontsize(axis_fs)
        a.yaxis.label.set_fontsize(axis_fs)

    # one legend for B and C: two columns under C, right of B, framed
    rend = fig.canvas.get_renderer()
    low = ax.get_tightbbox(rend).y0 / fig.dpi                  # bottom of C's x label (in)
    pad = 0.07                                                 # frame padding (S.legend_frame)
    leg = fig.legend(handles=_network_legend_handles(compact=True), loc="upper right",
                     bbox_to_anchor=((x + w - pad) / W, (low - legend_gap - pad) / H), ncol=2,
                     fontsize=fs - 2, frameon=False, handlelength=1.8, columnspacing=0.8, labelspacing=0.3,
                     handletextpad=0.35, borderpad=0, borderaxespad=0)
    for t in leg.get_texts()[::4]:                             # the two column headers
        t.set_color("#333333")
    if frame:
        fig.legend_frame = S.legend_frame(fig, [leg], pad_in=pad)
    bb = leg.get_window_extent(rend)
    b_right = ax_b.get_tightbbox(rend).x1 / fig.dpi
    if bb.y0 / fig.dpi - pad < 0.02 or bb.x0 / fig.dpi - pad < b_right + 0.04:
        print(f"note: the legend ({bb.x0 / fig.dpi - pad:.2f} in) reaches into B ({b_right:.2f} in) "
              "or off the page; move c_box or shorten the labels")
    fig.table = tab
    return fig, r, pipe


# ═════════════════════════════════════════════════════════════════════════════
# Figure: network schematic + Mexican hat
# ═════════════════════════════════════════════════════════════════════════════

def mexican_hat(n_excit=N_EXCIT, sigma_deg=20.0, j_pos=1.6):
    """E -> E weight against the distance between preferred angles: the same formula as
    ODR/network_architecture.mexican_hat_kernel (Compte et al. 2000), averaging to 1 over the ring.
    Returns (delta_deg from -180 to 180, J, J-)."""
    from math import erf, sqrt, pi
    tmp = sqrt(2 * pi) * sigma_deg * erf(180.0 / sqrt(2.0) / sigma_deg) / 360.0
    j_neg = (1.0 - j_pos * tmp) / (1.0 - tmp)
    delta = np.arange(-n_excit // 2, n_excit // 2 + 1) * 360.0 / n_excit
    return delta, j_neg + (j_pos - j_neg) * np.exp(-0.5 * delta ** 2 / sigma_deg ** 2), j_neg


def _arrow(ax, p0, p1, colour, style="-|>", lw=1.1, ms=8, rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=colour, lw=lw, mutation_scale=ms,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def panel_ring(ax, fs=S.FS, n_ring=24, cue_deg=90.0, show_rf=False):
    """Ring of E neurons (bistable, blue), I population in the middle (LIF, black), E -> I (arrow),
    I -> E (bar), cue as a green arrow onto its preferred neurons. show_rf=True: the neurons outside
    the receptive field around the cue in grey (w0 = 0 outside, as during the hold)."""
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#C8C8C8", lw=0.8, zorder=1)
    ang = np.arange(n_ring) * 360.0 / n_ring
    dist = np.abs((ang - cue_deg + 180) % 360 - 180)
    cols = [C_E if (not show_rf or d <= RF_RADIUS) else C_W0_ZERO for d in dist]
    ax.scatter(np.cos(np.radians(ang)), np.sin(np.radians(ang)), s=30, c=cols, edgecolors="white",
               linewidths=0.6, zorder=3)
    # I population: a small cluster in the middle
    pts = [(0.0, 0.0)] + [(0.17 * np.cos(a), 0.17 * np.sin(a)) for a in np.radians(np.arange(0, 360, 60) + 30)]
    ax.scatter(*zip(*pts), s=20, color=C_I, edgecolors="white", linewidths=0.5, zorder=3)
    ax.add_patch(Circle((0, 0), 0.32, fill=False, ec=C_I, lw=0.7, ls=(0, (2, 1.5)), zorder=2))
    # E -> I (excitatory) and I -> E (inhibitory), uniform all-to-all
    a_ei, a_ie = np.radians(215), np.radians(325)
    _arrow(ax, (0.9 * np.cos(a_ei), 0.9 * np.sin(a_ei)), (0.36 * np.cos(a_ei), 0.36 * np.sin(a_ei)), C_E)
    _arrow(ax, (0.36 * np.cos(a_ie), 0.36 * np.sin(a_ie)), (0.88 * np.cos(a_ie), 0.88 * np.sin(a_ie)), C_I,
           style="-[", ms=4)
    ax.text(0.62 * np.cos(a_ei) - 0.06, 0.62 * np.sin(a_ei) + 0.1, "E→I", fontsize=fs - 2, color=C_E,
            ha="right", va="bottom")
    ax.text(0.62 * np.cos(a_ie) + 0.06, 0.62 * np.sin(a_ie) + 0.1, "I→E", fontsize=fs - 2, color=C_I,
            ha="left", va="bottom")
    # cue
    c = np.radians(cue_deg)
    _arrow(ax, (1.62 * np.cos(c), 1.62 * np.sin(c)), (1.14 * np.cos(c), 1.14 * np.sin(c)), S.C_CUE,
           lw=1.8, ms=10, z=4)
    ax.text(1.68 * np.cos(c), 1.68 * np.sin(c), "cue", fontsize=fs - 1, color=S.C_CUE, ha="center",
            va="bottom", fontweight="bold")
    # labels
    ax.text(0, -0.42, f"I ({N_INHIB})", fontsize=fs - 2, color=C_I, ha="center", va="top")
    ax.text(-0.8, -0.86, f"E ({N_EXCIT})", fontsize=fs - 2, color=C_E, ha="right", va="top")
    for a, lab, ha, va in ((0, "0°", "left", "center"), (180, "180°", "right", "center"),
                           (270, "270°", "center", "top")):
        r = 1.16
        ax.text(r * np.cos(np.radians(a)), r * np.sin(np.radians(a)), lab, fontsize=fs - 2, color="#777777",
                ha=ha, va=va)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.5, 1.95)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_mexican_hat(ax, fs=S.FS):
    """E -> E weight J against the difference of preferred angles (J+ = 1.6, sigma = 20 deg, mean 1)."""
    d, J, j_neg = mexican_hat()
    ax.plot(d, J, color=C_E, lw=1.4, zorder=3)
    ax.fill_between(d, 1, J, where=J >= 1, color=C_E, alpha=0.25, lw=0, zorder=2)
    ax.fill_between(d, 1, J, where=J < 1, color=C_I, alpha=0.12, lw=0, zorder=2)
    ax.axhline(1, color="#999999", lw=0.7, ls=":", zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["−180°", "", "0°", "", "180°"])
    ax.set_yticks([j_neg, J.max()])
    ax.set_yticklabels([r"$J^-$", r"$J^+$"])
    ax.set_ylim(j_neg - 0.14, J.max() + 0.1)
    ax.set_xlabel(r"$\Delta\theta$ (preferred angle)", fontsize=fs - 1, labelpad=1)
    ax.set_title("E→E weight", fontsize=fs - 1, pad=3)
    ax.text(38, 1.42, "local\nexcitation", fontsize=fs - 2, color=C_E, ha="left", va="center", linespacing=1.0)
    ax.text(178, j_neg - 0.02, "lateral inhibition", fontsize=fs - 2, color="#555555", ha="right", va="top")
    ax.text(-178, 1.01, "mean", fontsize=fs - 2, color="#999999", ha="left", va="bottom")
    S.style_ax(ax, fs)


def make_figure_network(fs=S.FS, width=S.COL_W, height=1.95, cue_deg=90.0, show_rf=False):
    """A: the ring network (schematic). B: its E -> E connectivity profile. One column wide."""
    W, H = width, height
    fig = plt.figure(figsize=(W, H))
    ax_r = fig.add_axes([0.0, 0.0, 1.85 / W, 1.0])
    panel_ring(ax_r, fs=fs, cue_deg=cue_deg, show_rf=show_rf)
    ax_m = fig.add_axes([2.28 / W, 0.48 / H, 1.12 / W, 1.0 / H])
    panel_mexican_hat(ax_m, fs=fs)
    S.letter(fig, 0.02, H - 0.25, "A", W, H, fs)
    S.letter(fig, 1.88, H - 0.25, "B", W, H, fs)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure: error against firing rate per cue presentation
# ═════════════════════════════════════════════════════════════════════════════

def load_loop_rates(results_dir=RESULTS):
    path = os.path.join(results_dir, "cue500_repeats_looprates.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python exp_loop_rates.py` in ODR/ first")
    return pd.read_csv(path)


def rate_error_table(df=None, first=2, rate="rate_loop_hz"):
    """Per network: error (% of 180 deg) and firing rate per cue presentation, first averaged over the
    memories first..10 of each trial, then mean and s.d. over the trials (16 per network)."""
    df = load_loop_rates() if df is None else df
    d = df[df.memory >= first].copy()
    d["error_pct"] = d.error_deg / 180.0 * 100.0
    per_trial = d.groupby(["setup", "batch", "trial"])[["error_pct", rate, "rate_loop_e_hz", "rate_loop_i_hz"]].mean()
    g = per_trial.groupby("setup")
    out = pd.DataFrame({"error % (mean)": g.error_pct.mean(), "error % (s.d.)": g.error_pct.std(),
                        "rate Hz (mean)": g[rate].mean(), "rate Hz (s.d.)": g[rate].std(),
                        "E rate Hz": g.rate_loop_e_hz.mean(), "I rate Hz": g.rate_loop_i_hz.mean(),
                        "trials": g.size()})
    return out.reindex([n for n, *_ in NETWORKS])


def panel_rate_error(ax, tab=None, fs=S.FS):
    """Decoding error (% of the largest error, 180 deg) against the mean firing rate of all neurons
    (E + I) over one cue presentation (forget window -> cue -> distractor -> readout -> next forget
    window), memories 2-10, mean +- s.d. over trials. Colour = w0 outside the RF, filled = pulse,
    open = no pulse. No legend: the trial figure has one for B and C."""
    tab = rate_error_table() if tab is None else tab
    for name, lab, w0o, pulse in NETWORKS:
        if name not in tab.index or np.isnan(tab.loc[name, "error % (mean)"]):
            continue
        t, col = tab.loc[name], W0_OUT_COLOURS[w0o]
        ax.errorbar(t["rate Hz (mean)"], t["error % (mean)"], xerr=t["rate Hz (s.d.)"], yerr=t["error % (s.d.)"],
                    fmt="o", ms=4.5, color=col, mfc=col if pulse else "white", mec=col, mew=1.1,
                    elinewidth=0.8, capsize=0, label=lab, zorder=3 if pulse else 2)
    ax.axhline(50, color="#999999", lw=0.8, ls=":", zorder=1)        # random readout: mean error 90 deg
    ax.text(0.99, 50, "chance", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=fs - 2, color="#777777")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("Average Firing Rate (Hz)", fontsize=fs, labelpad=1)
    ax.set_ylabel("Average Error (%)", fontsize=fs, labelpad=1)
    S.style_ax(ax, fs)
