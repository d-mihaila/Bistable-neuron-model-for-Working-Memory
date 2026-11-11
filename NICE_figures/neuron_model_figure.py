"""
neuron_model_figure.py -- Fig. 1 of the paper: the bistable neuron. Saved as figures/fig1_neuron_model.pdf / .png.

Layout (ONE COLUMN, 3.5 in wide; sizes in inches):
    A          A = phase planes at I_ext = 0, IF-like (w0 = +1, left) and bistable (w0 = -1, right),
    B          drawn by shared_plots._phase_pair, with a larger key underneath;
               B = the bistable neuron under a negative, then a positive pulse, twice on one time axis:
               "simulation" (upper pair: ../single_neuron/neuron_figures.simulate_pulses) and "hardware"
               (lower pair: a mixed-feedback silicon neuron, recorded by Loris Mendolia, fig1_c.csv, panel_loris; input current in nA from the input voltage, output in
               nA, 1 V = 1 nA, standing for v), each named above its top-left corner. Pulse onsets match
               (0.5 s, 1.6 s), so the blue / green shading and the "rest" / "persistent activity" labels
               serve both. A blue dotted arrow runs straight down from the bistable box of A to the
               simulated trace.

The equations are not drawn: they are in the paper (Eqs. 1-5).

Run:  python neuron_model_figure.py      (or the notebook fig1_neuron_model.ipynb)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch

import style as S

nf, P = S.nf, S.P

# model protocol, pulse ONSETS as on the Loris chip (negative at 0.5 s, positive at 1.6 s from the start of the
# shown hardware trace): rest = 0.5-1.6 s in both. The model needs 200 ms pulses to switch (100 ms, as on the
# chip, does not silence it).
PULSES = dict(t_start=500, pulse_ms=200, gap_ms=900, t_end=2400)
LORIS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_c.csv")
# Loris chip, same protocol as B: time_s, VIN0_V (input voltage), ch1_V (neuron output, read as a voltage:
# 1 V = 1 nA; the output is a current that stands for the membrane potential)
LORIS_INPUT_NA = {0.0: 0.0, 0.31: 0.2, 0.35: 0.45}       # input voltage level (V) -> input current (nA)
# (0 V = 0 A, 0.3 V = 200 pA, 0.35 V = 450 pA; the recording holds at 0.31 V)


def panel_pulses(fig, W, H, x, y, w, h, w0=-1.0, fs=S.FS, protocol=PULSES, in_frac=0.38, time_axis=True,
                 state_labels=True, label_fs=None):
    """I_ext(t) on top, v(t) below, in a box of w x h inches with its lower-left corner at (x, y); in_frac
    of the height for I_ext. Shading: light blue = persistent activity, green = rest (the colours of the
    phase planes). time_axis=False: ticks without numbers and no "time (s)" (a panel under it has them).
    state_labels: "rest" / "persistent activity" above, at label_fs (default fs - 2), bold."""
    sim = nf.simulate_pulses(w0=w0, **protocol)
    t, I, v, st = sim["t"], sim["I"], sim["v"], sim["spike_times"]
    n0, _ = sim["neg"]
    p0, p1 = sim["pos"]
    # the picture must not lie: silent between the pulses, spiking before and after
    assert not np.any((st > n0 + 150) & (st < p0)), "neuron fired during the silent stretch"
    assert np.any(st > p1) and np.any(st < n0), "neuron should spike before and after"

    gap = 0.06
    h_v = (1 - in_frac) * (h - gap)
    h_I = h - gap - h_v
    ax_v = fig.add_axes([x / W, y / H, w / W, h_v / H])
    ax_I = fig.add_axes([x / W, (y + h_v + gap) / H, w / W, h_I / H])

    c_act = S.C_TRAJ
    c_act_txt = tuple(0.72 * c for c in to_rgb(S.C_TRAJ))   # darker, so the text is readable
    for ax in (ax_I, ax_v):
        ax.axvspan(t[0], n0, color=c_act, alpha=0.16, lw=0)
        ax.axvspan(n0, p0, color=S.C_REST, alpha=0.18, lw=0)
        ax.axvspan(p0, t[-1], color=c_act, alpha=0.16, lw=0)
        ax.set_xlim(t[0], t[-1])
        S.style_ax(ax, fs)

    ax_I.plot(t, I, color=S.C_INPUT, lw=1.4, drawstyle="steps-post")
    ax_I.set_ylim(-1.6, 1.6)
    ax_I.set_yticks([-1, 1])
    ax_I.set_yticklabels(["−1", "+1"])
    ax_I.set_ylabel(r"$I_{ext}$", fontsize=fs, labelpad=0)
    ax_I.tick_params(labelbottom=False, bottom=False)

    ax_v.plot(t, v, color=S.C_VTRACE, lw=0.5)
    ax_v.plot(st, v[st], "o", color="black", ms=1.1, mew=0)
    ax_v.set_ylim(-1.1, 1.5)
    ax_v.set_yticks([-1, 0, 1])
    ax_v.set_yticklabels(["−1", "0", "1"])
    ax_v.set_ylabel(r"$v$", fontsize=fs, labelpad=0)
    ax_v.set_xticks([0, 1000, 2000])
    if time_axis:
        ax_v.set_xticklabels(["0", "1", "2"])
        ax_v.set_xlabel("time (s)", fontsize=fs, labelpad=0)
    else:
        ax_v.tick_params(labelbottom=False)

    # what the neuron does in each stretch, above the input trace
    if state_labels:
        top = (y + h + 0.04) / H
        lab = dict(ha="center", va="bottom", fontsize=label_fs or fs - 2, fontweight="bold", linespacing=1.0)
        for x_ms, txt, col in (((n0 + p0) / 2, "rest", S.C_REST),
                               ((p0 + t[-1]) / 2, "persistent\nactivity", c_act_txt)):
            frac = (x_ms - t[0]) / (t[-1] - t[0])
            fig.text((x + frac * w) / W, top, txt, color=col, **lab)
    return ax_I, ax_v


def load_loris(path=LORIS_CSV):
    """The Loris chip recording of panel C: t (s), input voltage (V), neuron output (nA; 1 V = 1 nA)."""
    import pandas as pd
    d = pd.read_csv(path)
    return d.time_s.values, d.VIN0_V.values, d.ch1_V.values * 1.0      # V -> nA (1 V = 1 nA)


def panel_loris(fig, W, H, x, y, w, h, fs=S.FS, path=LORIS_CSV, spike_nA=0.5, labels=False, in_frac=0.45,
                from_settled=True, t_end=None, time_axis=True):
    """C: the same experiment on the Loris chip, drawn as panel_pulses draws B: input current on top
    (nA, each input voltage level converted with LORIS_INPUT_NA),
    neuron output (nA, stands for v) below, in a w x h inch box with its lower-left corner at (x, y).
    Shading as in B: light blue = spiking (start to the negative pulse; positive pulse to the end),
    green = rest (negative pulse to positive pulse), found from the input. labels: also write "rest" /
    "persistent activity" above (B already names the colours). in_frac: share of the height for the input
    (more than B's 0.38: its label, with the unit, is longer).
    from_settled: drop the start of the recording (0.35 V for 0.5 s) and begin at t = 0 where the input
    first sits at its holding value (0.31 V), as B begins at I_ext = 0. t_end: last time shown (s, after that
    shift; None = to the end). make_figure passes B's duration, so B and C share the same time axis."""
    t, vin, out = load_loris(path)
    base = np.median(vin)                                  # the holding input (0.31 V)
    if from_settled:
        i0 = np.argmax(np.abs(vin - base) < 0.005)         # first sample at the holding input
        t, vin, out = t[i0:] - t[i0], vin[i0:], out[i0:]
    if t_end is not None:
        keep = t <= t_end
        t, vin, out = t[keep], vin[keep], out[keep]
    i_neg = np.argmax(vin < base - 0.05)                   # negative pulse: input drops to 0 V
    i_pos = i_neg + np.argmax(vin[i_neg:] > base + 0.01)   # positive pulse: input up to 0.35 V
    n0, p0 = t[i_neg], t[i_pos]
    up = np.nonzero((out[1:] >= spike_nA) & (out[:-1] < spike_nA))[0] + 1
    st = t[up]                                             # spike times (upward crossings)
    # the picture must not lie: silent between the pulses, spiking before and after
    assert not np.any((st > n0 + 0.15) & (st < p0)), "Loris neuron fired during the silent stretch"
    assert np.any(st > p0) and np.any(st < n0), "Loris neuron should spike before and after"

    gap = 0.06
    h_v = (1 - in_frac) * (h - gap)
    h_I = h - gap - h_v
    ax_v = fig.add_axes([x / W, y / H, w / W, h_v / H])
    ax_I = fig.add_axes([x / W, (y + h_v + gap) / H, w / W, h_I / H])
    c_act = S.C_TRAJ
    for ax in (ax_I, ax_v):
        ax.axvspan(t[0], n0, color=c_act, alpha=0.16, lw=0)
        ax.axvspan(n0, p0, color=S.C_REST, alpha=0.18, lw=0)
        ax.axvspan(p0, t[-1], color=c_act, alpha=0.16, lw=0)
        ax.set_xlim(t[0], t[-1])
        S.style_ax(ax, fs)

    levels = np.array(sorted(LORIS_INPUT_NA))
    k = np.abs(vin[:, None] - levels[None, :]).argmin(axis=1)
    assert np.all(np.abs(vin - levels[k]) < 0.005), "input voltage not one of the calibrated levels"
    i_in = np.array([LORIS_INPUT_NA[v] for v in levels])[k]           # nA
    ax_I.plot(t, i_in, color=S.C_INPUT, lw=1.4, drawstyle="steps-post")
    ax_I.set_ylim(-0.07, 0.53)
    ax_I.set_yticks([0, 0.2, 0.45])
    ax_I.set_yticklabels(["0", "0.2", "0.45"])
    ax_I.set_ylabel(r"$I_{ext}$ (nA)", fontsize=fs, labelpad=1)
    ax_I.tick_params(labelbottom=False, bottom=False)

    ax_v.plot(t, out, color=S.C_VTRACE, lw=0.5)
    ax_v.set_ylim(-0.12, 1.15)
    ax_v.set_yticks([0, 1])
    ax_v.set_ylabel(r"$v$ (nA)", fontsize=fs, labelpad=1)
    ax_v.set_xticks(np.arange(0, t[-1] + 1e-9, 1.0))
    if time_axis:
        ax_v.set_xlabel("time (s)", fontsize=fs, labelpad=0)
    else:
        ax_v.tick_params(labelbottom=False)

    if labels:
        c_act_txt = tuple(0.72 * c for c in to_rgb(S.C_TRAJ))
        top = (y + h + 0.04) / H
        lab = dict(ha="center", va="bottom", fontsize=fs - 2, fontweight="bold", linespacing=1.0)
        for x_s, txt, col in (((n0 + p0) / 2, "rest", S.C_REST),
                              ((p0 + t[-1]) / 2, "persistent\nactivity", c_act_txt)):
            fig.text((x + (x_s - t[0]) / (t[-1] - t[0]) * w) / W, top, txt, color=col, **lab)
    return ax_I, ax_v


def _group_label(fig, W, H, x, y0, y1, text, fs=S.FS, colour="#333333", lw=0.8, tick=0.04):
    """A bracket "[" at x (inches) from y0 to y1 with `text` written up its left side (rotated)."""
    from matplotlib.lines import Line2D
    xs, ys = [x + tick, x, x, x + tick], [y0, y0, y1, y1]
    fig.add_artist(Line2D([v / W for v in xs], [v / H for v in ys], transform=fig.transFigure,
                          color=colour, lw=lw, solid_joinstyle="miter"))
    return fig.text((x - 0.05) / W, (y0 + y1) / 2 / H, text, rotation=90, ha="right", va="center",
                    fontsize=fs, color=colour)


def _align_ylabels(fig, axes, x, w, ys=None):
    """Put the y labels of axes that share the same left edge x and width w (inches) in one column: each
    label's right edge where the leftmost one is (fig.align_ylabels only aligns subplots, not add_axes).
    ys: the height of each label's centre (axes fraction; default 0.5), to keep long labels apart."""
    rend = fig.canvas.get_renderer()
    for a in axes:
        a.yaxis.get_tightbbox(rend)                       # places each label next to its tick labels first
    right = min(a.yaxis.label.get_window_extent(rend).x1 for a in axes) / fig.dpi   # inches
    for a, y in zip(axes, ys or [0.5] * len(axes)):
        a.yaxis.set_label_coords((right - x) / w, y)


def _frames(fig, W, H):
    """The dotted frames drawn by shared_plots._phase_pair, left to right, as (x, y, w, h) in inches.
    Read from the figure, so the arrow stays on the box if the frame margins change there."""
    boxes = [a for a in fig.artists if isinstance(a, FancyBboxPatch)]
    return sorted((b.get_x() * W, b.get_y() * H, b.get_width() * W, b.get_height() * H) for b in boxes)


def _ylabel_left(fig, ax):
    """Left edge (inches) of the y label of ax, as it will be drawn."""
    bb = ax.yaxis.get_tightbbox(fig.canvas.get_renderer())    # places the label first
    return bb.x0 / fig.dpi


def enlarge_key(fig, W, H, scale=1.3, fs=S.FS, ncol=3, x_shift=0.0, y_shift=0.0):
    """Redraw the key of A (made by _phase_pair at fs - 2) larger: text fs - 1, line widths and markers
    x scale, at the same anchor moved by (x_shift, y_shift) inches. Returns the new legend."""
    old = fig.legends[0]
    handles = list(getattr(old, "legend_handles", None) or old.legendHandles)
    labels = [t.get_text() for t in old.get_texts()]
    anchor = old.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    old.remove()
    for h in handles:
        h.set_linewidth(h.get_linewidth() * scale)
        h.set_markersize(h.get_markersize() * scale)
    return fig.legend(handles=handles, labels=labels, loc="upper center",
                      bbox_to_anchor=(anchor.x0 + x_shift / W, anchor.y0 + y_shift / H), ncol=ncol, frameon=False,
                      fontsize=fs - 1, handlelength=1.5, columnspacing=1.0, handletextpad=0.4,
                      labelspacing=0.35, borderpad=0, borderaxespad=0)


def path_arrow(fig, W, H, points, colour, radius=0.15, lw=0.9, ls=(0, (1.2, 2.0)), head=0.07):
    """Dotted arrow along axis-parallel straight segments through `points` (inches from the lower-left
    corner), with rounded corners of the given radius and a solid head at the last point, pointing
    along the last segment. Dotted like the frames of A."""
    pts = [np.asarray(p, float) for p in points]
    unit = lambda a, b: (b - a) / np.linalg.norm(b - a)
    d_last = unit(pts[-2], pts[-1])
    end_line = pts[-1] - 0.6 * head * d_last                  # the dotted line stops inside the head
    verts, codes = [pts[0]], [Path.MOVETO]
    for a, p, b in zip(pts[:-2], pts[1:-1], pts[2:]):
        r = min(radius, np.linalg.norm(p - a) / 2, np.linalg.norm(b - p) / 2)
        verts += [p - r * unit(a, p), p, p + r * unit(p, b)]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(end_line)
    codes.append(Path.LINETO)
    path = Path([(x / W, y / H) for x, y in verts], codes)
    fig.add_artist(PathPatch(path, fill=False, ec=colour, lw=lw, ls=ls, transform=fig.transFigure))
    tail = pts[-1] - head * d_last
    fig.add_artist(FancyArrowPatch((tail[0] / W, tail[1] / H), (pts[-1][0] / W, pts[-1][1] / H),
                                   transform=fig.transFigure, arrowstyle="-|>,head_length=0.5,head_width=0.22",
                                   mutation_scale=10, color=colour, lw=0, shrinkA=0, shrinkB=0))


def panel_placeholder(ax, text, fs=S.FS):
    """Empty panel with a light dashed frame and `text` in its centre (until the real panel exists)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#BBBBBB")
        sp.set_linestyle((0, (3, 2)))
        sp.set_linewidth(0.8)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=fs,
            color="#777777")


def make_figure(w0s=(+1.0, -1.0), w0_pulse=-1.0, fs=S.FS, width=S.COL_W,
                phase_left=0.47, side=1.25, phase_gap=0.44,
                b_left=0.55, right=0.06, b_bottom=0.36, trace_h=0.82, in_frac=0.42, pair_gap=0.34,
                key_scale=1.3, key_shift=(0.0, 0.0), key_h=0.5, arrow_x=0.08, arrow_radius=0.1,
                loris_csv=LORIS_CSV, starts=None):
    """Figure 1, one column wide: A on top, B under it.
    A: the two I_ext = 0 phase planes in the order of w0s (default IF-like left, bistable right; side x side
    inches each, dotted frames, larger key underneath, about key_h tall). B: the bistable neuron (w0 = w0_pulse) under the same pulse protocol, simulated (upper pair,
    panel_pulses) and on the Loris chip (lower pair, panel_loris, data loris_csv): each pair trace_h tall
    (I_ext taking in_frac), pair_gap apart, from b_left to `right` inches from the edge, the lower pair from
    b_bottom up. One time axis (0-2.4 s), at the bottom; the shading (blue spiking, green rest: 0.5-1.6 s in
    both) and the "rest" / "persistent activity" labels above serve both. "simulation" and "hardware" are
    written above the top-left corner of their pair.
    A dotted arrow in the bistable blue leaves the bottom of the bistable box. With the bistable plane on the
    right (default) it runs straight down, right of the key (arrow_x inches in from the box's right edge),
    onto the simulated trace; on the left it runs down left of the key and turns into "simulation".
    Returns the figure; fig.ax_sim / fig.ax_hw = the (input, output) axes of each pair."""
    W = width
    y_hw = b_bottom
    y_sim = y_hw + trace_h + pair_gap
    b_top = y_sim + trace_h
    label_h = 0.3                                         # "persistent / activity" above B (two lines)
    phase_bottom = b_top + label_h + 0.12 + key_h + 0.42 - key_shift[1]   # the key hangs 0.42 in under A's plots
    y_letter = phase_bottom + side + 0.44                 # level with the frame names
    H = y_letter + 0.18
    fig = plt.figure(figsize=(W, H))

    # ---- A: phase planes (shared_plots._phase_pair), key larger ------------------------------
    names = tuple(("bistable", P.W0_COLOURS[-1.0]) if w0 < 0 else ("IF-like", P.W0_COLOURS[1.0]) for w0 in w0s)
    P._phase_pair(fig, W, H, phase_left, phase_bottom, side, phase_gap, w0s,
                  starts or P.FIG4_STARTS, fs, 3.5, 3, box_labels=names)
    enlarge_key(fig, W, H, scale=key_scale, fs=fs, x_shift=key_shift[0], y_shift=key_shift[1])
    frames = _frames(fig, W, H)

    # ---- B: simulation (upper pair) and Loris chip (lower pair), one time axis ----------------
    b_w = W - right - b_left
    sim = panel_pulses(fig, W, H, b_left, y_sim, b_w, trace_h, w0=w0_pulse, fs=fs,
                       in_frac=in_frac, time_axis=False, label_fs=fs)
    t_b = (PULSES["t_end"] - 1) / 1000                    # the model's last time (s): the chip trace ends there
    hw = panel_loris(fig, W, H, b_left, y_hw, b_w, trace_h, fs=fs, path=loris_csv, in_frac=in_frac,
                     t_end=t_b)
    # one column of y labels; each I_ext label a little up, each v label a little down: "I_ext (nA)" is
    # longer than its axis and would touch "v (nA)"
    _align_ylabels(fig, [*sim, *hw], b_left, b_w, ys=[0.62, 0.44, 0.62, 0.44])
    title = dict(ha="left", va="bottom", fontsize=fs, fontweight="bold", color="#333333")
    txt_sim = fig.text(b_left / W, (y_sim + trace_h + 0.04) / H, "simulation", **title)
    fig.text(b_left / W, (y_hw + trace_h + 0.04) / H, "hardware", **title)
    fig.ax_sim, fig.ax_hw = sim, hw

    # ---- arrow from the bistable box ----------------------------------------------------------
    rend = fig.canvas.get_renderer()
    i_b = int(np.argmin(w0s))                             # the bistable plane (w0 < 0)
    fx, fy, fw, _ = frames[i_b]
    key = fig.legends[0].get_window_extent(rend)
    if i_b == len(w0s) - 1:                               # right: straight down, right of the key, onto the trace
        x_lane = fx + fw - arrow_x
        path_arrow(fig, W, H, [(x_lane, fy), (x_lane, b_top + 0.02)], S.C_BISTABLE, radius=arrow_radius)
        if key.x1 / fig.dpi > x_lane - 0.06:
            print(f"note: the arrow ({x_lane:.2f} in) runs into the key ({key.x1 / fig.dpi:.2f} in); "
                  "lower key_shift[0]")
    else:                                                 # left: down, left of the key, into "simulation"
        bb = txt_sim.get_window_extent(rend)
        y_end = (bb.y0 + bb.y1) / 2 / fig.dpi
        x_start = fx + arrow_x
        path_arrow(fig, W, H, [(x_start, fy), (x_start, y_end), (bb.x0 / fig.dpi - 0.04, y_end)], S.C_BISTABLE,
                   radius=arrow_radius)
        if key.x0 / fig.dpi < x_start + 0.06:
            print(f"note: the arrow ({x_start:.2f} in) runs into the key ({key.x0 / fig.dpi:.2f} in); "
                  "raise key_shift[0]")

    # ---- panel letters ----------------------------------------------------------------------
    S.letter(fig, 0.02, y_letter, "A", W, H, fs)
    S.letter(fig, 0.02, y_sim + trace_h + 0.04, "B", W, H, fs)
    return fig


if __name__ == "__main__":
    S.apply()
    fig = make_figure()
    print(S.save(fig, "fig1_neuron_model"))


