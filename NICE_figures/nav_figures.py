"""
nav_figures.py -- Fig. 4 of the paper: navigation from a single cue (model and task in ../navigation,
which this file only calls).

    A  the network: ring of 16 E neurons, cue, decision neurons E N W S with mutual inhibition,
       the two distance counters (NS, EW), each inhibiting the decision neurons of its axis
    B  the board around one apple at 45 deg (6 x 6 cells), with the steps taken
    C  that trial: E neurons on their w0 map (ODR colours and w0 bar), decision neurons (rows of
       both N, E, S, W from the top), distance counters (staircase until a counter fires, then one dot per spike at threshold)
    D  targets ('apples' in the code) per 10,000 spikes of the three networks (all bistable, heterogeneous, all LIF)
    E  the same for the two bistable rings against weight noise (CV)
    F  the same for the two bistable rings with / without RF and forget
       D-F: mean over the runs (10 seeds) +- s.d. across runs, from navigation/results/runs.csv;
       y label "targets reached" on D only (D-F share it); "per 10,000 spikes" is said in the caption.

Size: full text width, 7.16 x 3.3 in, drawn at its real size (SCALE = 1): every
text at TEXT_FS (7 pt), panel letters as in the other figures. A-C as before, a little smaller;
D-F shorter, with F's key on its right. Include it at its natural size (= \textwidth) in LaTeX.

Colours (SCHEMES["paper"]): ring blue (bistable), cue green, inhibition red;
decision neurons and steps by axis (N/S orange, E/W reddish purple) in A, B and C alike;
counters dark grey, EW solid and NS dashed. D-F: the networks in their MNIST colours and lines
(style.NETWORKS): all bistable = w0 = -1 blue, solid; heterogeneous = purple, dotted; all LIF = black.
F: fill = with forget, open = no forget (as the filled / open ODR markers for pulse / no pulse),
hatched = no RF.

    import nav_figures as NV
    fig = NV.make_figure()                 # simulates the trial once (needs torch), then reads the cache
    S.save(fig, "fig4_navigation")
"""

import hashlib
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch

import style as S

SNAKE = os.path.join(S.ROOT, "navigation")
CACHE = os.path.join(S.HERE, "cache")
TRIAL = dict(net_type="all bistable", pos=(2, 2), apple=(5, 5), seed=0)   # apple at 45 deg: route EEENNN
TEXT_FS = S.FS - 2          # every text of this figure, ticks included
SCALE = 1.0                 # the figure is drawn at this fraction of the sizes in make_figure (1: real size, 7.16 in wide)
RUNS_CSV = os.path.join(SNAKE, "results", "runs.csv")
NETS = ["all bistable", "heterogeneous", "all LIF"]
NET_STYLE = {"all bistable": S.net("const_w0_-1.00"),                          # blue, solid, circle
             "heterogeneous": S.net("lif_hidden_bistable_output_w0_-1.00"),    # purple, dotted, square
             "all LIF": S.net("all_lif_wait100")}                              # black, dotted, x
NET_TICKS = {"all bistable": "all\nbistable", "heterogeneous": "hetero-\ngeneous", "all LIF": "all\nLIF"}
CONDS = ["RF + forget", "no RF", "no forget", "no RF, no forget"]
COND_FILL = {"RF + forget": (True, False), "no RF": (True, True),           # (filled = forget, hatched = no RF)
             "no forget": (False, False), "no RF, no forget": (False, True)}

DIR_NAMES = ["E", "N", "W", "S"]                  # decision neuron k = compass direction k (navigation.py order)
DIR_ANGLES = [0.0, 90.0, 180.0, 270.0]
ROW_ORDER = ["N", "E", "S", "W"]                  # panel C rows, top to bottom: clockwise from N
ROW_Y = {k: 3 - ROW_ORDER.index(d) for k, d in enumerate(DIR_NAMES)}   # decision neuron k -> its row (N = 3)
bearing = lambda a: (90.0 - np.asarray(a, dtype=float)) % 360.0         # angle (E = 0, N = 90) -> compass bearing
D_YLABEL = "targets reached"                      # y label of D (D-F share the axis); "per 10,000 spikes" is in the caption
AXIS = [0, 1, 0, 1]                               # decision E N W S -> counter 0 (EW) or 1 (NS)
COUNTER_NAMES = ["EW", "NS"]

C_RING = S.C_BISTABLE                             # bistable blue (E dots, as in the ODR scheme)
C_EXC = S.C_BISTABLE                              # excitatory arrows (ODR scheme: E->I arrow)
C_INH = "#B2182B"                                 # inhibitory wires and bar ends: red
C_CUE = S.C_CUE                                   # ODR cue green
C_FORGET = "#7b3fa0"                              # ODR negative pulse / forget (hatched)
INK, MUTED, C_BLOCK, C_GRID = "#222222", "#8c8c8c", "#e3e3e3", "#d9d9d9"
W0_CMAP = LinearSegmentedColormap.from_list("w0", ["#0072B2", "#d9d9d9"])   # ODR w0 map, its -1 .. 0 part

NS_C, EW_C = "#E69F00", "#CC79A7"                 # Okabe-Ito orange / reddish purple (CVD-checked with blue, green)
C_COUNTER = "#4D4D4D"                             # counters: neutral dark grey
SCHEMES = {
    # the paper: decision neurons and steps by axis, counters grey (EW solid, NS dashed)
    "paper": dict(dec=[EW_C, NS_C, EW_C, NS_C], dec_text=INK, counter=[C_COUNTER, C_COUNTER],
                  counter_ls=["-", (0, (3, 1.6))], route=[EW_C, NS_C, EW_C, NS_C], counter_text="white"),
    # two other colour schemes, kept for comparison
    "axis": dict(dec=[EW_C, NS_C, EW_C, NS_C], dec_text=INK, counter=[EW_C, NS_C],
                 counter_ls=["-", "-"], route=[EW_C, NS_C, EW_C, NS_C]),
    "type": dict(dec=["#111111"] * 4, dec_text="white", counter=["#E69F00", "#E69F00"],
                 counter_ls=["-", (0, (3, 1.6))], route=["#111111"] * 4),
}


# ═════════════════════════════════════════════════════════════════════════════
# the trial (simulated with ../navigation, cached)
# ═════════════════════════════════════════════════════════════════════════════

def _code_md5():
    """Fingerprint of the navigation code: a cached trial is only reused while the code is unchanged."""
    h = hashlib.md5()
    for f in ("network.py", "neuron_model.py", "board.py", "navigation.py"):
        with open(os.path.join(SNAKE, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _simulate(net_type, pos, apple, seed):
    """One apple (target) through navigation.run_trial, everything recorded. ../navigation has its own
    neuron_model / network / board modules (as MNIST, N_MNIST and ODR): they are imported from
    ../navigation and the previous modules put back afterwards."""
    names = ["neuron_model", "network", "board", "navigation"]
    saved = {m: sys.modules.pop(m) for m in names if m in sys.modules}
    sys.path.insert(0, SNAKE)
    try:
        import navigation as sf
        lg = sf.run_trial(net_type, pos=tuple(pos), apple=tuple(apple), seed=seed)
        dec_w0 = float(sf.new_net(net_type, seed).dec_w0) if sf.TYPES[net_type][1] == "bistable" else None
    finally:
        sys.path.remove(SNAKE)
        for m in names:
            sys.modules.pop(m, None)
        sys.modules.update(saved)
    t = lg.trials[0]
    rec, tm = t["rec"], t["rec"]["timing"]
    keep = {k: np.asarray(v) for k, v in rec.items() if hasattr(v, "shape")}
    keep.update(slots=[(s["t0"], s["t1"]) for s in rec["slots"]], T=rec["T"],
                enc_t=rec.get("enc_t", tm.cue_off), timing=dict(cue_on=tm.cue_on, cue_off=tm.cue_off))
    info = {k: t[k] for k in ("pos", "apple", "theta", "rf_center", "n_axis", "enc_start", "end", "shortest")}
    info["steps"] = [dict(move=s["move"], pos=s["pos"], collision=s["collision"]) for s in t["steps"]]
    info["slot_log"] = [dict(step=s["step"], closed=np.asarray(s["closed"]), winner=s["winner"])
                        for s in t["slot_log"]]
    return dict(rec=keep, info=info, angles=np.asarray(lg.angles_deg), meta=dict(lg.meta, dec_w0=dec_w0),
                net_type=net_type, seed=seed)


def get_trial(net_type=TRIAL["net_type"], pos=TRIAL["pos"], apple=TRIAL["apple"], seed=TRIAL["seed"],
              refresh=False):
    """The recorded trial: simulated the first time (needs torch), then read from NICE_figures/cache/.
    A change in the navigation code makes it simulate again."""
    os.makedirs(CACHE, exist_ok=True)
    tag = net_type.replace(" ", "_")
    path = os.path.join(CACHE, f"nav_trial_{tag}_{pos[0]}{pos[1]}_to_{apple[0]}{apple[1]}_seed{seed}.pkl")
    md5 = _code_md5()
    if os.path.exists(path) and not refresh:
        try:
            with open(path, "rb") as f:
                tr = pickle.load(f)
            if tr.get("md5") == md5:
                return tr
        except Exception as e:                                 # e.g. written by another numpy version
            print(f"cache not readable ({type(e).__name__}): simulating again")
    tr = _simulate(net_type, pos, apple, seed)
    tr["md5"] = md5
    with open(path, "wb") as f:
        pickle.dump(tr, f)
    path_info = tr["info"]
    print(f"simulated: {net_type}, seed {seed}, {tuple(pos)} -> {tuple(apple)}: "
          f"{''.join(DIR_NAMES[s['move']] for s in path_info['steps'])} ({path_info['end']})")
    return tr


# ═════════════════════════════════════════════════════════════════════════════
# results (navigation/results/runs.csv, one row per run)
# ═════════════════════════════════════════════════════════════════════════════

def load_results(path=RUNS_CSV, lif_tag=None):
    """The runs of the paper: bistable networks (lif_tag "-") and the LIF networks of one LIF setting.
    lif_tag None: the only LIF setting in the file (an error if there are several)."""
    import pandas as pd
    df = pd.read_csv(path)
    tags = sorted(set(df.lif_tag.astype(str)) - {"-"})
    if lif_tag is None:
        if len(tags) > 1:
            raise ValueError(f"several LIF settings in {path}: pass lif_tag= one of {tags}")
        lif_tag = tags[0] if tags else "-"
    return df[df.lif_tag.astype(str).isin(["-", lif_tag])]


def stat(df, by, how="mean"):
    """Apples per 10,000 spikes per group: (value, s.d. across runs). how="mean": mean over the runs;
    how="pooled": all apples / all spikes of the group (the number in navigation's tables)."""
    g = df.groupby(by, sort=False)
    sd = g.apples_per_10k.std().fillna(0)
    if how == "pooled":
        tot = g[["apples", "all_spikes"]].sum()
        return 1e4 * tot.apples / tot.all_spikes, sd
    return g.apples_per_10k.mean(), sd


def _value_label(ax, x, top, v, fs):
    ax.text(x, top + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=fs, color=INK)


def panel_networks(ax, df, fs=TEXT_FS, how="mean", ylim=(0, 33)):
    """D: the three networks (RF + forget, no noise), one bar each, error bar = s.d. across runs."""
    base = df[df.exp == "base"]
    v, sd = stat(base, "net_type", how)
    nets = [n for n in NETS if n in v.index]
    for i, n in enumerate(nets):
        lo = min(sd[n], v[n])                                    # the error bar stops at 0
        ax.bar(i, v[n], width=0.62, color=NET_STYLE[n]["colour"], lw=0, zorder=2)
        ax.errorbar(i, v[n], yerr=[[lo], [sd[n]]], color=INK, lw=0.8, capsize=2, capthick=0.8, zorder=3)
        _value_label(ax, i, v[n] + sd[n], v[n], fs)
    ax.set_xticks(range(len(nets)))
    ax.set_xticklabels([NET_TICKS[n] for n in nets], fontsize=fs, linespacing=1.0)
    ax.set_xlim(-0.55, len(nets) - 0.45)
    _results_axis(ax, fs, ylim)


def panel_weight_noise(ax, df, fs=TEXT_FS, how="mean", ylim=(0, 33), dx=0.8, legend=False):
    """E: the two bistable rings against weight noise (CV 0 = the base runs), error bars = s.d. across runs.
    The two networks are drawn dx % apart so their error bars do not hide each other. legend=False:
    no legend (the colours are named under the bars of D and F; in this narrow panel a legend covers
    the error bars)."""
    nets = ["all bistable", "heterogeneous"]
    d = df[df.net_type.isin(nets) & df.exp.isin(["base", "weight_noise"])]
    v, sd = stat(d, ["net_type", "cv_weight"], how)
    from matplotlib.lines import Line2D
    hs = []
    for k, n in enumerate(nets):
        st = NET_STYLE[n]
        cv = np.sort(v.loc[n].index.values)
        x = 100 * cv + (k - 0.5) * dx
        ax.errorbar(x, v.loc[n].loc[cv].values, yerr=sd.loc[n].loc[cv].values, color=st["colour"], ls=st["ls"],
                    lw=1.2, marker=st["marker"], ms=3.2, mew=0.8, capsize=1.5, capthick=0.7, elinewidth=0.7, zorder=3)
        hs.append(Line2D([], [], color=st["colour"], ls=st["ls"], lw=1.2, marker=st["marker"], ms=3.2, label=n))
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlim(-4, 54)
    ax.set_xlabel("weight noise, CV (%)", fontsize=fs, labelpad=1)
    if legend:
        ax.legend(handles=hs, loc="lower left", frameon=False, fontsize=fs, handlelength=2.0, borderaxespad=0.1,
                  handletextpad=0.4, labelspacing=0.25)
    _results_axis(ax, fs, ylim)


def _cond_bar(ax, x, h, col, cond, w):
    filled, hatched = COND_FILL[cond]
    ax.bar(x, h, width=w, color=col if filled else "white", edgecolor=col, lw=0.9,
           hatch="////" if hatched else None, zorder=2)
    if filled and hatched:                                       # white hatch on a filled bar
        ax.bar(x, h, width=w, color="none", edgecolor="white", lw=0, hatch="////", zorder=2.5)
        ax.bar(x, h, width=w, color="none", edgecolor=col, lw=0.9, zorder=2.6)


def panel_rf_forget(ax, df, fs=TEXT_FS, how="mean", ylim=(0, 33), legend_anchor=(0.5, -0.2),
                    legend_loc="upper center", legend_ncol=2):
    """F: the two bistable rings in the four RF x forget conditions (RF + forget = the base runs).
    Fill = network colour; filled = forget, open = no forget, hatched = no RF. Legend (grey): by default
    underneath in 2 x 2; make_figure puts it on the right in one column (legend_loc / legend_anchor /
    legend_ncol)."""
    nets = ["all bistable", "heterogeneous"]
    d = df[df.net_type.isin(nets) & df.exp.isin(["base", "rf_forget"])]
    v, sd = stat(d, ["net_type", "condition"], how)
    w, pitch, gap = 0.72, 0.86, 0.75
    centres = []
    for i, n in enumerate(nets):
        x0 = i * (len(CONDS) * pitch + gap)
        col = NET_STYLE[n]["colour"]
        for j, c in enumerate(CONDS):
            if (n, c) not in v.index:
                continue
            x = x0 + j * pitch
            _cond_bar(ax, x, v[(n, c)], col, c, w)
            lo = min(sd[(n, c)], v[(n, c)])
            ax.errorbar(x, v[(n, c)], yerr=[[lo], [sd[(n, c)]]], color=INK, lw=0.8, capsize=1.6, capthick=0.8,
                        zorder=3)
            _value_label(ax, x, v[(n, c)] + sd[(n, c)], v[(n, c)], fs)
        centres.append(x0 + (len(CONDS) - 1) * pitch / 2)
    ax.set_xticks(centres)
    ax.set_xticklabels(nets, fontsize=fs)
    ax.set_xlim(-0.85, centres[-1] + (len(CONDS) - 1) * pitch / 2 + 0.6)   # room for the first value label
    _results_axis(ax, fs, ylim)
    from matplotlib.patches import Patch
    grey = "#7a7a7a"
    hs = []
    for c in CONDS:
        filled, hatched = COND_FILL[c]
        hs.append(Patch(facecolor=grey if filled else "white", edgecolor="white" if filled and hatched else grey,
                        hatch="////" if hatched else None, lw=0.9, label=c))
    leg = ax.legend(handles=hs, loc=legend_loc, bbox_to_anchor=legend_anchor, ncol=legend_ncol, frameon=False,
                    fontsize=fs, handlelength=1.4, handleheight=0.9, columnspacing=1.2, handletextpad=0.4,
                    labelspacing=0.3, borderaxespad=0)
    for h in leg.legend_handles:                                 # grey outline round the hatched filled swatch
        h.set_edgecolor(grey)
    if hs:
        leg.legend_handles[1].set_edgecolor("white")             # "no RF": white hatch on grey
    return leg


def _results_axis(ax, fs, ylim):
    ax.set_ylim(*ylim)
    ax.set_yticks([0, 10, 20, 30])
    S.style_ax(ax, fs + 1)                    # tick labels at fs


# ═════════════════════════════════════════════════════════════════════════════
# A: network scheme
# ═════════════════════════════════════════════════════════════════════════════

def _u(a):
    return np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])


def _arrow(ax, p, q, col, lw=1.1, ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=col, lw=lw, mutation_scale=ms, shrinkA=0,
                                 shrinkB=0, zorder=3))


def _bar(ax, p, direction, col=C_INH, half=0.17, lw=1.4):
    """Inhibitory synapse: a short bar at p, across `direction`."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    n = np.array([-d[1], d[0]])
    ax.plot(*np.c_[p - half * n, p + half * n], color=col, lw=lw, solid_capstyle="butt", zorder=4)


def _wire(ax, pts, col=C_INH, lw=0.9, halo=False):
    """Inhibitory wire along pts, a bar at the end (the target)."""
    pts = np.asarray(pts, float)
    kw = dict(path_effects=[pe.Stroke(linewidth=lw + 3, foreground="white"), pe.Normal()]) if halo else {}
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, zorder=3, solid_joinstyle="miter", **kw)
    _bar(ax, pts[-1], pts[-1] - pts[-2], col)


def panel_scheme(ax, scheme="paper", fs=TEXT_FS, cue_deg=45.0):
    """Ring, cue (at cue_deg), decision neurons, distance counters. Text: only "cue", the directions,
    the counter names and "distance counter"."""
    sc = SCHEMES[scheme]
    ax.set_aspect("equal")
    ax.axis("off")
    # ring: 16 E neurons, neighbours excite each other
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=C_RING, lw=1.0, alpha=0.45, zorder=1)
    for a in np.arange(16) * 22.5:
        ax.add_patch(Circle(_u(a), 0.11, fc=C_RING, ec="white", lw=0.4, zorder=3))
    # decision neurons, each excited by the ring around its direction
    R, rd = 2.2, 0.37
    for k, (name, a) in enumerate(zip(DIR_NAMES, DIR_ANGLES)):
        p = R * _u(a)
        ax.add_patch(Circle(p, rd, fc=sc["dec"][k], ec="#111111", lw=0.7, zorder=5))
        ax.text(*p, name, ha="center", va="center", fontsize=fs, fontweight="bold", color=sc["dec_text"],
                zorder=6)
        _arrow(ax, 1.2 * _u(a), (R - rd - 0.04) * _u(a), C_EXC)
    # mutual inhibition between neighbouring decision neurons (bar at both ends)
    for a in DIR_ANGLES:
        a0, a1 = a + 15, a + 75
        ax.add_patch(Arc((0, 0), 2 * R, 2 * R, theta1=a0, theta2=a1, color=C_INH, lw=0.9, zorder=2))
        _bar(ax, R * _u(a0), _u(a0 + 90))
        _bar(ax, R * _u(a1), _u(a1 + 90))
    # distance counters, each inhibiting the two decision neurons of its axis
    ns, ew = np.array([-3.3, 3.25]), np.array([3.3, -3.25])
    bw, bh = 1.25, 0.62
    for j, c in ((1, ns), (0, ew)):
        ax.add_patch(FancyBboxPatch((c[0] - bw / 2, c[1] - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=sc["counter"][j], ec="#111111", lw=0.7, zorder=5))
        ax.text(*c, COUNTER_NAMES[j], ha="center", va="center", fontsize=fs, fontweight="bold",
                color=sc.get("counter_text", INK), zorder=6)
    ax.text(ns[0] - bw / 2, ns[1] + bh / 2 + 0.14, "distance counter", ha="left", va="bottom", fontsize=fs,
            color=INK)
    ax.text(ew[0] + bw / 2, ew[1] - bh / 2 - 0.14, "distance counter", ha="right", va="top", fontsize=fs,
            color=INK)
    top = R + rd
    _wire(ax, [ns + [bw / 2, 0], [0, ns[1]], [0, top + 0.04]])                        # NS -> N
    _wire(ax, [ns - [0, bh / 2], [ns[0], -R], [-rd - 0.04, -R]])                      # NS -> S
    _wire(ax, [ew + [0, bh / 2], [ew[0], 0], [top + 0.04, 0]])                         # EW -> E
    _wire(ax, [ew - [bw / 2, 0], [-R, ew[1]], [-R, -rd - 0.04]], halo=True)            # EW -> W
    # cue, at the apple's direction
    p0 = 3.35 * _u(cue_deg)
    _arrow(ax, p0, 1.2 * _u(cue_deg), C_CUE, lw=1.6, ms=8)
    ax.text(p0[0] + 0.05, p0[1] + 0.12, "cue", ha="center", va="bottom", fontsize=fs, fontweight="bold",
            color=C_CUE)
    ax.set_xlim(-4.15, 4.15)
    ax.set_ylim(-4.2, 4.1)


# ═════════════════════════════════════════════════════════════════════════════
# B: the board around the apple, with the steps
# ═════════════════════════════════════════════════════════════════════════════

def panel_route(ax, info, scheme="paper", fs=TEXT_FS, margin=1):
    """The cells from the agent to the apple plus `margin` cells around (3 x 3 steps -> 6 x 6 cells).
    Agent (grey square), apple (green), the apple's direction (dotted green), one arrow per step
    in the colour of its decision neuron."""
    sc = SCHEMES[scheme]
    (x0, y0), (x1, y1) = info["pos"], info["apple"]
    lo_x, hi_x = min(x0, x1) - margin, max(x0, x1) + margin
    lo_y, hi_y = min(y0, y1) - margin, max(y0, y1) + margin
    ax.set_aspect("equal")
    for x in np.arange(lo_x - 0.5, hi_x + 1):
        ax.plot([x, x], [lo_y - 0.5, hi_y + 0.5], color=C_GRID, lw=0.6, zorder=0)
    for y in np.arange(lo_y - 0.5, hi_y + 1):
        ax.plot([lo_x - 0.5, hi_x + 0.5], [y, y], color=C_GRID, lw=0.6, zorder=0)
    ax.plot([x0, x1], [y0, y1], color=C_CUE, lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.add_patch(FancyBboxPatch((x0 - 0.32, y0 - 0.32), 0.64, 0.64, boxstyle="round,pad=0,rounding_size=0.14",
                                fc="#555555", ec="none", zorder=3))
    ax.add_patch(Circle((x1, y1), 0.3, fc=C_CUE, ec="none", zorder=3))
    prev = tuple(info["pos"])
    for st in info["steps"]:
        ax.add_patch(FancyArrowPatch(prev, tuple(st["pos"]), arrowstyle="-|>", color=sc["route"][st["move"]],
                                     lw=1.5, mutation_scale=8, shrinkA=1, shrinkB=3, zorder=4))
        prev = tuple(st["pos"])
    ax.set_xlim(lo_x - 0.5, hi_x + 0.5)
    ax.set_ylim(lo_y - 0.5, hi_y + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#999999")
        sp.set_linewidth(0.8)


# ═════════════════════════════════════════════════════════════════════════════
# C: the trial
# ═════════════════════════════════════════════════════════════════════════════

def panels_trial(fig, rects, tr, scheme="paper", fs=TEXT_FS):
    """rects = (E raster, decision, counters) as figure-fraction [l, b, w, h]. Returns the three axes."""
    sc = SCHEMES[scheme]
    rec, info, meta = tr["rec"], tr["info"], tr["meta"]
    ang = np.asarray(tr["angles"])
    tm, T = rec["timing"], int(rec["T"])
    cue_on, cue_off = tm["cue_on"], tm["cue_off"]
    s = lambda t: np.asarray(t) / 1000.0                       # ms -> s
    ax_e, ax_d, ax_v = (fig.add_axes(r) for r in rects)
    for ax in (ax_e, ax_d, ax_v):
        ax.set_xlim(0, s(T))
        for sl in rec["slots"]:                                # one step per 200 ms slot
            ax.axvline(s(sl[1]), color="#cfcfcf", lw=0.5, zorder=0)
        for t_ in (cue_on, cue_off):
            ax.axvline(s(t_), color=C_CUE, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        S.style_ax(ax, fs + 1)                    # tick labels at fs

    # E neurons on their w0 map: forget window hatched (w0 = +1 and the negative pulse), cue: w0 = -1
    # everywhere, then w0 = -1 inside the RF (the memory), 0 outside
    # rows by compass bearing, clockwise from N at the top: N, E, S, W (as the decision rows)
    n = len(ang)
    w0 = np.zeros((T, n))
    w0[cue_on:cue_off] = meta["w0_mem"]
    w0[cue_off:] = np.asarray(rec["w0_map"][0])
    b_e = bearing(ang)                                         # bearing of each ring neuron
    order = np.argsort(b_e)                                    # rows of the w0 map: bearing 0, 22.5, ...
    im = ax_e.imshow(w0[:, order].T, aspect="auto", origin="lower", cmap=W0_CMAP, vmin=-1, vmax=0, alpha=0.55,
                     extent=(0, s(T), -11.25, 348.75), interpolation="nearest", zorder=0)
    ax_e.axvspan(0, s(cue_on), facecolor="white", edgecolor="none", zorder=0.5)
    ax_e.axvspan(0, s(cue_on), facecolor="none", hatch="////", edgecolor=C_FORGET, lw=0, alpha=0.55, zorder=1)
    tt, nn = np.nonzero(rec["spk_e"][0])
    ax_e.scatter(s(tt), b_e[nn], s=3.0, c=INK, linewidths=0, zorder=3, rasterized=True)
    ax_e.plot([0], [float(bearing(info["theta"]))], marker=">", ms=5, color=C_CUE, clip_on=False,
              transform=ax_e.get_yaxis_transform(), zorder=5)              # the apple's direction
    ax_e.set_ylim(348.75, -11.25)                              # N (bearing 0) at the top
    ax_e.set_yticks([0.0, 90.0, 180.0, 270.0])
    ax_e.set_yticklabels(ROW_ORDER, fontsize=fs)
    ax_e.set_ylabel("E neurons", fontsize=fs, labelpad=2)
    cax = ax_e.inset_axes([1.02, 0.0, 0.025, 1.0])                         # w0 bar, as in the ODR figure
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0])
    cb.ax.set_yticklabels(["−1", "0"])
    cb.ax.tick_params(labelsize=fs, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    cb.ax.annotate("$w_0$", xy=(1.0, 0.5), xycoords="axes fraction", xytext=(12, 0), textcoords="offset points",
                   ha="left", va="center", fontsize=fs)
    # event labels above the E raster (as in the ODR figure)
    xt = ax_e.get_xaxis_transform()
    ax_e.text(0.0, 1.03, "forget", transform=xt, ha="left", va="bottom", fontsize=fs, color=C_FORGET)
    ax_e.text(s(cue_on), 1.03, "cue", transform=xt, ha="left", va="bottom", fontsize=fs, color=C_CUE,
              fontweight="bold")

    # decision neurons (rows N, E, S, W from the top, as the E raster); grey = blocked by its counter in that slot
    spk_d = rec["spk_d"][0]
    for (t0, t1), sl in zip(rec["slots"], info["slot_log"]):
        for k in range(4):
            if sl["closed"][k]:
                y = ROW_Y[k]
                ax_d.fill_between([s(t0), s(t1)], y - 0.42, y + 0.42, color=C_BLOCK, lw=0, zorder=0)
    for k in range(4):
        y = ROW_Y[k]
        ax_d.vlines(s(np.flatnonzero(spk_d[:, k])), y - 0.32, y + 0.32, color=sc["dec"][k], lw=0.9, zorder=2)
    ax_d.set_ylim(-0.6, 3.6)
    ax_d.set_yticks([3, 2, 1, 0])
    ax_d.set_yticklabels(ROW_ORDER, fontsize=fs)
    dec_w0 = meta.get("dec_w0")
    dec_label = "decision" if dec_w0 is None else "decision,\n" + f"$w_0={dec_w0:g}$".replace("-", "−")
    ax_d.set_ylabel(dec_label, fontsize=fs, labelpad=2, linespacing=1.0)

    # distance counters: staircase from when they are set (end of the cue), then one dot per spike at threshold
    enc_t = int(rec["enc_t"])
    v, spk = rec["v_z"][0], rec["spk_z"][0]
    lab_y = []
    for j in (0, 1):
        col, ls = sc["counter"][j], sc["counter_ls"][j]
        st = np.flatnonzero(spk[:, j])
        t_end = st[0] if len(st) else T
        tt = np.arange(enc_t, t_end)
        ax_v.plot(s(tt), v[enc_t:t_end, j], color=col, lw=1.4, ls=ls, drawstyle="steps-post", zorder=3)
        if len(st):
            ax_v.plot([s(t_end), s(t_end)], [v[t_end - 1, j], 1.0], color=col, lw=1.4, ls=ls, zorder=3)
            ax_v.plot(s(st), np.ones(len(st)), "o", ms=2.6, color=col, mec="none", zorder=4)
        lab_y.append(v[T - 1, j] if not len(st) else 1.0)
    if abs(lab_y[0] - lab_y[1]) < 0.2:                         # names at the right end, kept apart
        m = (lab_y[0] + lab_y[1]) / 2
        lab_y = [m + 0.1, m - 0.1] if lab_y[0] >= lab_y[1] else [m - 0.1, m + 0.1]
    for j in (0, 1):
        ax_v.text(s(T) + 0.02, lab_y[j], COUNTER_NAMES[j], ha="left", va="center", fontsize=fs,
                  color=sc["counter"][j] if scheme != "type" else INK, fontweight="bold")
    ax_v.axhline(1.0, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax_v.set_ylim(0.15, 1.15)
    ax_v.set_yticks([1.0])
    ax_v.set_yticklabels(["θ"], fontsize=fs)
    ax_v.set_ylabel("distance\ncounter", fontsize=fs, labelpad=2, linespacing=1.0)
    for ax in (ax_e, ax_d):
        ax.tick_params(labelbottom=False)
    ticks = np.arange(0, s(T) + 1e-9, 0.5)
    ax_v.set_xticks(ticks)
    ax_v.set_xticklabels([f"{t:g}" for t in ticks])
    ax_v.set_xlabel("time (s)", fontsize=fs, labelpad=1)
    return ax_e, ax_d, ax_v


def make_figure(tr=None, scheme="paper", fs=TEXT_FS, width=S.PAGE_W, height=1.92,
                a_box=(0.0, -0.02, 1.68, 1.68), b_box=(1.86, 0.5, 0.95, 0.95),
                c_left=3.6, c_right=0.4, c_bottom=0.3, c_heights=(0.56, 0.42, 0.36), c_gap=0.06,
                results=True, df=None, how="mean", row2_h=1.36, row2_bottom=0.33, row2_axes_h=0.76,
                d_x=(0.48, 2.0), e_x=(2.42, 3.74), f_x=(4.12, 5.95), ylim=(0, 33), e_legend=False,
                f_legend_frame=True, scale=SCALE, letter_fs=S.FS):
    """A | B | C, full width; with results=True also D | E | F underneath (the A-C row moved up by row2_h).
    tr: a trial from get_trial (default: TRIAL, simulated once, then cached). df: the runs (default:
    load_results()). how: "mean" (mean over runs) or "pooled" (navigation's tables).
    Sizes in inches (real size with scale = 1): a_box / b_box = (x, y, w, h) inside the A-C row (`height`
    tall); C spans c_left .. width - c_right, its three panels (E raster, decision, counters) c_heights tall
    from c_bottom up. D, E, F: x ranges of their axes (f_x right None = C's right edge); D-F axes
    row2_axes_h tall from row2_bottom up; F's key in one column on its right, framed as the legends of
    Figures 2 and 3 (f_legend_frame). scale: the figure is made `scale` times as large (every position
    scales with it; text, lines and markers keep their size in points). fs: every text (TEXT_FS = 7 pt);
    letter_fs: the panel letters, as in the other figures (S.letter: letter_fs + 3, bold)."""
    tr = tr or get_trial()
    W = width
    y0 = row2_h if results else 0.0                   # the A-C row sits above the D-F row
    H = height + y0
    fig = plt.figure(figsize=(W * scale, H * scale))   # positions below are fractions of W x H
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    panel_scheme(box(a_box[0], a_box[1] + y0, *a_box[2:]), scheme, fs, cue_deg=tr["info"]["theta"])
    panel_route(box(b_box[0], b_box[1] + y0, *b_box[2:]), tr["info"], scheme, fs)
    c_w = W - c_left - c_right
    he, hd, hv = c_heights
    y_v = c_bottom + y0
    y_d = y_v + hv + c_gap
    y_e = y_d + hd + c_gap
    rects = [[c_left / W, y / H, c_w / W, h / H] for y, h in ((y_e, he), (y_d, hd), (y_v, hv))]
    panels_trial(fig, rects, tr, scheme, fs)
    y_letter = H - 0.2
    for x, letter in ((0.02, "A"), (b_box[0] - 0.02, "B"), (c_left - 0.5, "C")):
        S.letter(fig, x, y_letter, letter, W, H, letter_fs)
    fig.trial = tr
    if results:
        df = load_results() if df is None else df
        yb, hb = row2_bottom, row2_axes_h
        ax_d = box(d_x[0], yb, d_x[1] - d_x[0], hb)
        ax_e = box(e_x[0], yb, e_x[1] - e_x[0], hb)
        f_right = f_x[1] if f_x[1] is not None else W - c_right
        ax_f = box(f_x[0], yb, f_right - f_x[0], hb)
        panel_networks(ax_d, df, fs, how, ylim)
        ax_d.set_ylabel(D_YLABEL, fontsize=fs, labelpad=3, linespacing=1.0)   # the y axis of D, E and F
        panel_weight_noise(ax_e, df, fs, how, ylim, legend=e_legend)
        leg = panel_rf_forget(ax_f, df, fs, how, ylim, legend_loc="center left",
                              legend_anchor=(1 + 0.16 / (f_right - f_x[0]), 0.5), legend_ncol=1)
        if f_legend_frame and leg is not None:
            fig.legend_frame = S.legend_frame(fig, [leg])
        y_letter2 = yb + hb + 0.08
        for x, letter in ((0.02, "D"), (e_x[0] - 0.36, "E"), (f_x[0] - 0.36, "F")):
            S.letter(fig, x, y_letter2, letter, W, H, letter_fs)
        fig.results = df
    return fig
