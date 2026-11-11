"""
mnist_figures.py -- the MNIST and N-MNIST figures of the paper (same code, dataset="MNIST" or
"N-MNIST"). Shown in MNIST.ipynb and NMNIST.ipynb.

    make_figure_accuracy()   A accuracy vs delay, B vs weight-noise CV, C vs input-noise CV,
                             one shared legend underneath
    make_figure_rasters()    one raster per network (hidden + output, one test digit, 5 s), accuracy
                             at 0.1 / 1 / 2 / 5 s under each raster, spike rate in the last second on
                             its right; right: accuracy at 5 s against that spike rate
    make_figure_mnist_nmnist()  the paper's Figure 2: A MNIST accuracy vs delay, B MNIST rasters (0-1 s
                             and 4-5 s, cut in between) with their spike rates, C N-MNIST accuracy vs
                             delay; one framed legend under all three

Every number comes from the dataset's results folder (nothing is retrained):
    MNIST    MNIST/results/mnist_ms_experiments/
    N-MNIST  N_MNIST/results/nmnist_w0_experiments/
with, in each:
    post_hoc/waiting/twait_sweep.csv                   accuracy vs delay        (exp_waiting.py, 1280 digits)
    post_hoc/weight_noise_cv/weight_noise_cv_sweep.csv accuracy vs weight CV    (exp_noise_weights_cv.py)
    post_hoc/input_noise_cv/input_noise_cv_sweep.csv   accuracy vs input CV     (exp_noise_input_cv.py)
    post_hoc/delay_activity/delay_activity.csv         spikes per 100 ms bin    (exp_delay_activity.py, 512 digits)
    <run>/weights.pt                                   the rasters (one test digit run through each network)
Styles (colour, line, marker, label) come from style.NETWORKS, so they match every other figure.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

import style as S

P = S.P
CACHE = os.path.join(S.HERE, "cache")

# per dataset: code folder, results folder, x label of the input-noise panel, raster digit
DATASETS = {
    "MNIST": dict(folder="MNIST", results="mnist_ms_experiments",
                  input_noise_label="input-rate noise, CV (%)",
                  min_acc_5s=50),              # raster figure: only networks with >= 50 % at 5 s
    "N-MNIST": dict(folder="N_MNIST", results="nmnist_w0_experiments",
                    input_noise_label="input-event noise, CV (%)",
                    test_index=7294,           # a "7", as the MNIST raster digit
                    min_acc_5s=50),            # raster figure: only networks with >= 50 % at 5 s, as MNIST
}

N_HIDDEN, N_OUT = 100, 10           # both datasets
CHANCE = 10.0                       # % (10 classes)
T_STIM_MS = 100                     # input on for 100 ms (both datasets)
ACC_DELAYS_MS = [100, 1000, 2000, 5000]


def post_hoc_dir(dataset):
    c = DATASETS[dataset]
    return os.path.join(S.ROOT, c["folder"], "results", c["results"], "post_hoc")


# ═════════════════════════════════════════════════════════════════════════════
# data
# ═════════════════════════════════════════════════════════════════════════════

def load(dataset="MNIST"):
    """All result tables of one dataset (tidy DataFrames). A table that does not exist yet
    (e.g. the CV noise sweeps before run_noise_cv.sh has run) is None."""
    def rd(*p):
        path = os.path.join(post_hoc_dir(dataset), *p)
        if not os.path.exists(path):
            print(f"[{dataset}] not found yet: {os.path.relpath(path, S.ROOT)}")
            return None
        return pd.read_csv(path)
    return dict(dataset=dataset,
                delay=rd("waiting", "twait_sweep.csv"),
                weight_cv=rd("weight_noise_cv", "weight_noise_cv_sweep.csv"),
                input_cv=rd("input_noise_cv", "input_noise_cv_sweep.csv"),
                activity=rd("delay_activity", "delay_activity.csv"))


def spike_rate(activity, from_ms=4000, to_ms=5000):
    """Mean firing rate (Hz per neuron) of hidden + output neurons between from_ms and to_ms
    after the input onset (input off at 100 ms), averaged over the test digits of
    delay_activity.csv. Its bins are 100 ms long and start at bin_start_ms. Returns a Series."""
    a = activity[(activity.bin_start_ms >= from_ms) & (activity.bin_start_ms < to_ms)]
    spikes = a.groupby("run_name")[["hidden_spikes", "output_spikes"]].sum().sum(axis=1)
    return spikes / (N_HIDDEN + N_OUT) / ((to_ms - from_ms) / 1000)


def accuracy_at(delay, t_wait_ms):
    """Test accuracy (%) of every run at one delay, from twait_sweep.csv."""
    d = delay[delay.t_wait == t_wait_ms]
    return d.set_index("run_name").test_acc * 100


def runs_present(data, runs=None):
    """runs=None -> the paper selection (style.PAPER_RUNS[dataset]); runs="all" -> every network.
    Keeps only runs that have results."""
    if runs is None:
        runs = S.PAPER_RUNS[data["dataset"]]
    elif runs == "all":
        runs = S.NETWORK_ORDER
    have = set(data["delay"].run_name)
    return [r for r in runs if r in have]


def raster_columns(runs, n_cols=2):
    """Two columns: constant-w0 networks on the left, all others on the right (in `runs` order).
    Otherwise (or with another n_cols): filled column by column."""
    const = [r for r in runs if S.net(r)["group"] == "constant w0"]
    other = [r for r in runs if r not in const]
    if n_cols == 2 and const and other:
        return [const, other]
    n_rows = int(np.ceil(len(runs) / n_cols))
    return [runs[i * n_rows:(i + 1) * n_rows] for i in range(n_cols)]


# ═════════════════════════════════════════════════════════════════════════════
# shared drawing helpers
# ═════════════════════════════════════════════════════════════════════════════

def _line(ax, x, y, run, err=None, lw=1.2, ms=3.2):
    st = S.net(run)
    mfc = "none" if st["open"] else st["colour"]
    ax.plot(x, y, color=st["colour"], ls=st["ls"], lw=lw, marker=st["marker"], ms=ms, mfc=mfc,
            mew=0.8, label=st["label"], zorder=3)
    if err is not None:
        ax.fill_between(x, y - err, y + err, color=st["colour"], alpha=0.15, lw=0, zorder=2)


def _acc_axis(ax, fs, ylabel=True):
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(CHANCE, color="#999999", lw=0.8, ls=":", zorder=1)
    if ylabel:
        ax.set_ylabel("test accuracy (%)", fontsize=fs, labelpad=1)
    else:
        ax.set_yticklabels([])
    S.style_ax(ax, fs)


def w0_strip(fig, rect, fs, compact=False):
    """Discrete colour bar of the constant-w0 networks: blue = bistable (w0 < 0), red = IF-like
    (w0 > 0), with a gap at w0 = 0 where the two regimes meet. compact: "constant w0" on the left of
    the bar and the regime names above it (instead of title above, regime names below)."""
    from matplotlib.patches import Rectangle
    vals = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    xs = [0, 1, 2, 3.4, 4.4, 5.4]                       # 0.4 gap between -0.5 and +0.5
    ax = fig.add_axes(rect)
    for x, v in zip(xs, vals):
        ax.add_patch(Rectangle((x, 0), 0.94, 1, color=S.W0_COLOURS[v], lw=0))
    ax.set_xlim(-0.05, xs[-1] + 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([x + 0.47 for x in xs])
    ax.set_xticklabels([f"{v:+g}".replace("-", "\u2212") for v in vals], fontsize=fs - 2)
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    y_reg, va_reg = (1.35, "bottom") if compact else (-1.6, "top")
    if compact:
        ax.text(-0.35, 0.5, r"constant $w_0$", ha="right", va="center", fontsize=fs - 1, transform=ax.transData)
    else:
        ax.set_title(r"constant $w_0$", fontsize=fs - 1, pad=2)
    ax.text(1.47, y_reg, "bistable", ha="center", va=va_reg, fontsize=fs - 2, color=S.C_BISTABLE,
            fontweight="bold", transform=ax.transData)
    ax.text(4.87, y_reg, "IF-like", ha="center", va=va_reg, fontsize=fs - 2, color=S.C_IFLIKE,
            fontweight="bold", transform=ax.transData)
    return ax


def network_legend(fig, runs, anchor, fs, ncol=3):
    """Legend of every non-constant-w0 network in `runs` (the constant ones are in w0_strip)."""
    hs = []
    for r in runs:
        st = S.net(r)
        if st["group"] == "constant w0":
            continue
        mfc = "none" if st["open"] else st["colour"]
        hs.append(Line2D([], [], color=st["colour"], ls=st["ls"], lw=1.2, marker=st["marker"], ms=3.5,
                         mfc=mfc, mew=0.8, label=st["label"]))
    return fig.legend(handles=hs, loc="upper left", bbox_to_anchor=anchor, ncol=ncol, frameon=False,
                      fontsize=fs - 2, handlelength=2.2, columnspacing=1.0, handletextpad=0.4,
                      labelspacing=0.35, borderpad=0, borderaxespad=0)


# ═════════════════════════════════════════════════════════════════════════════
# Figure: accuracy vs delay / weight noise / input noise
# ═════════════════════════════════════════════════════════════════════════════

def panel_delay(ax, delay, runs, fs=S.FS, ylabel=True):
    """Accuracy against the silent delay before the read-out (log axis)."""
    for r in runs:
        d = delay[delay.run_name == r].sort_values("t_wait")
        _line(ax, d.t_wait.values / 1000, d.test_acc.values * 100, r)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xlim(0.085, 5.9)
    ax.set_xlabel("delay (s)", fontsize=fs, labelpad=1)
    _acc_axis(ax, fs, ylabel)
    ax.text(5.9, CHANCE + 1.5, "chance", ha="right", va="bottom", fontsize=fs - 2, color="#777777")


def panel_noise(ax, df, runs, xlabel, fs=S.FS, ylabel=False):
    """Accuracy (mean, shaded: +-1 s.d. over the noise draws) against the noise CV (%).
    df None (sweep not run yet): empty panel with a note."""
    if df is None:
        ax.text(0.5, 0.5, "not run yet\n(run_noise_cv.sh)", transform=ax.transAxes, ha="center",
                va="center", fontsize=fs - 1, color="#999999")
        ax.set_xticks([])
        ax.set_xlabel(xlabel, fontsize=fs, labelpad=1)
        _acc_axis(ax, fs, ylabel)
        return
    for r in runs:
        g = df[df.run_name == r].groupby("cv").test_acc
        m, s = g.mean() * 100, g.std(ddof=1).fillna(0) * 100
        _line(ax, m.index.values * 100, m.values, r, err=s.values)
    cvs = np.sort(df.cv.unique()) * 100
    ax.set_xticks(cvs)
    ax.set_xticklabels([f"{c:g}" for c in cvs])
    pad = 0.06 * (cvs[-1] - cvs[0])
    ax.set_xlim(cvs[0] - pad, cvs[-1] + pad)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=1)
    _acc_axis(ax, fs, ylabel)


def make_figure_accuracy(runs=None, data=None, dataset="MNIST", fs=S.FS, width=S.PAGE_W, ax_w=1.85, ax_h=1.55,
                         left=0.5, gap=0.42, bottom=1.25, top=0.32, share_y_labels=False):
    """A: accuracy vs delay. B: vs weight-noise CV. C: vs input-noise CV. Same networks, same styles,
    same 0-100 % axis; one legend underneath (w0 colour bar + the other networks).
    Noise panels: read out at each network's trained delay (100 ms). A noise sweep that has not been
    run yet is shown as an empty panel saying so."""
    data = data or load(dataset)
    runs = runs_present(data, runs)
    H = bottom + ax_h + top
    W = width
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([(left + i * (ax_w + gap)) / W, bottom / H, ax_w / W, ax_h / H]) for i in range(3)]
    panel_delay(axes[0], data["delay"], runs, fs)
    panel_noise(axes[1], data["weight_cv"], runs, "weight noise, CV (%)", fs, ylabel=share_y_labels)
    panel_noise(axes[2], data["input_cv"], runs, DATASETS[data["dataset"]]["input_noise_label"], fs,
                ylabel=share_y_labels)
    if not share_y_labels:                                  # the y tick labels are shared: keep them on B, C too
        for ax in axes[1:]:
            ax.set_yticklabels([0, 25, 50, 75, 100])
    for ax, letter in zip(axes, "ABC"):
        x0 = ax.get_position().x0 * W
        S.letter(fig, x0 - 0.42, bottom + ax_h + 0.08, letter, W, H, fs)

    # one legend for the three panels: w0 colour bar on the left, the other networks to its right
    if any(S.net(r)["group"] == "constant w0" for r in runs):
        w0_strip(fig, [left / W, (bottom - 0.88) / H, 1.35 / W, 0.11 / H], fs)
        leg_x = left + 1.65
    else:
        leg_x = left
    n_other = sum(S.net(r)["group"] != "constant w0" for r in runs)
    network_legend(fig, runs, (leg_x / W, (bottom - 0.47) / H), fs, ncol=2 if n_other <= 4 else 3)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure: rasters + spike rate + accuracy vs spike rate
# ═════════════════════════════════════════════════════════════════════════════

def _drop_clashing_modules(folder):
    """MNIST and N_MNIST have modules with the same names (neuron_model,
    network_architecture, save_experiment, ...). Drop the already-imported ones that do not come
    from `folder`, so that folder's own files are imported."""
    folder = os.path.abspath(folder)
    for f in os.listdir(folder):
        name = f[:-3]
        mod = sys.modules.get(name) if f.endswith(".py") else None
        src = getattr(mod, "__file__", None) if mod is not None else None
        if src and os.path.dirname(os.path.abspath(src)) != folder:
            del sys.modules[name]


def _simulate_nmnist(runs, test_index, t_wait):
    """One N-MNIST test digit (cached frames, index `test_index`) through each network,
    100 ms input + t_wait ms of silence. Returns {run: dict(hidden, output, label, t_stim)}."""
    import torch
    folder = os.path.join(S.ROOT, "N_MNIST")
    _drop_clashing_modules(folder)
    sys.path.insert(0, folder)
    cwd = os.getcwd()
    os.chdir(folder)                                     # the data / cache / results paths are relative
    try:
        import nmnist_w0_lib as lib
        from rebuild_network import load_trained_model
        _, testloader = lib.get_dataloaders(batch_size=1)
        x, y = testloader.dataset[test_index]
        x = lib.flatten_batch(torch.as_tensor(np.asarray(x))[None], "cpu", torch.float)
        out = {}
        for run in runs:
            model, _, _ = load_trained_model(lib.DEFAULT_OUT_DIR, run, device="cpu")
            model.T_wait = t_wait
            with torch.no_grad():
                _, recs = model.forward(x)
            out[run] = dict(hidden=recs["spk"][0].numpy(), output=recs["out_spk"][0].numpy(),
                            label=int(y), t_stim=lib.NUM_STEPS)
        return out
    finally:
        os.chdir(cwd)
        sys.path.remove(folder)


def weights_md5(dataset, run):
    """Fingerprint of a run's weights.pt: a cached raster is only reused while the weights are
    unchanged (a retrained run is simulated again)."""
    import hashlib
    c = DATASETS[dataset]
    path = os.path.join(S.ROOT, c["folder"], "results", c["results"], run, "weights.pt")
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_traces(runs, dataset="MNIST", t_wait=4900, sample_index=0, seed=0, refresh=False):
    """Spikes of one test digit through each network (100 ms input + t_wait ms of silence = 5 s).
    Simulated the first time (needs torch), then read from NICE_figures/cache/, so the notebook
    does not re-run the networks. A run whose weights.pt changed since (retrained) is simulated again.
      MNIST:   shared_plots.example_traces -- test digit 0 of the seeded test loader (a "7").
      N-MNIST: the cached test frames of DATASETS["N-MNIST"]["test_index"] (a "7")."""
    os.makedirs(CACHE, exist_ok=True)
    if dataset == "MNIST":
        path = os.path.join(CACHE, f"mnist_traces_twait{t_wait}_digit{sample_index}_seed{seed}.npz")
    else:
        path = os.path.join(CACHE, f"nmnist_traces_twait{t_wait}_test{DATASETS[dataset]['test_index']}.npz")
    traces = {}
    if os.path.exists(path) and not refresh:
        z = np.load(path)
        for r in runs:
            if f"{r}|hidden" in z and f"{r}|md5" in z and str(z[f"{r}|md5"]) == weights_md5(dataset, r):
                traces[r] = dict(hidden=z[f"{r}|hidden"], output=z[f"{r}|output"],
                                 label=int(z["label"]), t_stim=int(z["t_stim"]))
    missing = [r for r in runs if r not in traces]
    if missing:
        if dataset == "MNIST":
            _drop_clashing_modules(os.path.join(S.ROOT, "MNIST"))
            new = P.example_traces(missing, sample_index=sample_index, t_wait=t_wait, seed=seed, dataset="MNIST")
        else:
            new = _simulate_nmnist(missing, DATASETS[dataset]["test_index"], t_wait)
        traces.update(new)
        old = dict(np.load(path)) if os.path.exists(path) and not refresh else {}
        for r, tr in new.items():
            old[f"{r}|hidden"] = tr["hidden"].astype(bool)
            old[f"{r}|output"] = tr["output"].astype(bool)
            old[f"{r}|md5"] = np.array(weights_md5(dataset, r))
        tr0 = next(iter(traces.values()))
        old["label"], old["t_stim"] = tr0["label"], tr0["t_stim"]
        np.savez_compressed(path, **old)
    return traces


RASTER_SEGMENTS = ((0, 1000), (4000, 5000))   # Figure 2: show 0-1 s and 4-5 s, cut in between
SEG_GAP = 0.3                                   # width of the cut, in seconds of the time axis


def _time_axis_map(T, segments=None, seg_gap=SEG_GAP):
    """Time (ms) -> x of the raster (s). segments=None: the whole trial, x = t. Otherwise only the
    times inside the segments ((t0, t1) ms each) are drawn, one after the other, seg_gap apart.
    Returns (to_x, keep, xlim, segs_x): to_x(t_ms) -> x, keep(t_ms) -> bool mask,
    xlim, and the x range of each segment."""
    if segments is None:
        segments = ((0, T),)
    segs_x, x0 = [], 0.0
    for t0, t1 in segments:
        segs_x.append((x0, x0 + (t1 - t0) / 1000))
        x0 += (t1 - t0) / 1000 + seg_gap

    def to_x(t):
        t = np.asarray(t, dtype=float)
        x = np.full(t.shape, np.nan)
        for (t0, t1), (xa, _) in zip(segments, segs_x):
            m = (t >= t0) & (t <= t1)
            x[m] = xa + (t[m] - t0) / 1000
        return x

    keep = lambda t: ~np.isnan(to_x(t))
    return to_x, keep, (segs_x[0][0], segs_x[-1][1]), segs_x


def _ax_inches(ax):
    """Width and height of an axes, in inches."""
    pos, (fw, fh) = ax.get_position(), ax.figure.get_size_inches()
    return pos.width * fw, pos.height * fh


def _layer_brace(ax, y0, y1, label, fs, x_in=-0.03, w_in=0.04, lw=0.8, colour="#333333"):
    """Curly brace "{" left of the raster, from y0 to y1 (data), its right end x_in inches from
    the axes' left edge and w_in inches wide; the layer name on its left (fs - 3)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import blended_transform_factory
    aw, _ = _ax_inches(ax)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    xr, xm, xl = x_in / aw, (x_in - w_in / 2) / aw, (x_in - w_in) / aw
    ym, q = (y0 + y1) / 2, (y1 - y0) * 0.2
    verts = [(xr, y0), (xm, y0), (xm, y0 + q), (xm, ym - q), (xm, ym), (xl, ym),
             (xm, ym), (xm, ym + q), (xm, y1 - q), (xm, y1), (xr, y1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), transform=tr, fc="none", ec=colour, lw=lw,
                           capstyle="round", joinstyle="round", clip_on=False))
    ax.text(xl - 0.025 / aw, ym, label, transform=tr, ha="right", va="center", fontsize=fs - 3,
            color=colour)


def _cut_marks(ax, segs_x, xlim, size_in=0.035, lw=0.6):
    """Bottom spine cut between the segments: "/" at the end of one segment, a dotted line across
    the cut, "/" at the start of the next (a /.../ on the time axis)."""
    aw, ah = _ax_inches(ax)
    fx = lambda x: (x - xlim[0]) / (xlim[1] - xlim[0])
    dx, dy = 0.4 * size_in / aw, size_in / ah
    lw_spine = ax.spines["bottom"].get_linewidth()
    for (_, x_end), (x_start, _) in zip(segs_x[:-1], segs_x[1:]):
        a, b = fx(x_end), fx(x_start)
        ax.plot([a, b], [0, 0], transform=ax.transAxes, color="#555555", lw=lw_spine, ls=(0, (0.8, 1.6)),
                clip_on=False, solid_capstyle="butt")
        for x in (a, b):
            ax.plot([x - dx / 2, x + dx / 2], [-dy / 2, dy / 2], transform=ax.transAxes, color="black",
                    lw=lw, clip_on=False)
    # the solid spine only under the segments
    ax.spines["bottom"].set_visible(False)
    for xa, xb in segs_x:
        ax.plot([fx(xa), fx(xb)], [0, 0], transform=ax.transAxes, color="black", lw=lw_spine,
                clip_on=False, solid_capstyle="butt")


def panel_raster(ax, tr, run, fs=S.FS, gap=3, hid_scale=0.16, out_scale=1.4, segments=None, seg_gap=SEG_GAP,
                 braces=True):
    """Hidden layer on top (grey), output layer below: the neuron of the true digit in the network
    colour, the other outputs light grey. Grey band = input on. Same drawing as the Cosyne Figure 2.
    segments: None = the whole trial; ((t0, t1), ...) ms = only those stretches, cut in between
    (Figure 2: 0-1 s and 4-5 s). hid_scale / out_scale: height of one hidden / output neuron;
    gap: space between the layers (same units). braces: a "{" and the layer name left of each layer.
    Sets ax.time_ticks = (positions, labels) for a time axis under the raster."""
    st = S.net(run)
    T = tr["hidden"].shape[0]
    to_x, keep, xlim, segs_x = _time_axis_map(T, segments, seg_gap)
    th, nh = np.nonzero(tr["hidden"])
    to, no = np.nonzero(tr["output"])
    n_h = tr["hidden"].shape[1]
    n_o = tr["output"].shape[1]
    y_h0 = n_o * out_scale + gap                      # first hidden neuron
    ax.axvspan(*to_x([0, tr["t_stim"]]), color="#e8e8e8", lw=0, zorder=0)
    k = keep(th)
    ax.scatter(to_x(th[k]), nh[k] * hid_scale + y_h0, s=0.15, c="#555555", linewidths=0, rasterized=True)
    k = keep(to)
    to, no = to[k], no[k]
    right = no == tr["label"]
    ax.scatter(to_x(to[~right]), no[~right] * out_scale, s=1.5, c="#bdbdbd", linewidths=0, rasterized=True)
    ax.scatter(to_x(to[right]), no[right] * out_scale, s=1.5, c=st["colour"], linewidths=0, rasterized=True)
    for xa, xb in segs_x:                             # the true digit's row, in each segment
        ax.plot([xa, xb], [tr["label"] * out_scale] * 2, color=st["colour"], lw=0.4, alpha=0.4, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5 * out_scale - 0.5, y_h0 + n_h * hid_scale + 0.5)
    ax.set_yticks([])
    S.style_ax(ax, fs)
    ax.spines["left"].set_visible(False)
    if braces:
        _layer_brace(ax, y_h0 - 0.3 * hid_scale, y_h0 + (n_h - 0.7) * hid_scale, "hidden", fs)
        _layer_brace(ax, -0.4 * out_scale, (n_o - 0.6) * out_scale, "output", fs)
    # time ticks: the ends of each segment (and every second inside a long one)
    pos, lab = [], []
    for (t0, t1), (xa, _) in zip(segments or ((0, T),), segs_x):
        for t in range(int(np.ceil(t0 / 1000)), int(t1 / 1000) + 1):
            pos.append(xa + t - t0 / 1000)
            lab.append(f"{t}")
    ax.time_ticks = (pos, lab)
    ax.time_map = to_x
    if segments is not None and len(segs_x) > 1:
        _cut_marks(ax, segs_x, xlim)
    return ax


def _title_with_marker(ax, run, fs, y=1.2, label=None):
    """Network name above the raster (y in axes fraction), its marker (as in the scatter plot) in front.
    label: text instead of the network's label in style.NETWORKS."""
    st = S.net(run)
    mfc = "none" if st["open"] else st["colour"]
    ax.plot([0.012], [y], marker=st["marker"], color=st["colour"], mfc=mfc, mew=0.8, ms=3.5,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.04, y, label or st["label"], transform=ax.transAxes, ha="left", va="center", fontsize=fs - 2,
            color=st["colour"])


def raster_runs(data, runs=None, min_acc_5s="default"):
    """The networks of a raster stack: runs_present, minus every network whose accuracy at 5 s is
    below min_acc_5s (%). "default" = the dataset's own value in DATASETS, None = keep every network."""
    runs = runs_present(data, runs)
    acc5 = accuracy_at(data["delay"], 5000)
    if min_acc_5s == "default":
        min_acc_5s = DATASETS[data["dataset"]].get("min_acc_5s")
    if min_acc_5s is None:
        return runs
    dropped = [r for r in runs if acc5[r] < min_acc_5s]
    if dropped:
        print(f"left out (accuracy at 5 s < {min_acc_5s:g} %): " +
              ", ".join(f"{r} ({acc5[r]:.0f} %)" for r in dropped))
    return [r for r in runs if acc5[r] >= min_acc_5s]


def raster_row(fig, W, H, x, y, run, tr, tw, rate, fs=S.FS, raster_w=1.55, raster_h=0.2,
               acc_delays_ms=ACC_DELAYS_MS, header=None, time_axis=False, name_dy=0.09, acc_dy=0.045,
               label=None, header_dy=0.16, show_acc=True, time_dy=0.19, segments=None):
    """One network: raster (lower-left corner at x, y inches, raster_w x raster_h), its name and marker
    above, the test accuracy at each delay of acc_delays_ms under the time axis (tw = twait_sweep.csv),
    and its spike rate (rate[run], Hz) on the right. header: text above the rate column (first row),
    its bottom header_dy inches above the raster; time_axis: draw the time axis in seconds under this
    raster (last row). name_dy / acc_dy: inches from the raster to the centre of its name / to the top of
    the accuracies. label: name instead of the style.NETWORKS label. show_acc=False: no accuracies (and
    no delay ticks). time_dy: inches from the raster to its time axis; 0 = the raster's own x axis.
    segments: shown stretches of the trial (ms), see panel_raster; None = the whole trial.
    Returns the axes."""
    from matplotlib.transforms import blended_transform_factory
    up = lambda inch: 1 + inch / raster_h            # inches above the raster's top -> axes fraction
    down = lambda inch: -inch / raster_h             # inches below its bottom      -> axes fraction
    ax = fig.add_axes([x / W, y / H, raster_w / W, raster_h / H])
    panel_raster(ax, tr, run, fs, segments=segments)
    _title_with_marker(ax, run, fs, y=up(name_dy), label=label)
    col = S.net(run)["colour"]
    # accuracies under the time axis, at their delay (a delay inside a cut is left out)
    T = tr["hidden"].shape[0]
    xs = []
    shown = [d for d in (acc_delays_ms if show_acc else [])
             if not np.isnan(ax.time_map([min(tr["t_stim"] + d, T)])[0])]
    for i, dly in enumerate(shown):
        a = tw[(tw.run_name == run) & (tw.t_wait == dly)].test_acc.mean() * 100
        xt = ax.time_map([min(tr["t_stim"] + dly, T)])[0]
        xs.append(xt)
        ax.text(xt, down(acc_dy), f"{a:.0f}%", transform=blended_transform_factory(ax.transData, ax.transAxes),
                va="top", ha="right" if i == len(shown) - 1 else "center", fontsize=fs - 2,
                fontweight="bold", color=col, clip_on=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=2.0, color=col)
    # spike rate on the right
    ax.text(1 + 0.06 / raster_w, 0.5, f"{rate[run]:.2g}" if rate[run] >= 0.01 else "<0.01",
            transform=ax.transAxes, ha="left", va="center", fontsize=fs - 1, fontweight="bold", color=col)
    if header:                                       # column header for the rates
        ax.text(1 + 0.06 / raster_w, up(header_dy), header, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=fs - 2, color="#333333", linespacing=1.0)
    pos, lab = ax.time_ticks
    if time_axis and time_dy <= 0:                   # time axis = the raster's own x axis
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
        ax.tick_params(axis="x", labelsize=fs - 1, length=2.0, pad=1, color="black")
        ax.set_xlabel("time (s)", fontsize=fs, labelpad=1)
    elif time_axis:                                  # time axis under the last raster of a column
        sec = ax.secondary_xaxis(down(time_dy))
        sec.set_xticks(pos)
        sec.set_xticklabels(lab)
        sec.tick_params(labelsize=fs - 1, length=2.0, pad=1)
        sec.spines["bottom"].set_linewidth(0.6)
        sec.set_xlabel("time (s)", fontsize=fs, labelpad=1)
    return ax


def make_figure_rasters(runs=None, data=None, traces=None, dataset="MNIST", fs=S.FS, width=None, n_cols=1,
                        raster_w=1.55, raster_h=0.2, row_pitch=0.53, col_label_w=0.45, rate_w=0.42,
                        col_gap=0.12, scatter_w=1.75, scatter_gap=1.1, bottom=0.55, top=0.48,
                        acc_delays_ms=ACC_DELAYS_MS, rate_window_ms=(4000, 5000), min_acc_5s="default"):
    """Left: one raster per network in `runs`, one under the other (n_cols=2: constant w0 left, the
    others right; see raster_columns), with the test accuracy at each delay of `acc_delays_ms` under it (twait_sweep.csv, 1280
    digits) and the mean spike rate of hidden + output neurons in `rate_window_ms` on its right
    (delay_activity.csv, 512 digits). Right: accuracy at 5 s against that spike rate, one symbol per
    network. min_acc_5s (%): leave out, in both panels, every network whose accuracy at 5 s is below it;
    "default" = the dataset's own value in DATASETS (MNIST: 50), None = keep every network.
    Sizes in inches; width=None = as wide as the content. Returns (fig, traces)."""
    data = data or load(dataset)
    runs = raster_runs(data, runs, min_acc_5s)
    rate = spike_rate(data["activity"], *rate_window_ms)
    acc5 = accuracy_at(data["delay"], 5000)
    traces = traces or get_traces(runs, dataset=data["dataset"])
    tw = data["delay"]

    columns = raster_columns(runs, n_cols)
    n_cols = len(columns)
    n_rows = max(len(c) for c in columns)
    col_w = col_label_w + raster_w + rate_w
    span = (n_rows - 1) * row_pitch + raster_h            # height of the raster stack
    sc_h = min(2.4, max(span, 1.7))                       # scatter: as tall as the stack, 1.7-2.4 in
    block = max(span, sc_h)
    H = bottom + block + top
    x_sc = n_cols * col_w + (n_cols - 1) * col_gap + scatter_gap       # left edge of the scatter plot
    W = width or (x_sc + scatter_w + 0.1)
    fig = plt.figure(figsize=(W, H))
    win = f"{rate_window_ms[0] / 1000:g}–{rate_window_ms[1] / 1000:g} s"

    for c, r, run in [(c, r, run) for c, col_runs in enumerate(columns) for r, run in enumerate(col_runs)]:
        x = c * (col_w + col_gap) + col_label_w
        y = bottom + (block - span) + (n_rows - 1 - r) * row_pitch     # stack top-aligned
        raster_row(fig, W, H, x, y, run, traces[run], tw, rate, fs, raster_w, raster_h, acc_delays_ms,
                   header=f"rate {win}\n(Hz)" if r == 0 else None, time_axis=r == len(columns[c]) - 1)

    # ---- right: accuracy at 5 s against spike rate -----------------------------------------
    sc_y = bottom + (block - sc_h) / 2
    ax = fig.add_axes([x_sc / W, sc_y / H, scatter_w / W, sc_h / H])
    for run in runs:
        st = S.net(run)
        mfc = "none" if st["open"] else st["colour"]
        ax.plot(rate[run], acc5[run], ls="none", marker=st["marker"], color=st["colour"], mfc=mfc,
                mew=1.0, ms=5.5 if st["marker"] in "x+" else 4.5,
                zorder=4 if st["group"] == "LIF" else 3)          # LIF crosses on top of the 0 Hz dots
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.4)
    ax.set_xlim(-0.02, 200)
    ax.set_xticks([0, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel(f"spike rate, {win} (Hz)", fontsize=fs, labelpad=1)
    ax.set_ylabel("test accuracy at 5 s (%)", fontsize=fs, labelpad=1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(CHANCE, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.text(190, CHANCE + 1.5, "chance", ha="right", va="bottom", fontsize=fs - 2, color="#777777")
    S.style_ax(ax, fs)

    y_letter = bottom + block + 0.2                       # both letters on one line, above the first title
    S.letter(fig, 0.02, y_letter, "A", W, H, fs)
    S.letter(fig, x_sc - 0.55, y_letter, "B", W, H, fs)
    fig.rate, fig.acc5 = rate, acc5          # the plotted numbers, for the notebook table
    return fig, traces


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 of the paper: MNIST delay | MNIST rasters | N-MNIST delay
# ═════════════════════════════════════════════════════════════════════════════

def legend_frame(fig, artists, **kw):
    """Thin rounded frame around a group of legend artists: style.legend_frame (same in every figure)."""
    return S.legend_frame(fig, artists, **kw)


def make_figure_mnist_nmnist(runs=None, data=None, data_nmnist=None, traces=None, fs=S.FS, width=S.PAGE_W,
                             left=0.5, ax_w=1.38, gap_ab=0.42, gap_bc=0.78, col_label_w=0.4, rate_w=0.55,
                             raster_h=None, row_pitch=None, bottom=0.88, top=0.27, legend_dy=0.45, show_acc=False,
                             acc_delays_ms=ACC_DELAYS_MS, rate_window_ms=(4000, 5000), min_acc_5s="default",
                             raster_labels=None, segments=RASTER_SEGMENTS, frame=True):
    """A: MNIST test accuracy against the delay. B: MNIST rasters, one test digit per network (as
    make_figure_rasters, without its accuracy-vs-rate plot): spike rate in rate_window_ms on the right of
    each raster; show_acc=True also writes the accuracy at each delay of acc_delays_ms under each raster
    (the same numbers as A; makes the figure taller); networks below min_acc_5s at 5 s are left out ("default":
    DATASETS["MNIST"], 50 %). C: N-MNIST test accuracy against the delay. A and C are the same plot for
    the two datasets: same networks, styles and axes (solid = constant w0, dotted = every other network),
    so one legend under the three panels serves both; frame=True draws a thin frame around it.
    segments: the stretches of the trial shown in B (ms; default 0-1 s and 4-5 s, cut in between);
    None = the whole 5 s.
    A and C are as tall as the raster stack of B (top of the first name to the time axis); the rasters
    take the width left between A and C. rate_w: the rate column (numbers and header); gap_bc: from the
    rate column to C's axes (C's tick labels and y label sit in it). raster_h / row_pitch None: 0.13 / 0.26 in
    without the accuracies, 0.18 / 0.46 in with them. raster_labels: {run: shorter name above its raster}
    (default: the heterogeneous network is "heterogeneous"; its full name is in the legend).
    Sizes in inches. Returns (fig, traces)."""
    if raster_labels is None:
        raster_labels = {"lif_hidden_bistable_output_w0_-1.00": "heterogeneous"}
    data = data or load("MNIST")
    data_nmnist = data_nmnist or load("N-MNIST")
    runs_a = runs_present(data, runs)
    runs_c = runs_present(data_nmnist, runs)
    runs_b = raster_runs(data, runs, min_acc_5s)
    rate = spike_rate(data["activity"], *rate_window_ms)
    traces = traces or get_traces(runs_b, dataset="MNIST")
    win = f"{rate_window_ms[0] / 1000:g}–{rate_window_ms[1] / 1000:g} s"

    # horizontal: A | B (labels, rasters, rates) | C, filling `width`
    W = width
    x_b = left + ax_w + gap_ab                        # left edge of B (hidden / output labels)
    x_c = W - 0.08 - ax_w                             # left edge of C's axes
    raster_w = x_c - gap_bc - rate_w - col_label_w - x_b
    # vertical: the raster stack sets the height of A and C
    raster_h = raster_h or (0.18 if show_acc else 0.13)
    row_pitch = row_pitch or (0.46 if show_acc else 0.26)
    name_h = 0.13 if show_acc else 0.12               # name above the first raster
    time_h = 0.19 if show_acc else 0.0                # its time axis under the last (0 = its own x axis)
    ax_h = name_h + (len(runs_b) - 1) * row_pitch + raster_h + time_h
    H = bottom + ax_h + top
    fig = plt.figure(figsize=(W, H))

    # A and C: accuracy against delay
    ax_a = fig.add_axes([left / W, bottom / H, ax_w / W, ax_h / H])
    ax_c = fig.add_axes([x_c / W, bottom / H, ax_w / W, ax_h / H])
    panel_delay(ax_a, data["delay"], runs_a, fs)
    panel_delay(ax_c, data_nmnist["delay"], runs_c, fs)

    # B: rasters, top of the first name level with the top of A, time axis level with A's x axis
    tw = data["delay"]
    for r, run in enumerate(runs_b):
        y = bottom + time_h + (len(runs_b) - 1 - r) * row_pitch
        raster_row(fig, W, H, x_b + col_label_w, y, run, traces[run], tw, rate, fs, raster_w, raster_h,
                   acc_delays_ms, header=f"rate {win}\n(Hz)" if r == 0 else None, header_dy=0.03,
                   time_axis=r == len(runs_b) - 1, name_dy=0.08 if show_acc else 0.06, acc_dy=0.035,
                   label=raster_labels.get(run), show_acc=show_acc, time_dy=time_h, segments=segments)

    # letters, each with its dataset
    y_letter = bottom + ax_h + 0.12
    for x, letter, name in ((0.02, "A", "MNIST"), (x_b - 0.12, "B", "MNIST"), (x_c - 0.5, "C", "N-MNIST")):
        S.letter(fig, x, y_letter, letter, W, H, fs)
        fig.text((x + 0.2) / W, y_letter / H, name, fontsize=fs + 1, va="bottom")

    # one legend for A and C (and the colours of B), centred under the three panels:
    # w0 colour bar on the left, the other networks to its right, a thin frame around both
    runs_all = [r for r in S.NETWORK_ORDER if r in set(runs_a) | set(runs_c)]
    y_leg = bottom - legend_dy                         # top of the legend
    leg = network_legend(fig, runs_all, (0, y_leg / H), fs, ncol=2)
    leg_w = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    strip_name_w, strip_w, strip_gap = 0.72, 1.35, 0.35   # "constant w0" on the left of the colour bar
    has_strip = any(S.net(r)["group"] == "constant w0" for r in runs_all)
    group_w = (strip_name_w + strip_w + strip_gap if has_strip else 0) + leg_w
    x0 = (W - group_w) / 2
    in_frame = [leg]
    if has_strip:
        x0 += strip_name_w
        in_frame.append(w0_strip(fig, [x0 / W, (y_leg - 0.24) / H, strip_w / W, 0.11 / H], fs, compact=True))
        x0 += strip_w + strip_gap
    leg.set_bbox_to_anchor((x0 / W, y_leg / H))
    if frame:
        fig.legend_frame = legend_frame(fig, in_frame)
    fig.rate = rate                                   # the plotted rates, for the notebook table
    return fig, traces



# ═════════════════════════════════════════════════════════════════════════════
# Figure: w0 before and after learning (two-tailed network)
# ═════════════════════════════════════════════════════════════════════════════

W0_KEYS = ("w0_hidden_initial", "w0_hidden_final", "w0_output_initial", "w0_output_final")


def two_tailed_run(dataset="MNIST"):
    """The learnable two-tailed network of the paper selection."""
    return next(r for r in S.PAPER_RUNS[dataset] if r.startswith("dist_two_tailed"))


def load_w0(run, dataset="MNIST"):
    """Initial and learnt w0 of one run (hidden and output layer), as saved in its weights.pt.
    Cached in NICE_figures/cache/ with the md5 of weights.pt, so torch is only needed the first time."""
    os.makedirs(CACHE, exist_ok=True)
    tag = "mnist" if dataset == "MNIST" else "nmnist"
    path = os.path.join(CACHE, f"{tag}_w0_{run}.npz")
    md5 = weights_md5(dataset, run)
    if os.path.exists(path):
        z = np.load(path)
        if str(z["md5"]) == md5:
            return {k: z[k] for k in W0_KEYS}
    import torch
    c = DATASETS[dataset]
    w = torch.load(os.path.join(S.ROOT, c["folder"], "results", c["results"], run, "weights.pt"), map_location="cpu")
    missing = [k for k in W0_KEYS if k not in w]
    if missing:
        raise KeyError(f"{run}: weights.pt has no {missing} (only runs with a per-neuron w0 have them)")
    out = {k: np.asarray(w[k], dtype=float).ravel() for k in W0_KEYS}
    np.savez(path, md5=np.array(md5), **out)
    return out


def make_figure_w0(run=None, dataset="MNIST", fs=S.FS, width=S.COL_W, left=0.72, right=0.08, h_hidden=1.0,
                   h_output=0.8, gap=0.1, bottom=0.42, top=0.08, xlim=(-2.75, 3.75), bin_w=0.25):
    """w0 of the two-tailed learnable network before (grey outline) and after (network colour) training:
    histogram of the hidden-layer w0 (top) and of the output-layer w0 (bottom), same w0 axis and bins
    (0 is a bin edge, so no bin mixes the two regimes). Background: blue = bistable (w0 < 0),
    red = IF-like (w0 > 0). One column wide; sizes in inches."""
    run = run or two_tailed_run(dataset)
    w = load_w0(run, dataset)
    col, grey = S.net(run)["colour"], "#8C8C8C"
    W, ax_w = width, width - left - right
    H = bottom + h_output + gap + h_hidden + top
    fig = plt.figure(figsize=(W, H))
    ax_o = fig.add_axes([left / W, bottom / H, ax_w / W, h_output / H])
    ax_h = fig.add_axes([left / W, (bottom + h_output + gap) / H, ax_w / W, h_hidden / H], sharex=ax_o)
    bins = np.arange(np.floor(xlim[0] / bin_w) * bin_w, xlim[1] + bin_w / 2, bin_w)

    for ax, layer, ylabel in ((ax_h, "hidden", "hidden layer\n(neurons)"), (ax_o, "output", "output layer\n(neurons)")):
        ax.axvspan(xlim[0], 0, color=S.C_BISTABLE, alpha=0.07, lw=0, zorder=0)
        ax.axvspan(0, xlim[1], color=S.C_IFLIKE, alpha=0.07, lw=0, zorder=0)
        ax.axvline(0, color="#999999", lw=0.6, ls=":", zorder=1)
        ax.hist(w[f"w0_{layer}_final"], bins=bins, color=col, alpha=0.6, lw=0, label="learnt", zorder=2)
        ax.hist(w[f"w0_{layer}_initial"], bins=bins, histtype="step", color=grey, lw=1.0, label="initial",
                zorder=3)
        ax.set_ylabel(ylabel, fontsize=fs - 1, labelpad=2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
        S.style_ax(ax, fs)
    ax_o.set_xlim(*xlim)
    ax_o.set_xlabel(r"$w_0$", fontsize=fs, labelpad=1)
    ax_h.tick_params(labelbottom=False)

    # regime names and legend in the top panel, above the peaks
    ymax = ax_h.get_ylim()[1]
    ax_h.set_ylim(0, ymax * 1.45)
    ax_h.text(xlim[0] + 0.1, ymax * 1.4, "bistable", ha="left", va="top", fontsize=fs - 2,
              color=S.C_BISTABLE, fontweight="bold")
    ax_h.text(xlim[1] - 0.1, ymax * 1.4, "IF-like", ha="right", va="top", fontsize=fs - 2,
              color=S.C_IFLIKE, fontweight="bold")
    hs, ls = ax_h.get_legend_handles_labels()
    order = [ls.index("initial"), ls.index("learnt")]
    ax_h.legend([hs[i] for i in order], [ls[i] for i in order], loc="upper center", frameon=False,
                fontsize=fs - 2, handlelength=1.2, borderaxespad=0.1, handletextpad=0.4)
    ax_o.set_ylim(0, ax_o.get_ylim()[1] * 1.1)
    return fig


def w0_summary(run=None, dataset="MNIST"):
    """Numbers behind make_figure_w0: per layer, before and after training."""
    run = run or two_tailed_run(dataset)
    w = load_w0(run, dataset)
    rows = []
    for layer in ("hidden", "output"):
        a, b = w[f"w0_{layer}_initial"], w[f"w0_{layer}_final"]
        rows.append({"layer": layer, "n": len(a),
                     "bistable (w0<0) initial %": round(100 * (a < 0).mean(), 1),
                     "bistable (w0<0) learnt %": round(100 * (b < 0).mean(), 1),
                     "switched regime": int(((a < 0) != (b < 0)).sum()),
                     "mean w0 initial": round(a.mean(), 2), "mean w0 learnt": round(b.mean(), 2),
                     "mean |change|": round(np.abs(b - a).mean(), 2)})
    return pd.DataFrame(rows).set_index("layer")
