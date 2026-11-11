"""
style.py -- one style for every figure of the paper (Neuromorphic Computing and Engineering).

Import this first in every notebook of NICE_figures/:

    import style as S
    S.apply()                         # rcParams: fonts, line widths, embedded fonts in the PDF
    fig = ...
    S.save(fig, "fig1_neuron_model")  # -> NICE_figures/figures/fig1_neuron_model.pdf + .png

Colours are read from the drawing code, never copied (three network colours are set here: C_HETERO,
C_MIXED, C_UNIFORM below):
    ../single_neuron/neuron_figures.py   phase-plane colours (nullclines, spike, rest, reset ...)
    shared_plots.py                      w0 colour scale, LIF colour, ODR cue / distractor
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))            # .../NICE_figures
ROOT = os.path.dirname(HERE)                                  # repository root
SINGLE_NEURON = os.path.join(ROOT, "single_neuron")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# SINGLE_NEURON: neuron_figures.py, params.py.  HERE: shared_plots.py.
for p in (SINGLE_NEURON, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import neuron_figures as nf          # noqa: E402  (needs SINGLE_NEURON on the path)
import shared_plots as P             # noqa: E402

# ── sizes ────────────────────────────────────────────────────────────────────
FS = 9                     # base font size, as in the Cosyne figures (fs=9)
#   FS + 3  panel letters (bold)       FS + 1  panel titles, frame names
#   FS      axis labels                FS - 1  tick labels, small titles
#   FS - 2  legends, in-plot labels
COL_W = 3.5                # IOP / IEEE one-column width (in)
PAGE_W = 7.16              # full text width (in)

# ── colours (read from the source code, never copied) ────────────────────────
# neuron model / phase planes
C_VNULL, C_WNULL = nf.C_VNULL, nf.C_WNULL     # v-nullcline (black), w-nullcline (orange, dashed)
C_CYCLE = nf.C_CYCLE                           # limit cycle (red)
C_TRAJ = nf.C_TRAJ                             # spiking / persistent activity (light blue)
C_REST = nf.C_REST                             # rest (green)
C_RESET = nf.C_RESET                           # reset point (c, d)
C_SADDLE = nf.C_SADDLE
C_INPUT = "#333333"                            # I_ext trace
C_VTRACE = "#1f1f1f"                           # v trace

# neuron types: w0 scale (blue = bistable w0 < 0, red = IF-like w0 > 0)
W0_COLOURS = P.W0_COLOURS                      # {-2, -1, -0.5, +0.5, +1, +2}
C_BISTABLE = P.W0_COLOURS[-1.0]
C_IFLIKE = P.W0_COLOURS[1.0]
C_LIF = P.FIG2_MNIST_COLOURS["all_lif_wait100"]                        # black
# network colours of the paper
C_HETERO = "#8E44AD"                           # purple: heterogeneous (LIF hidden + bistable output)
C_MIXED = "#E377C2"                            # pink: two-tailed learnable w0
C_UNIFORM = "#009E73"                          # green: fixed uniform w0 (magenta would clash with the pink)

# network runs -> (label, colour), as in the MNIST rasters of the abstract
RUN_LABELS = P.FIG2_MNIST_LABELS
RUN_COLOURS = P.FIG2_MNIST_COLOURS

# ODR task
C_CUE = P.ODR_CUE_GREEN
C_DISTRACTOR = P.ODR_DIS_RED
# ODR networks, colour = w0 outside the RF during the hold (as the Fig. 2 colours):
# no RF (w0 = -1 everywhere) = bistable blue; RF, w0_out = 0 = the heterogeneous purple (bistable inside
# the RF, not outside); RF, w0_out = +1 = the w0 = +1 red of the colour scale.
ODR_W0_OUT_COLOURS = {-1: C_BISTABLE, 0: C_HETERO, 1: W0_COLOURS[1.0]}


def w0_colour(w0):
    return P.w0_colour(w0)


def w0_label(w0):
    """$w_0=-1$ with a true minus sign; $w_0=0$ without a sign."""
    return (rf"$w_0={w0:+g}$" if w0 else r"$w_0=0$").replace("-", "−")


# ── matplotlib defaults ──────────────────────────────────────────────────────
def apply(dpi=150):
    """Set the rcParams shared by every figure. Sizes set explicitly in the figure code win."""
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": 300,
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS - 1,
        "ytick.labelsize": FS - 1,
        "legend.fontsize": FS - 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,          # embed TrueType fonts (journals refuse Type 3)
        "ps.fonttype": 42,
    })


def style_ax(ax, fs=FS):
    """Ticks and spines as in the Cosyne figures."""
    ax.tick_params(labelsize=fs - 1, length=2.0, pad=1.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def letter(fig, x_in, y_in, text, W, H, fs=FS):
    """Bold panel letter at (x, y) inches from the lower-left corner."""
    fig.text(x_in / W, y_in / H, text, fontsize=fs + 3, fontweight="bold", va="bottom")


def legend_frame(fig, artists, pad_in=0.07, colour="#8C8C8C", lw=0.6, radius_in=0.05):
    """Thin rounded grey frame around a group of legend artists (legends, axes, texts) of `fig`, the same
    in every figure. The frame measures its artists each time the figure is drawn, so it fits them in the
    notebook, in the 300 dpi PNG and in the PDF alike (text with super/subscripts is laid out a little
    differently at each resolution), and moves with them when save() crops the figure. Returns the patch."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Bbox, IdentityTransform

    class _GroupFrame(FancyBboxPatch):
        def _fit(self, renderer):
            dpi = fig.dpi
            boxes = [a.get_tightbbox(renderer) if hasattr(a, "get_tightbbox") else a.get_window_extent(renderer)
                     for a in artists]
            bb, p = Bbox.union(boxes), pad_in * dpi                        # display units (pixels)
            self.set_bounds(bb.x0 - p, bb.y0 - p, bb.width + 2 * p, bb.height + 2 * p)
            self.set_boxstyle("round", pad=0, rounding_size=radius_in * dpi)

        def draw(self, renderer):
            self._fit(renderer)
            super().draw(renderer)

        def get_window_extent(self, renderer=None):
            if renderer is not None:
                self._fit(renderer)
            return super().get_window_extent(renderer)

    frame = _GroupFrame((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=1", transform=IdentityTransform(),
                        fc="none", ec=colour, lw=lw, zorder=-1)
    fig.add_artist(frame)
    frame._fit(fig.canvas.get_renderer())
    return frame


def save(fig, name, png=True):
    """Save to NICE_figures/figures/<name>.pdf (and .png at 300 dpi)."""
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"), bbox_inches="tight")
    if png:
        fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300, bbox_inches="tight")
    return os.path.join(FIG_DIR, name + ".pdf")


# ── network types (MNIST / N-MNIST runs) ─────────────────────────────────────
# One entry per trained network: label, colour, line style, marker, group, open marker.
# Rules (the same in every figure, MNIST and N-MNIST):
#   lines            -> SOLID for every constant-w0 network, DOTTED for every other network
#   constant w0      -> the w0 colour scale above (blue = bistable w0 < 0, red = IF-like w0 > 0)
#   LIF              -> black / grey
#   mixed layers     -> heterogeneous (LIF hidden + bistable output) = purple
#   w0 distributions -> two-tailed learnable = pink, fixed uniform = green;
#                       open marker = w0 fixed after sampling, filled = w0 learnt
DOT = (0, (1.2, 1.4))


def _net(label, colour, ls, marker, group, open_marker=False):
    return dict(label=label, colour=colour, ls=ls, marker=marker, group=group, open=open_marker)


NETWORKS = {
    "all_lif_wait100": _net("LIF", C_LIF, DOT, "x", "LIF"),
    "all_lif_wait0": _net("LIF, trained without delay", "#8A8A8A", DOT, "+", "LIF"),
}
for _w0 in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
    NETWORKS[f"const_w0_{_w0:+.2f}"] = _net(w0_label(_w0), W0_COLOURS[_w0], "-", "o",
                                           "constant w0")
HETERO_LABEL = "heterogeneous (hidden: LIF, output: bistable)"
NETWORKS.update({
    "lif_hidden_bistable_output_w0_-1.00": _net(HETERO_LABEL, C_HETERO, DOT, "s", "mixed layers"),
    "bistable_hidden_w0_-1.00_lif_output": _net(r"bistable hidden ($w_0=-1$), LIF output", "#7FA31A", DOT, "v",
                                                "mixed layers"),
})


def uniform_range(mean, std):
    """sample_w0('uniform', mean, std) draws from U[mean - std*sqrt(3), mean + std*sqrt(3)] (variance = std^2)."""
    h = std * 3 ** 0.5
    return mean - h, mean + h


# w0 distributions (the paper uses mean -1 on both datasets; mean -1.25 was an earlier N-MNIST run): same colour / marker per kind,
# label written from the run's own parameters.
for _m in (-1.0, -1.25):
    _t = f"mean{_m:+.2f}_std0.50"
    _lo, _hi = uniform_range(_m, 0.5)
    NETWORKS.update({
        f"dist_two_tailed_{_t}_learnable": _net(rf"two-tailed $w_0$ ($\pm{abs(_m):g}$), learnt", C_MIXED, DOT, "D",
                                                "w0 distribution"),
        # label rounded to whole numbers: U[-1.87, -0.13] -> U[-2, 0]; exact range in the text
        f"dist_uniform_{_t}_fixed": _net(rf"$w_0\sim\mathcal{{U}}[{int(round(_lo))},\,{int(round(_hi))}]$, fixed", C_UNIFORM, DOT, "^",
                                         "w0 distribution", open_marker=True),
        f"dist_normal_1_{_t}_learnable": _net(rf"$w_0\sim\mathcal{{N}}({_m:g},0.5)$, learnt", "#4B2C7F", DOT, "o",
                                              "w0 distribution"),
        f"dist_normal_{_t}_fixed": _net(rf"$w_0\sim\mathcal{{N}}({_m:g},0.5)$, fixed", "#4B2C7F", DOT, "s",
                                        "w0 distribution", open_marker=True),
    })
NETWORKS["dist_normal_0_mean+0.00_std0.50_learnable"] = _net(r"$w_0\sim\mathcal{N}(0,0.5)$, learnt", "#B9A0D6",
                                                             DOT, "o", "w0 distribution")
NETWORK_ORDER = list(NETWORKS)          # the order used in legends and raster stacks


# ── the networks shown in the paper (same selection for MNIST and N-MNIST) ───────────────
# LIF (trained with the delay), every constant w0, the heterogeneous network, the two-tailed
# learnt w0 and the fixed uniform w0 (mean -1, s.d. 0.5), on both datasets.
_CONST = [f"const_w0_{w:+.2f}" for w in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)]
PAPER_RUNS = {
    "MNIST": ["all_lif_wait100", *_CONST, "lif_hidden_bistable_output_w0_-1.00",
              "dist_two_tailed_mean-1.00_std0.50_learnable", "dist_uniform_mean-1.00_std0.50_fixed"],
    # N-MNIST: the same selection (the w0 distributions with mean -1, as MNIST)
    "N-MNIST": ["all_lif_wait100", *_CONST, "lif_hidden_bistable_output_w0_-1.00",
                "dist_two_tailed_mean-1.00_std0.50_learnable", "dist_uniform_mean-1.00_std0.50_fixed"],
}


def net(run):
    """Style of one network; unknown runs get a neutral grey, so nothing silently takes a used colour."""
    return NETWORKS.get(run, _net(run, "#BBBBBB", DOT, ".", "other"))