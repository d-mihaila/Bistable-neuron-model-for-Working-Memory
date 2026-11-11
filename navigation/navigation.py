"""
navigation.py

Spatial working memory for navigation: the ODR ring (bistable neurons, RF gate,
forget pulse) used to steer an agent to an apple that is shown only once.
navigation.ipynb runs everything; nothing runs on import. From a terminal:
    python navigation.py --procs 6          (runs or resumes every experiment)

The task
--------
- A 10 x 10 board with walls, one agent, one apple at a time. The agent steps
  N, E, S or W (straight back allowed).
- Each apple is shown ONCE (one look): forget window (300 ms), cue (100 ms).
  Then the agent walks from memory, one step per 200 ms slot.
- The apple is eaten if the agent reaches it. It is lost if no decision neuron
  fires (two empty slots), the counters are used up on the wrong side, or the
  agent walks into a wall. Then a new apple appears.
- A run is 30 apples; 10 runs (seeds 0-9) per condition.

The network (network.py)
------------------------
16 E ring (no I), apple cue, 4 decision neurons (E, N, W, S) with mutual
inhibition, 2 counters (distance meters: E/W and N/S). Three versions:
    all bistable  : bistable ring + bistable decision neurons
    heterogeneous : bistable ring + LIF decision neurons
    all LIF       : LIF ring + LIF decision neurons (no w0: no RF, no memory)
The LIF is the MNIST / N-MNIST LIF. Its settings are one dict (LIF_DEFAULT),
which the notebook can change (lif=...).

Experiments
-----------
    base         : the three networks (RF + forget, no noise)
    input_noise  : the three networks, input CV 10-50% (cue rate of every ring
                   neuron x its own gain ~ N(1, CV), clipped at 0, new at every look)
    rf_forget    : the two bistable rings without RF, without forget, without both
    weight_noise : the two bistable rings, weight CV 10-50% (every synapse of the
                   ring and decision layer x (1 + CV xi), once per run; as in MNIST)
Every run is one row of results/runs.csv (all metrics); run_all resumes.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from network import SnakeNet, cue_profile, signed_angle, DIR_NAMES, DIR_ANGLES, HEADING_STEPS
from board import Board

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_E = 16
BOARD = 10
N_APPLES = 30              # apples per run, each shown once
SLOT_MS = 200
CUE_RATE_HZ = 750.0        # peak cue rate (Hz)
CUE_W_BISTABLE = 0.084     # current per cue spike, bistable ring
TYPES = {"all bistable": ("bistable", "bistable"),          # (ring, decision)
         "heterogeneous": ("bistable", "lif"),
         "all LIF": ("lif", "lif")}
TYPE_ORDER = list(TYPES)
BISTABLE_RINGS = ["all bistable", "heterogeneous"]
CV_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# decision layer
DEC_FADE = 90.0                                              # each decision neuron reads the ring within 90 deg
BISTABLE_DEC = dict(g_de=100.0, g_dd=50.0, tonic=0.05)       # bistable decision neurons (input x dt_eff 0.025)
DEC_SCALE = {"lif": 1.0, "bistable": 100.0 / 6.0}            # counter -> decision scale per decision model

# The LIF (MNIST / N-MNIST LIF_simple). The notebook may change any of these.
LIF_DEFAULT = dict(
    beta=0.5,        # voltage kept per ms (MNIST: 0.5). Lower = leakier = less excitable
    threshold=1.0,   # spike threshold (MNIST: 1). Higher = less excitable
    cue_w=0.4,       # LIF ring: current per cue spike. Rule: as many cue-window spikes as the bistable ring
    ring_gain=1.0,   # LIF ring: factor on the E -> E weights
    g_de=40.0,       # LIF decision: ring -> decision gain. Rule: one ring spike 45 deg off fires it
    g_dd=3.0,        # LIF decision: one decision spike inhibits the others by this much
    tonic=0.015,     # LIF decision: tonic drive on all four (bistable: 0.05)
)

# counters (distance meters)
AXIS = np.array([0, 1, 0, 1])          # decision E, N, W, S -> counter 0 (E/W) or 1 (N/S)
AXIS_NAMES = ["E/W", "N/S"]
ENC_MAP = np.eye(2)[AXIS]              # (4 decision, 2 counters): which decision neurons a counter blocks
ENC_STEP = 0.25                        # one step = one kick of +0.25
G_ENC = 10.0                           # counter -> decision inhibition (LIF units; x16.7 for bistable decision)
ENC_SELF = 1.002 * (1.0 - float(np.exp(-1.0 / 5.0)))        # self-excitation: once it fires it keeps firing (~36 Hz)
ENC_TAU_SYN = 20.0                     # ms: slow inhibitory synapse, so a firing counter blocks without gaps

CONDITIONS = {(True, True): "RF + forget", (False, True): "no RF", (True, False): "no forget",
              (False, False): "no RF, no forget"}
COND_ORDER = list(CONDITIONS.values())
OUT_DIR = "results"


def cond_name(rf=True, forget=True):
    return CONDITIONS[(bool(rf), bool(forget))]


def lif_params(lif=None):
    p = dict(LIF_DEFAULT)
    p.update(lif or {})
    return p


def lif_tag(lif=None):
    p = lif_params(lif)
    return (f"beta{p['beta']:g}_th{p['threshold']:g}_cue{p['cue_w']:g}_gain{p['ring_gain']:g}_gde{p['g_de']:g}"
            f"_gdd{p['g_dd']:g}_tonic{p['tonic']:g}")


def has_lif(net_type):
    return "lif" in TYPES[net_type]


# ---------------------------------------------------------------------------
# Network and counters
# ---------------------------------------------------------------------------

def new_net(net_type="all bistable", seed=0, rf=True, forget=True, lif=None, weight_cv=0.0):
    """16 E ring (no I), 4 decision neurons, 2 counters.
    rf=False: w0 = -1 on the whole ring after the cue. forget=False: no forget window
    (w0 stays -1, no pulse). weight_cv: weight noise (network.add_weight_noise)."""
    p = lif_params(lif)
    e_model, dec_model = TYPES[net_type]
    kw = dict(e_model=e_model, dec_model=dec_model, lif_beta=p["beta"], lif_threshold=p["threshold"],
              stim_weight_e=p["cue_w"] if e_model == "lif" else CUE_W_BISTABLE,
              ee_gain=p["ring_gain"] if e_model == "lif" else 1.0,
              dec_fade_deg=DEC_FADE, g_enc=G_ENC * DEC_SCALE[dec_model], enc_self=ENC_SELF,
              enc_tau_syn=ENC_TAU_SYN, enc_map=ENC_MAP, seed=seed)
    if dec_model == "lif":
        kw.update(g_de=p["g_de"], g_dd=p["g_dd"])
    else:
        kw.update(g_de=BISTABLE_DEC["g_de"], g_dd=BISTABLE_DEC["g_dd"])
    if not rf:
        kw["w0_out"] = -1.0
    if not forget:
        kw.update(w0_pre=-1.0, pre_amp=0.0)
    net = SnakeNet(N_E, **kw)
    net.add_weight_noise(weight_cv, seed=30_000 + int(seed))
    net.type_name = net_type
    net.tonic = p["tonic"] if dec_model == "lif" else BISTABLE_DEC["tonic"]
    return net


def enc_start(n_steps, step=ENC_STEP):
    """Start voltage of a counter that allows n steps (n = 0: above threshold, closed)."""
    return 1.0 - (np.asarray(n_steps, float) - 0.5) * step


def steps_left(v, step=ENC_STEP):
    return np.maximum(0, np.round((1.0 - np.asarray(v, float)) / step + 0.5 - 1e-9)).astype(int)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

@dataclass
class RunLog:
    net_type: str
    rf: bool
    forget: bool
    cv_input: float
    cv_weight: float
    size: int
    angles_deg: np.ndarray
    dec_profile: np.ndarray
    meta: dict = field(default_factory=dict)
    trials: list = field(default_factory=list)
    hits: int = 0
    time_ms: int = 0
    spikes: dict = field(default_factory=lambda: dict(e=0, d=0, enc=0, stim=0))

    def summary(self):
        n = len(self.trials)
        sp = self.spikes
        net = sp["e"] + sp["d"] + sp["enc"]
        inp = sp["stim"]
        ends = [t["end"] for t in self.trials]
        hit_trials = [t for t in self.trials if t["end"] == "apple"]
        excess = sum(len(t["steps"]) - t["shortest"] for t in hit_trials)
        rf_err = [abs(float(signed_angle(t["rf_center"] - t["theta"]))) for t in self.trials]
        H, tot = self.hits, net + inp
        return dict(net_type=self.net_type, rf=self.rf, forget=self.forget, condition=cond_name(self.rf, self.forget),
                    cv_input=self.cv_input, cv_weight=self.cv_weight, board=self.size, apples_shown=n, apples=H,
                    hit_rate=H / n if n else np.nan, ends_memory=ends.count("memory"), ends_wall=ends.count("wall"),
                    ends_route_done=ends.count("route done"), ends_timeout=ends.count("timeout"),
                    steps=sum(len(t["steps"]) for t in self.trials), excess_steps=excess,
                    closed_steps=sum(s["closed"] for t in self.trials for s in t["steps"]),
                    rf_err_deg=float(np.mean(rf_err)) if rf_err else np.nan, time_s=self.time_ms / 1000,
                    all_spikes=tot, net_spikes=net, input_spikes=inp, e_spikes=sp["e"], d_spikes=sp["d"],
                    enc_spikes=sp["enc"], cue_spikes=sp["stim"],
                    spikes_per_apple=tot / H if H else np.nan, spikes_per_apple_shown=tot / n if n else np.nan,
                    apples_per_10k=1e4 * H / tot if tot else np.nan)


def play(net, board, n_apples=N_APPLES, cv_input=0.0, seed=0, record=False, verbose=False):
    """n_apples apples, each shown once. Returns a RunLog."""
    log = RunLog(net_type=getattr(net, "type_name", ""), rf=net.w0_out == 0.0,
                 forget=not (net.w0_pre == -1.0 and net.pre_amp == 0.0), cv_input=cv_input, cv_weight=net.weight_cv,
                 size=board.size, angles_deg=net.angles_deg, dec_profile=net.dec_profile,
                 meta=dict(w0_pre=net.w0_pre, pre_amp=net.pre_amp, w0_mem=net.w0_mem, w0_out=net.w0_out,
                           rf_radius_deg=net.rf_radius_deg, n_excit=net.n_excit, e_model=net.e_model,
                           dec_model=net.dec_model, slot_ms=SLOT_MS))
    noise_rng = np.random.default_rng(20_000 + int(seed))      # input noise (same draws as NSEW_minimal)
    state = None
    n = len(net.angles_deg)
    tonic = np.full(4, net.tonic)
    for trial in range(n_apples):
        apple, pos0 = board.apple, board.pos
        dx, dy = apple[0] - pos0[0], apple[1] - pos0[1]
        theta = float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)
        cue = cue_profile(net.angles_deg, theta, net.stim_width_deg, CUE_RATE_HZ)
        if cv_input > 0:
            cue = cue * np.clip(noise_rng.normal(1.0, cv_input, n), 0.0, None)
        look = net.look_begin(cue[None, :], state=state, record=record)          # the only time it is shown
        n_axis = np.array([abs(dx), abs(dy)])
        v0 = enc_start(n_axis)
        net.enc_set(look, v0[None, :])                 # the counters get the distance per axis (no direction)
        info = dict(index=trial, pos=pos0, apple=apple, r=float(np.hypot(dx, dy)), theta=theta,
                    rf_center=float(look["rf_center"][0]), n_axis=n_axis, enc_start=v0, cue=cue,
                    shortest=int(abs(dx) + abs(dy)), steps=[], slot_log=[], end=None)

        def closed_now():
            v, recent = net.enc_state(look)
            c_axis = (v[0] >= 1.0) | recent[0]
            return c_axis[AXIS], v[0], c_axis

        def slot():
            closed, v, c_axis = closed_now()
            w, c = net.look_slot(look, SLOT_MS, dec_bias=tonic)
            info["slot_log"].append(dict(winner=int(w[0]), counts=c[0], step=False, closed=closed.copy(),
                                         enc_v=v.copy(), axis_closed=c_axis.copy(),
                                         left=np.where(c_axis, 0, steps_left(v)), pos=board.pos))
            return int(w[0]), closed

        end, n_slots, max_slots = None, 0, 2 * info["shortest"] + 4
        while end is None:
            _, _, c_axis = closed_now()
            if c_axis.all():
                end = "route done"                     # both counters used up, no apple: the direction was wrong
                break
            if n_slots >= max_slots:
                end = "timeout"
                break
            w, closed = slot()
            n_slots += 1
            if w < 0:
                w, closed = slot()                     # one retry slot
                n_slots += 1
            if w < 0:
                end = "memory"                         # no decision neuron fires: the memory does not drive them
                break
            info["slot_log"][-1]["step"] = True
            res = board.step(w)
            info["steps"].append(dict(move=w, pos=board.pos, got=res["ate"], collision=res["collision"],
                                      closed=bool(closed[w])))
            if res["collision"]:
                end = "wall"
                break
            net.enc_kick(look, AXIS[w], ENC_STEP)      # a copy of the step goes to its axis counter
            if res["ate"]:
                end = "apple"
        if end == "apple":
            log.hits += 1
        else:
            board.apple = board.next_apple()           # the apple is lost; a new one appears
        info["end"] = end
        rec, state = net.look_end(look)
        for k, v in rec["counts"].items():
            if k in log.spikes:
                log.spikes[k] += int(np.asarray(v).sum())
        log.time_ms += rec["T"]
        info.update(T=rec["T"], rec=rec if record else None)
        log.trials.append(info)
        if verbose:
            path = "".join(DIR_NAMES[s_["move"]] for s_ in info["steps"]) or "-"
            print(f"apple {trial:2d} at {apple} from {pos0}: {theta:5.1f} deg, RF {info['rf_center']:5.1f}, "
                  f"|dx| {abs(dx)} |dy| {abs(dy)} -> {path:14s} {end}")
    return log


def run_one(net_type, seed=0, rf=True, forget=True, cv_input=0.0, cv_weight=0.0, lif=None, n_apples=N_APPLES,
            size=BOARD, record=False, verbose=False):
    """One run of n_apples apples. Returns (summary row, RunLog)."""
    net = new_net(net_type, seed, rf=rf, forget=forget, lif=lif, weight_cv=cv_weight)
    with torch.no_grad():
        lg = play(net, Board(size, seed=seed), n_apples=n_apples, cv_input=cv_input, seed=seed, record=record,
                  verbose=verbose)
    row = lg.summary()
    row.update(seed=seed, lif_tag=lif_tag(lif) if has_lif(net_type) else "-",
               **{f"lif_{k}": v for k, v in lif_params(lif).items()})
    return row, lg


def run_trial(net_type="all bistable", pos=(3, 5), apple=(6, 3), seed=0, rf=True, forget=True, lif=None,
              size=BOARD):
    """One apple from pos, everything recorded (for the one-trial figure)."""
    net = new_net(net_type, seed, rf=rf, forget=forget, lif=lif)
    with torch.no_grad():
        return play(net, Board(size, seed=seed, start=pos, apple=apple), n_apples=1, seed=seed, record=True)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_configs():
    cfg = lambda exp, t, rf=True, fg=True, ci=0.0, cw=0.0: dict(exp=exp, net_type=t, rf=rf, forget=fg, cv_input=ci,
                                                                cv_weight=cw)
    cfgs = [cfg("base", t) for t in TYPE_ORDER]
    cfgs += [cfg("input_noise", t, ci=cv) for t in TYPE_ORDER for cv in CV_LEVELS[1:]]
    cfgs += [cfg("rf_forget", t, rf=rf, fg=fg) for t in BISTABLE_RINGS
             for rf, fg in ((False, True), (True, False), (False, False))]
    cfgs += [cfg("weight_noise", t, cw=cv) for t in BISTABLE_RINGS for cv in CV_LEVELS[1:]]
    return cfgs


def _job(args):
    torch.set_num_threads(1)
    cfg, seed, lif = args
    t0 = time.time()
    row, _ = run_one(cfg["net_type"], seed, rf=cfg["rf"], forget=cfg["forget"], cv_input=cfg["cv_input"],
                     cv_weight=cfg["cv_weight"], lif=lif, size=cfg.get("board", BOARD))
    row.update(exp=cfg["exp"], wall_s=time.time() - t0)
    return row


def _key(r):
    return (r["exp"], r["net_type"], str(r["rf"]) == "True", str(r["forget"]) == "True",
            round(float(r["cv_input"]), 3), round(float(r["cv_weight"]), 3), int(r["seed"]), int(r.get("board", BOARD)),
            str(r["lif_tag"]))


def run_all(out_csv=None, seeds=range(10), n_proc=1, lif=None, configs=None, verbose=True):
    """Run every configuration x seed not yet in out_csv, appending row by row (an
    interrupted run resumes). Runs of a network with LIF neurons are kept apart by
    their LIF settings (lif_tag): new settings -> new runs. Returns the table."""
    import pandas as pd
    out_csv = os.path.join(OUT_DIR, "runs.csv") if out_csv is None else out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    cfgs = [dict(c, board=c.get("board", BOARD)) for c in (experiment_configs() if configs is None else configs)]
    done, cols = set(), None
    if os.path.exists(out_csv):
        d = pd.read_csv(out_csv, dtype=str)
        cols = list(d.columns)
        done = {_key(r) for _, r in d.iterrows()}
    tag = lambda c: lif_tag(lif) if has_lif(c["net_type"]) else "-"
    jobs = [(c, s, lif) for c in cfgs for s in seeds if _key(dict(c, seed=s, lif_tag=tag(c))) not in done]
    if verbose:
        print(f"{len(jobs)} runs to do ({len(done)} already in {out_csv})", flush=True)
    t0 = time.time()

    def write(row, i):
        nonlocal cols
        df = pd.DataFrame([row])
        if cols is None:
            cols = list(df.columns)
            df.to_csv(out_csv, index=False)
        else:
            df.reindex(columns=cols).to_csv(out_csv, mode="a", header=False, index=False)
        if verbose:
            print(f"{i + 1}/{len(jobs)} {row['exp']} | {row['net_type']} | {row['condition']} | CV in {row['cv_input']} "
                  f"w {row['cv_weight']} | s{row['seed']}: {row['apples']}/{row['apples_shown']} apples, "
                  f"{row['wall_s']:.0f} s (total {time.time() - t0:.0f} s)", flush=True)

    if n_proc > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(n_proc) as pool:
            for i, row in enumerate(pool.imap_unordered(_job, jobs)):
                write(row, i)
    else:
        for i, j in enumerate(jobs):
            write(_job(j), i)
    return pd.read_csv(out_csv)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def tables(df, lif=None):
    """Split the runs for the current LIF settings into the four experiments:
    base, input_noise (with CV 0 from base), rf_forget (bistable rings, the four
    conditions), weight_noise (bistable rings, with CV 0 from base)."""
    import pandas as pd
    tag = lif_tag(lif)
    df = df[(df["lif_tag"].astype(str) == "-") | (df["lif_tag"].astype(str) == tag)]
    base = df[df.exp == "base"]
    out = dict(base=base,
               input_noise=pd.concat([base, df[df.exp == "input_noise"]]),
               rf_forget=pd.concat([base[base.net_type.isin(BISTABLE_RINGS)], df[df.exp == "rf_forget"]]),
               weight_noise=pd.concat([base[base.net_type.isin(BISTABLE_RINGS)], df[df.exp == "weight_noise"]]))
    return out


def pooled(df, by):
    """Sums over runs, then per apple eaten (pooled) and apples per 10,000 spikes."""
    g = df.groupby(by, sort=False)
    cols = ["apples", "apples_shown", "all_spikes", "net_spikes", "input_spikes", "e_spikes", "d_spikes", "enc_spikes",
            "cue_spikes", "ends_memory", "ends_wall", "ends_route_done", "ends_timeout", "excess_steps", "closed_steps"]
    st = g[cols].sum()
    st["runs"] = g.size()
    st["hit_rate_mean"] = g["hit_rate"].mean()
    st["hit_rate_sd"] = g["hit_rate"].std().fillna(0)
    st["rf_err_deg"] = g["rf_err_deg"].mean()
    st["apples_per_10k_sd"] = g["apples_per_10k"].std().fillna(0)
    A = st["apples"].where(st["apples"] > 0)
    st["spikes_per_apple"] = st["all_spikes"] / A
    st["spikes_per_apple_shown"] = st["all_spikes"] / st["apples_shown"]
    st["apples_per_10k"] = 1e4 * st["apples"] / st["all_spikes"]
    for c in ("input_spikes", "e_spikes", "d_spikes", "enc_spikes"):
        st[c + "_per_apple"] = st[c] / A
    return st


def table(df, by, order=None):
    """Readable table: one row per group."""
    import pandas as pd
    st = pooled(df, by)
    if order is not None:
        st = st.loc[[o for o in order if o in st.index]]
    return pd.DataFrame({
        "runs": st["runs"],
        "apples eaten (%)": [f"{100 * m:.0f} ± {100 * s:.0f}" for m, s in zip(st["hit_rate_mean"], st["hit_rate_sd"])],
        "lost: no decision": st["ends_memory"].astype(int),
        "lost: wrong side": st["ends_route_done"].astype(int),
        "lost: wall": st["ends_wall"].astype(int),
        "direction error (°)": st["rf_err_deg"].round(1),
        "input spikes / apple": st["input_spikes_per_apple"].round(0),
        "ring spikes / apple": st["e_spikes_per_apple"].round(0),
        "decision spikes / apple": st["d_spikes_per_apple"].round(0),
        "counter spikes / apple": st["enc_spikes_per_apple"].round(0),
        "all spikes / apple eaten": st["spikes_per_apple"].round(0),
        "all spikes / apple shown": st["spikes_per_apple_shown"].round(0),
        "apples / 10k spikes": st["apples_per_10k"].round(1),
    })


def save_table(tab, name, out_dir=OUT_DIR):
    d = os.path.join(out_dir, "tables")
    os.makedirs(d, exist_ok=True)
    tab.to_csv(os.path.join(d, name))
    return tab


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

APPLE_C, AGENT_C, TRAIL_C = "#1a9850", "#5c2e0e", "#a0522d"
EXC_C, INH_C, FORGET_C, INK, MUTED, GRID = "#2166ac", "#b2182b", "#7b3fa0", "#222222", "#8c8c8c", "#e3e3e3"
CUE_BG, RF_BG, CLOSED_BG, ENC_C = "#deebf7", "#9ecae1", "#d9d9d9", "#d95f02"
TYPE_COLORS = {"all bistable": "#E69F00", "heterogeneous": "#08519c", "all LIF": "#D55E00"}
COND_COLORS = {"RF + forget": "#4a90c2", "no RF": "#bdbdbd", "no forget": "#b39ddb", "no RF, no forget": "#525252"}
SOURCES = [("apple cue (input)", "input_spikes", EXC_C), ("ring E", "e_spikes", "#6a51a3"),
           ("decision", "d_spikes", "#E69F00"), ("counters", "enc_spikes", ENC_C)]


def _mpl():
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_architecture(fs=12, figsize=(9.4, 10.2)):
    """The minimal network: ring, apple cue, 4 decision neurons with mutual
    inhibition, 2 counters and the decision neurons they block."""
    plt = _mpl()
    from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Arc
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    u = lambda a: np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])

    def arrow(ax, p, q, col, lw=2.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=col, lw=lw, mutation_scale=15, shrinkA=0,
                                     shrinkB=0, zorder=3))

    def bar(ax, p, direction, col, half=0.13, lw=2.2):
        """Inhibitory synapse: a short bar at p, perpendicular to direction."""
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        n = np.array([-d[1], d[0]])
        ax.plot(*np.c_[p - half * n, p + half * n], color=col, lw=lw, solid_capstyle="butt", zorder=4)

    def wire(ax, pts, col, lw=1.9, halo=False):
        """Inhibitory wire along pts, a bar at the end (the target)."""
        pts = np.asarray(pts, float)
        kw = dict(path_effects=[pe.Stroke(linewidth=lw + 4, foreground="white"), pe.Normal()]) if halo else {}
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, zorder=3, solid_joinstyle="round", **kw)
        bar(ax, pts[-1], pts[-1] - pts[-2], col)

    with plt.rc_context({"font.size": fs}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-4.4, 4.4); ax.set_ylim(-5.9, 4.1)
        # ring
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color=EXC_C, lw=1.0, alpha=0.35, zorder=2)
        for a in np.arange(16) * 22.5:
            ax.add_patch(Circle(u(a), 0.085, fc=EXC_C, ec="white", lw=0.8, zorder=4))
        ax.text(0, 0.18, "ring", ha="center", va="center", fontsize=fs + 1, color=EXC_C, fontweight="bold")
        ax.text(0, -0.3, "16 E neurons;\nneighbours excite\neach other", ha="center", va="center", fontsize=fs - 3,
                color=MUTED)
        # apple cue
        apple = np.array([2.95, 2.95])
        ax.add_patch(Circle(apple, 0.27, fc=APPLE_C, ec="white", zorder=5))
        arrow(ax, apple - 0.3 * u(45), 1.13 * u(45), APPLE_C, lw=2.6)
        ax.text(apple[0], apple[1] + 0.4, "apple cue\n(shown once)", ha="center", va="bottom", fontsize=fs - 1,
                color=APPLE_C)
        # decision neurons, read the ring
        R = 2.05
        pos = {}
        for k, a in zip(DIR_NAMES, DIR_ANGLES):
            p = R * u(a)
            pos[k] = p
            ax.add_patch(Circle(p, 0.28, fc="white", ec=INK, lw=2, zorder=5))
            ax.text(*p, k, ha="center", va="center", fontsize=fs + 1, fontweight="bold", zorder=6)
            arrow(ax, 1.13 * u(a), (R - 0.3) * u(a), EXC_C, lw=2.2)
        # mutual inhibition: arcs between neighbouring decision neurons, a bar at both ends
        for a in DIR_ANGLES:
            a0, a1 = a + 11, a + 79
            ax.add_patch(Arc((0, 0), 2 * R, 2 * R, theta1=a0, theta2=a1, color=INH_C, lw=1.6, zorder=3))
            bar(ax, R * u(a0), u(a0 + 90), INH_C)
            bar(ax, R * u(a1), u(a1 + 90), INH_C)
        ax.text(*(1.62 * u(225)), "inhibit\neach other", ha="center", va="center", fontsize=fs - 3, color=INH_C,
                rotation=-45)
        # counters, wired to the two decision neurons of their axis
        ns, ew = np.array([-3.45, 3.25]), np.array([3.45, -3.3])
        for name, c in (("N/S", ns), ("E/W", ew)):
            ax.add_patch(FancyBboxPatch((c[0] - 0.75, c[1] - 0.32), 1.5, 0.64, boxstyle="round,pad=0.03", fc="#fdd49e",
                                        ec=INK, lw=1.5, zorder=5))
            ax.text(*c, f"{name}\ncounter", ha="center", va="center", fontsize=fs - 2, fontweight="bold", zorder=6)
        wire(ax, [ns + [0.77, 0], [0, ns[1]], pos["N"] + [0, 0.3]], INH_C)                       # N/S -> N
        wire(ax, [ns - [0, 0.34], [ns[0], -R], pos["S"] - [0.3, 0]], INH_C)                      # N/S -> S
        wire(ax, [ew + [0, 0.34], [ew[0], 0], pos["E"] + [0.3, 0]], INH_C)                       # E/W -> E
        wire(ax, [ew - [0.77, 0], [-R, ew[1]], pos["W"] - [0, 0.3]], INH_C, halo=True)           # E/W -> W
        ax.text(ns[0], ns[1] - 0.45, "set at the look:\n|dy| steps allowed;\n+1 kick per\nN or S step",
                ha="left", va="top", fontsize=fs - 3, color=ENC_C, transform=ax.transData,
                bbox=dict(fc="white", ec="none", pad=0.5))
        ax.text(ew[0], ew[1] - 0.45, "set at the look: |dx| steps allowed;\n+1 kick per E or W step",
                ha="center", va="top", fontsize=fs - 3, color=ENC_C)
        # legend and caption
        h = [Line2D([], [], color=EXC_C, lw=2.2, marker=">", ms=7, label="excites"),
             Line2D([], [], color=INH_C, lw=1.8, marker="|", ms=10, mew=2.2, label="inhibits"),
             Line2D([], [], color=APPLE_C, lw=2.6, marker=">", ms=7, label="cue input")]
        ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.215), ncol=3, frameon=False, fontsize=fs - 2)
        ax.text(-4.35, -4.75,
                "Ring: the cue starts a bump. Inside the RF (w0 = −1) it keeps firing: the memory.\n"
                "Decision neurons: each reads the ring within 90° of its direction. They inhibit each other\n"
                "(every pair); the one with most spikes in 200 ms is the step.\n"
                "Counters: one neuron per axis that only adds up. Full after |dx| (or |dy|) steps; then it fires,\n"
                "keeps firing, and blocks both decision neurons of its axis. It knows how far, not which way.",
                fontsize=fs - 3, va="top", ha="left")
        ax.set_title("The network", loc="left", fontsize=fs + 2)
    return fig


def draw_board(ax, size, pos, apple, trail=None, apple_on=True, fs=10, title=None):
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    ax.set_aspect("equal")
    for i in range(size + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, size - 0.5], color=GRID, lw=0.8, zorder=0)
        ax.plot([-0.5, size - 0.5], [i - 0.5, i - 0.5], color=GRID, lw=0.8, zorder=0)
    if trail is not None and len(trail) > 1:
        tr = np.array(trail, float)
        ax.plot(tr[:, 0], tr[:, 1], color=TRAIL_C, alpha=0.3, lw=2.2, zorder=1)
    ax.add_patch(FancyBboxPatch((pos[0] - 0.4, pos[1] - 0.4), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                                color=AGENT_C, lw=0, zorder=3))
    if apple_on and apple is not None:
        ax.add_patch(Circle(apple, 0.36, color=APPLE_C, zorder=3))
    ax.add_patch(Rectangle((-0.5, -0.5), size, size, fill=False, ec=INK, lw=3, zorder=6))
    ax.set_xlim(-0.5, size - 0.5); ax.set_ylim(-0.5, size - 0.5)
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.tick_params(labelsize=fs - 3, colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=fs)


def draw_route(ax, start, steps, lw=2.4):
    prev = start
    for s_ in steps:
        sx, sy = HEADING_STEPS[s_["move"]]
        if s_["collision"]:
            ax.plot(prev[0] + 0.5 * sx, prev[1] + 0.5 * sy, marker="X", color=INK, ms=10, mec="white", zorder=7)
        else:
            ax.annotate("", xy=s_["pos"], xytext=prev, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=TRAIL_C, lw=lw, mutation_scale=12, shrinkA=0, shrinkB=4))
            prev = s_["pos"]


def plot_trial(log, i=0, fs=11, figsize=(14, 13)):
    """One apple: the board with the route, then (one time axis) the input, the ring,
    the decision neurons, the counters' spikes and the counters' voltage."""
    plt = _mpl()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    lk = log.trials[i]
    rec = lk["rec"]
    tm, T = rec["timing"], rec["T"]
    meta = log.meta
    ang = log.angles_deg
    n = len(ang)
    path = "".join(DIR_NAMES[s_["move"]] for s_ in lk["steps"]) or "–"
    forget = meta["pre_amp"] != 0
    with plt.rc_context({"font.size": fs}):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(6, 2, height_ratios=[3.0, 0.8, 1.3, 0.85, 0.5, 1.05], width_ratios=[1, 1.35],
                              left=0.08, right=0.97, top=0.93, bottom=0.09, hspace=0.16, wspace=0.08)
        ax_b = fig.add_subplot(gs[0, 0]); ax_t = fig.add_subplot(gs[0, 1]); ax_t.axis("off")
        ax_in = fig.add_subplot(gs[1, :]); ax_e = fig.add_subplot(gs[2, :], sharex=ax_in)
        ax_d = fig.add_subplot(gs[3, :], sharex=ax_in); ax_z = fig.add_subplot(gs[4, :], sharex=ax_in)
        ax_v = fig.add_subplot(gs[5, :], sharex=ax_in)
        draw_board(ax_b, log.size, lk["pos"], lk["apple"], fs=fs, title="The route, walked from memory")
        draw_route(ax_b, lk["pos"], lk["steps"])
        # facts
        lines = [f"network: {log.net_type} ({cond_name(log.rf, log.forget)})",
                 f"apple at {lk['theta']:.0f}° (compass), {lk['n_axis'][0]} across, {lk['n_axis'][1]} up/down",
                 f"RF centre (the ring's own decoding): {lk['rf_center']:.0f}°",
                 f"counters set at the look: E/W {lk['enc_start'][0]:.3f}, N/S {lk['enc_start'][1]:.3f}",
                 f"walked: {path}   →   {lk['end']}"]
        ax_t.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=fs, transform=ax_t.transAxes,
                  linespacing=1.6)
        h = [Patch(color=FORGET_C, alpha=0.35, label="forget: −3 pulse, w0 = +1"),
             Patch(color=CUE_BG, label="cue: w0 = −1 on every E neuron"),
             Patch(color=RF_BG, label="RF: w0 = −1 kept (the memory)"),
             Line2D([], [], color=APPLE_C, lw=2, label="apple direction"),
             Line2D([], [], color="#08519c", lw=1.5, ls="--", label="RF centre"),
             Patch(color=CLOSED_BG, label="decision blocked by its counter"),
             Line2D([], [], marker="o", ls="", color=EXC_C, label="cue spikes"),
             Line2D([], [], color=ENC_C, lw=2, label="counter")]
        ax_t.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, frameon=False, fontsize=fs - 1)
        # common background
        for ax in (ax_in, ax_e, ax_d, ax_z, ax_v):
            if forget:
                ax.axvspan(0, tm.cue_on, color=FORGET_C, alpha=0.25, lw=0, zorder=0)
            for t_ in (tm.cue_on, tm.cue_off):
                ax.axvline(t_, color=INK, lw=0.7, ls=":")
            for sl in rec["slots"]:
                ax.axvline(sl["t1"], color=MUTED, lw=0.6, alpha=0.5)
            ax.set_xlim(0, T)
            _style(ax)
        # input
        tt, nn = np.nonzero(rec["stim_spk"][0])
        ax_in.scatter(tt, ang[nn], s=6, color=EXC_C, lw=0, rasterized=True, zorder=3)
        # ring: w0 background, spikes, apple and RF lines
        if meta["e_model"] == "bistable":
            cat = np.zeros((T, n))
            cat[tm.cue_on:tm.cue_off] = 1
            cat[tm.cue_off:] = np.where(np.asarray(rec["w0_map"][0]) == meta["w0_mem"], 2, 0)
            ax_e.imshow(cat.T, aspect="auto", origin="lower", cmap=ListedColormap([(1, 1, 1, 0), CUE_BG, RF_BG]),
                        vmin=-0.5, vmax=2.5, extent=(0, T, -11.25, 348.75), interpolation="nearest", zorder=0)
        tt, nn = np.nonzero(rec["spk_e"][0])
        ax_e.scatter(tt, ang[nn], s=9, color=INK, lw=0, rasterized=True, zorder=2)
        th = lk["theta"]
        ax_e.plot([tm.cue_on, T], [th, th], color=APPLE_C, lw=1.6, zorder=3)
        ax_e.plot([tm.cue_off, T], [lk["rf_center"]] * 2, color="#08519c", lw=1.3, ls="--", zorder=3)
        for ax in (ax_in, ax_e):
            ax.set_ylim(-11.25, 348.75)
            ax.set_yticks([0, 90, 180, 270]); ax.set_yticklabels(["E", "N", "W", "S"], fontsize=fs - 2)
        # decision neurons (rows E, N, W, S); gray = blocked by its counter in that slot
        spk_d = rec["spk_d"][0]
        for k in range(4):
            ax_d.vlines(np.flatnonzero(spk_d[:, k]), k - 0.35, k + 0.35, color=INK, lw=1.3, zorder=2)
        steps = iter(lk["steps"])
        for sl, slg in zip(rec["slots"], lk["slot_log"]):
            for k in range(4):
                if slg["closed"][k]:
                    ax_d.fill_between([sl["t0"], sl["t1"]], k - 0.45, k + 0.45, color=CLOSED_BG, lw=0, zorder=0)
            letter = DIR_NAMES[next(steps)["move"]] if slg["step"] else "·"
            ax_d.text((sl["t0"] + sl["t1"]) / 2, 3.6, letter, ha="center", va="bottom", fontsize=fs, fontweight="bold",
                      color=INK if letter != "·" else MUTED)
        ax_d.set_ylim(-0.6, 4.4); ax_d.set_yticks(range(4)); ax_d.set_yticklabels(DIR_NAMES, fontsize=fs - 1)
        # counters: spikes and voltage (from the moment they are set, at the end of the cue)
        enc_t = rec.get("enc_t", tm.cue_off)
        for j in range(2):
            ax_z.vlines(np.flatnonzero(rec["spk_z"][0, :, j]), 1 - j - 0.35, 1 - j + 0.35, color=ENC_C, lw=1.0)
        ax_z.set_ylim(-0.6, 1.6); ax_z.set_yticks([1, 0]); ax_z.set_yticklabels(AXIS_NAMES, fontsize=fs - 1)
        if rec.get("v_z") is not None:
            t_ax = np.arange(enc_t, T)
            for j, ls in ((0, "-"), (1, "--")):
                ax_v.plot(t_ax, rec["v_z"][0, enc_t:, j], color=ENC_C, lw=1.6, ls=ls,
                          label=f"{AXIS_NAMES[j]} counter ({lk['n_axis'][j]} steps)")
            ax_v.axhline(1.0, color=INK, lw=0.8, ls=":")
            ax_v.text(T, 1.0, " threshold", va="center", ha="left", fontsize=fs - 2, color=INK)
            ax_v.text(enc_t / 2, 0.55, "set at the end\nof the cue", ha="center", va="center", fontsize=fs - 2,
                      color=MUTED)
            ax_v.set_ylim(-0.1, 1.35)
            ax_v.legend(loc="upper left", fontsize=fs - 2, frameon=False, ncol=2)
        ax_in.set_ylabel("input", fontsize=fs)
        ax_e.set_ylabel("E neuron", fontsize=fs)
        ax_d.set_ylabel("decision", fontsize=fs)
        ax_z.set_ylabel("counter\nspikes", fontsize=fs - 1, color=ENC_C)
        ax_v.set_ylabel("counter\nvoltage", fontsize=fs - 1, color=ENC_C)
        ax_v.set_xlabel("time in the look (ms)", fontsize=fs)
        for ax in (ax_in, ax_e, ax_d, ax_z):
            plt.setp(ax.get_xticklabels(), visible=False)
        fig.suptitle(f"One apple: {log.net_type}", x=0.08, y=0.985, ha="left", va="top", fontsize=fs + 2)
    return fig


def plot_comparison(base, fs=11, figsize=(19, 5.2)):
    """The three networks: apples eaten, spikes per apple by source, apples per 10,000 spikes."""
    plt = _mpl()
    st = pooled(base, "net_type")
    order = [t for t in TYPE_ORDER if t in st.index]
    with plt.rc_context({"font.size": fs}):
        fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw=dict(width_ratios=[0.8, 1.6, 0.9], wspace=0.4))
        ax = axes[0]
        for i, t in enumerate(order):
            d = 100 * base[base.net_type == t]["hit_rate"].values
            ax.bar(i, d.mean(), color=TYPE_COLORS[t], width=0.6)
            ax.scatter(i + np.linspace(-0.18, 0.18, len(d)), d, s=12, color=INK, alpha=0.6, zorder=3, lw=0)
            ax.text(i, d.max() + 2, f"{d.mean():.0f}%", ha="center", va="bottom", fontsize=fs - 1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([t.replace(" ", "\n") for t in order], fontsize=fs - 1)
        ax.set_ylim(0, 110); ax.set_title("apples eaten (% of shown)", loc="left", fontsize=fs)
        _style(ax)
        ax = axes[1]
        y = np.arange(len(order))[::-1]
        totals = {t: float(st.loc[t, "spikes_per_apple"]) for t in order}
        ref = [v for t, v in totals.items() if np.isfinite(v) and t != "all LIF"] or \
              [v for v in totals.values() if np.isfinite(v)] or [1.0]
        xmax = 1.6 * max(ref)
        for yy, t in zip(y, order):
            left = 0.0
            if st.loc[t, "apples"] > 0:
                for lab, c, col in SOURCES:
                    v = float(st.loc[t, c] / st.loc[t, "apples"])
                    ax.barh(yy, v, left=left, height=0.6, color=col, lw=0, label=lab if yy == y[0] else None)
                    left += v
            if not np.isfinite(totals[t]):
                ax.text(0.02 * xmax, yy, f"no apple eaten ({st.loc[t, 'spikes_per_apple_shown']:,.0f} spikes per apple "
                        "shown)", va="center", fontsize=fs - 1)
            elif left > xmax:
                ax.text(0.98 * xmax, yy, f"{left:,.0f} →", va="center", ha="right", fontsize=fs - 1, color="white",
                        fontweight="bold")
            else:
                ax.text(left + 0.01 * xmax, yy, f"{left:,.0f}", va="center", fontsize=fs - 1)
        ax.set_yticks(y); ax.set_yticklabels(order)
        ax.set_xlim(0, xmax); ax.set_xlabel("spikes per apple eaten (longer bars are cut)")
        ax.set_title("spikes per apple eaten, by source", loc="left", fontsize=fs)
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=4, frameon=False, fontsize=fs - 1)
        _style(ax)
        ax = axes[2]
        for i, t in enumerate(order):
            v = float(st.loc[t, "apples_per_10k"])
            d = base[base.net_type == t]["apples_per_10k"].values
            ax.bar(i, v, color=TYPE_COLORS[t], width=0.6)
            ax.scatter(i + np.linspace(-0.18, 0.18, len(d)), d, s=12, color=INK, alpha=0.6, zorder=3, lw=0)
            ax.text(i, max(v, np.nanmax(d) if len(d) else v) * 1.03, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=fs - 1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([t.replace(" ", "\n") for t in order], fontsize=fs - 1)
        ax.set_title("apples per 10,000 spikes", loc="left", fontsize=fs)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        _style(ax)
    return fig


def plot_noise(df, col, types=None, fs=11, figsize=(18, 4.8), xlabel=None):
    """Noise sweep (col = 'cv_input' or 'cv_weight'): apples per 10,000 spikes, apples eaten,
    spikes per apple eaten. Error bars: s.d. over runs."""
    plt = _mpl()
    types = [t for t in (types or TYPE_ORDER) if t in set(df.net_type)]
    df = df[df.net_type.isin(types)]
    st = pooled(df, ["net_type", col])
    g = df.groupby(["net_type", col])
    panels = [("apples per 10,000 spikes", st["apples_per_10k"], st["apples_per_10k_sd"]),
              ("apples eaten (% of shown)", 100 * g["hit_rate"].mean(), 100 * g["hit_rate"].std()),
              ("spikes per apple eaten", st["spikes_per_apple"], None)]
    with plt.rc_context({"font.size": fs}):
        fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw=dict(wspace=0.3))
        for ax, (lab, vals, err) in zip(axes, panels):
            for t in types:
                v = vals.loc[t]
                x = 100 * np.asarray(v.index, float)
                if err is not None:
                    ax.errorbar(x, v.values, yerr=err.loc[t].values, color=TYPE_COLORS[t], marker="o", lw=2, capsize=3,
                                label=t)
                else:
                    ax.plot(x, v.values, color=TYPE_COLORS[t], marker="o", lw=2, label=t)
            ax.set_xticks([0, 10, 20, 30, 40, 50]); ax.set_xlabel(xlabel or f"{col.replace('_', ' ')} (%)")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
            ax.set_title(lab, loc="left", fontsize=fs)
            _style(ax)
        axes[0].legend(loc="upper center", bbox_to_anchor=(1.7, -0.2), ncol=3, frameon=False, fontsize=fs)
    return fig


def _bars(ax, groups, series, values, colors, fmt="{:.1f}", fs=10, err=None, na_text="no apple eaten"):
    w = 0.8 / len(series)
    top = 1e-9
    for j, s in enumerate(series):
        for i, g in enumerate(groups):
            x = i + (j - (len(series) - 1) / 2) * w
            v = values.get((g, s), np.nan)
            if not np.isfinite(v):
                if (g, s) in values and na_text:
                    ax.text(x, 0, " " + na_text, rotation=90, ha="center", va="bottom", fontsize=fs - 3, color=MUTED)
                continue
            e = err.get((g, s)) if err else None
            e = float(e) if e is not None and np.isfinite(e) and e > 0 else 0.0
            ax.bar(x, v, width=w * 0.9, color=colors[s], yerr=e or None, error_kw=dict(ecolor=INK, lw=0.8, capsize=2),
                   label=s if i == 0 else None)
            top = max(top, v + e)
            ax.text(x, v + e, " " + fmt.format(v), ha="center", va="bottom", fontsize=fs - 3)
    ax.set_xticks(range(len(groups)))
    return top


def plot_rf_forget(rfg, fs=11, figsize=(18, 5.0)):
    """The two bistable rings in the four RF x forget conditions: apples eaten, spikes per
    apple eaten, apples per 10,000 spikes."""
    plt = _mpl()
    rfg = rfg.assign(condition=[cond_name(str(a) == "True", str(b) == "True") for a, b in zip(rfg["rf"], rfg["forget"])])
    types = [t for t in BISTABLE_RINGS if t in set(rfg.net_type)]
    conds = [c for c in COND_ORDER if c in set(rfg.condition)]
    st = pooled(rfg, ["net_type", "condition"])
    g = rfg.groupby(["net_type", "condition"])
    panels = [("apples eaten (% of shown)", (100 * g["hit_rate"].mean()).to_dict(), (100 * g["hit_rate"].std()).to_dict(),
               "{:.0f}"),
              ("spikes per apple eaten", st["spikes_per_apple"].to_dict(), None, "{:.0f}"),
              ("apples per 10,000 spikes", st["apples_per_10k"].to_dict(), st["apples_per_10k_sd"].to_dict(), "{:.1f}")]
    with plt.rc_context({"font.size": fs}):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, gridspec_kw=dict(wspace=0.3))
        for ax, (lab, vals, err, fmt) in zip(axes, panels):
            top = _bars(ax, types, conds, vals, COND_COLORS, fmt, fs, err=err)
            ax.set_xticklabels(types, fontsize=fs - 1)
            ax.set_ylim(0, top * 1.2)
            ax.set_title(lab, loc="left", fontsize=fs)
            _style(ax)
        axes[0].legend(loc="upper center", bbox_to_anchor=(1.75, -0.12), ncol=4, frameon=False, fontsize=fs)
    return fig


def animate_runs(logs, path=None, n_apples=None, fps=4, fs=11, dpi=90, hold=2):
    """Many apples at a glance, one board per run side by side (e.g. the three networks).
    Per apple: a look frame (apple filled), one frame per step (apple dashed: the agent
    cannot see it; blue arrow = the RF, the direction the ring holds), and end frames.
    path: .gif, or .mp4 (needs ffmpeg)."""
    plt = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from matplotlib.patches import Circle
    logs = list(logs.values()) if isinstance(logs, dict) else list(logs)
    seqs = []
    for lg in logs:
        fr = []
        for i, lk in enumerate(lg.trials[:n_apples]):
            fr.append((i, "look", 0))
            fr += [(i, "step", k) for k in range(1, len(lk["steps"]) + 1)]
            fr += [(i, "end", len(lk["steps"]))] * hold
        seqs.append(fr)
    fig, axes = plt.subplots(1, len(logs), figsize=(5.6 * len(logs), 6.4), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.12, wspace=0.12)
    fig.text(0.03, 0.975, "Each apple is shown once (filled). Then the agent walks from memory (dashed apple: it cannot "
             "see it).\nBlue arrow: the direction the ring holds (the RF).", ha="left", va="top", fontsize=fs)
    lost = {"memory": "lost: no decision", "wall": "lost: wall", "route done": "lost: wrong side",
            "timeout": "lost: timeout"}

    def panel(ax, lg, fr):
        i, ph, k = fr
        lk = lg.trials[i]
        pos = lk["pos"] if k == 0 else lk["steps"][k - 1]["pos"]
        eaten = ph == "end" and lk["end"] == "apple"
        draw_board(ax, lg.size, pos, lk["apple"], trail=[lk["pos"]] + [s_["pos"] for s_ in lk["steps"][:k]],
                   apple_on=ph == "look", fs=fs)
        draw_route(ax, lk["pos"], lk["steps"][:k], lw=2.0)
        if ph != "look" and not eaten:
            ax.add_patch(Circle(lk["apple"], 0.36, fill=False, ec=APPLE_C, lw=1.8, ls="--", zorder=3))
        if ph == "step" or (ph == "end" and not eaten):
            a = np.deg2rad(lk["rf_center"])
            ax.annotate("", xy=(pos[0] + 1.3 * np.cos(a), pos[1] + 1.3 * np.sin(a)), xytext=pos, zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color="#4a90c2", lw=2.4, mutation_scale=14))
        got = sum(1 for l in lg.trials[:i] if l["end"] == "apple") + int(eaten)
        state = {"look": "look: the apple is shown", "step": f"step {k} of {lk['shortest']} needed",
                 "end": "eaten" if eaten else lost.get(lk["end"], lk["end"])}[ph]
        ax.set_title(f"{lg.net_type}, {cond_name(lg.rf, lg.forget)}\napple {i + 1}:  {state}", loc="left", fontsize=fs,
                     color=APPLE_C if eaten else (INH_C if ph == "end" else INK))
        ax.set_xlabel(f"apples eaten {got} of {i + 1}", fontsize=fs)

    def draw(f):
        for ax, lg, fr in zip(axes, logs, seqs):
            ax.cla()
            panel(ax, lg, fr[min(f, len(fr) - 1)])
        return []

    anim = FuncAnimation(fig, draw, frames=max(len(f) for f in seqs), interval=1000 / fps, blit=False)
    if path is not None:
        anim.save(path, writer=FFMpegWriter(fps=fps) if str(path).endswith(".mp4") else PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
    return anim


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run (or resume) every navigation experiment.")
    ap.add_argument("--out", default=None, help=f"default {OUT_DIR}/runs.csv")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    run_all(a.out, seeds=range(a.seeds), n_proc=a.procs)
