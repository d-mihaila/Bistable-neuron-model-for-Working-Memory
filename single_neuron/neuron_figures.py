"""
Two paper-ready figures for the bistable neuron (numpy + matplotlib only):

    plot_phase_planes(w0)    1x3 phase planes: I_ext < 0, = 0, > 0
    plot_pulse_response(w0)  I_ext(t) and v(t) for a negative then a positive pulse

Design goals: easy to read at a glance. Big text, few labels, direct labelling
instead of legends, one colour code shared by both figures:
    purple = spiking     green = rest     blue = the calcium parameter w0

Time is in milliseconds. The neuron is advanced with ONE RK4 step per ms, exactly
like the networks (1 network step = 1 ms, see MNIST / N_MNIST). The only place
the model's own time unit appears is TIME_PER_MS below.

Model (Eqs. 1-3 of the paper; parameters in params.py):
    dv/dt = v^2 - w^2 + I_ext + I_TC          I_TC = w0^2 / (1 - a^2)
    dw/dt = eps * (a v - w + w0)
    if v >= v_th:  v <- c,  w <- d
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from params import a, eps, c, d, v_th

# ── one colour code for both figures ─────────────────────────────────────────
C_SPIKE = '#aa3377'   # purple: spiking
C_REST  = '#228833'   # green : rest
C_W0    = '#0057c8'   # blue  : the w0 parameter (same blue as in the equations)
C_VNULL = '#111111'   # v-nullcline
C_WNULL = '#d45c1e'   # w-nullcline
C_RESET = '#e8a000'   # reset point (c, d)
C_CYCLE = '#e41a1c'   # limit cycle (red)
C_TRAJ  = '#5ab4f0'   # light blue: example trajectories that end spiking (phase planes)
C_SADDLE = '#6a6a6a'  # saddle marker (grey, so it is not confused with the red limit cycle)

TIME_PER_MS = 0.025   # model time units per ms (1 network step = 1 ms)


def I_TC(w0):
    """Critical current: the input at which the rest state and the saddle merge."""
    return w0**2 / (1 - a**2)


def _style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 1.0, 'pdf.fonttype': 42,   # editable text in PDF
    })


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: phase planes
# ═════════════════════════════════════════════════════════════════════════════

def _rhs(v, w, w0, I_eff):
    return v**2 - w**2 + I_eff, eps * (a * v - w + w0)


def _trajectory(v0, w0_, w0, I_eff, t_max=80.0, dt=0.01, v_escape=6.0):
    """Integrate the smooth 2-D system (no reset). Stops when v escapes (= a spike)."""
    v, w = v0, w0_
    vs, ws = [v], [w]
    for _ in range(int(t_max / dt)):
        k1 = _rhs(v, w, w0, I_eff)
        k2 = _rhs(v + dt/2*k1[0], w + dt/2*k1[1], w0, I_eff)
        k3 = _rhs(v + dt/2*k2[0], w + dt/2*k2[1], w0, I_eff)
        k4 = _rhs(v + dt*k3[0],   w + dt*k3[1],   w0, I_eff)
        v += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        w += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        vs.append(v); ws.append(w)
        if v > v_escape:
            return np.array(vs), np.array(ws), True
    return np.array(vs), np.array(ws), False


def _fixed_points(w0, I_eff):
    """Intersections of the nullclines w = +-sqrt(v^2 + I_eff) and w = a v + w0."""
    A, B, C = a**2 - 1, 2 * a * w0, w0**2 - I_eff
    disc = B**2 - 4 * A * C
    if abs(disc) < 1e-9:
        v = -B / (2 * A)
        return [(v, a * v + w0, 'critical')]
    if disc < 0:
        return []
    out = []
    for v in ((-B + np.sqrt(disc)) / (2 * A), (-B - np.sqrt(disc)) / (2 * A)):
        w = a * v + w0
        det = 2 * eps * (a * w - v)
        out.append((v, w, 'saddle' if det < 0 else 'stable'))
    return out



def _limit_cycle(w0, I_eff, dt=0.005, t_max=300.0):
    """
    The spiking orbit of the full neuron (smooth flow + reset to (c, d) at v = v_th).
    Returns (path, jump) for the last full cycle, or None if the neuron settles to rest.
    path = points from the reset to the threshold; jump = (threshold point, reset point).
    """
    v, w = c, d
    spikes, path, cur = 0, [], [(v, w)]
    last_cycle = None
    for _ in range(int(t_max / dt)):
        k1 = _rhs(v, w, w0, I_eff)
        k2 = _rhs(v + dt/2*k1[0], w + dt/2*k1[1], w0, I_eff)
        k3 = _rhs(v + dt/2*k2[0], w + dt/2*k2[1], w0, I_eff)
        k4 = _rhs(v + dt*k3[0],   w + dt*k3[1],   w0, I_eff)
        v += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        w += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        cur.append((v, w))
        if v >= v_th:
            spikes += 1
            last_cycle = (np.array(cur), ((v, w), (c, d)))
            v, w = c, d
            cur = [(v, w)]
    if spikes < 6:          # fires a few times, then rests: no limit cycle
        return None
    return last_cycle

# start points of the example trajectories (same in every panel)
_STARTS = [(-3.0, -0.6), (-2.2, 1.8), (-0.3, -2.2), (1.4, -1.4), (2.4, 1.8), (0.8, 2.8)]


def _phase_panel(ax, w0, I_ext, lim=3.5, starts=None):
    I_eff = I_ext + I_TC(w0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')

    # light flow field for direction
    g = np.linspace(-lim, lim, 40)
    V, W = np.meshgrid(g, g)
    DV, DW = _rhs(V, W, w0, I_eff)
    ax.streamplot(V, W, DV, DW, color='#d9d9d9', density=0.9, linewidth=0.7,
                  arrowsize=0.8, zorder=1)

    # nullclines
    v = np.linspace(-lim, lim, 600)
    inside = v**2 + I_eff
    root = np.where(inside >= 0, np.sqrt(np.abs(inside)), np.nan)
    for sgn in (+1, -1):
        ax.plot(v, sgn * root, color=C_VNULL, lw=2.6, zorder=3, solid_capstyle='round')
    ax.plot(v, a * v + w0, color=C_WNULL, lw=2.6, ls=(0, (4, 2)), zorder=3)

    # trajectories, coloured by where they end up
    for (v0, w00) in (_STARTS if starts is None else starts):
        if v0 >= v_th:          # initial v must be below the spike threshold (v_th = 1)
            continue
        vs, ws, spikes = _trajectory(v0, w00, w0, I_eff)
        col = C_TRAJ if spikes else C_REST
        m = (np.abs(vs) <= lim) & (np.abs(ws) <= lim)
        ax.plot(vs[m], ws[m], color=col, lw=1.8, alpha=0.9, zorder=4)
        ax.plot(v0, w00, 'o', color=col, ms=5, mec='white', mew=0.8, zorder=5)
        j = np.argmax(np.cumsum(np.hypot(np.diff(vs[m]), np.diff(ws[m]))) > 0.8) if m.sum() > 3 else 0
        if 0 < j < m.sum() - 1:                       # one arrowhead per trajectory
            vm, wm = vs[m], ws[m]
            ax.annotate('', xy=(vm[j + 1], wm[j + 1]), xytext=(vm[j], wm[j]),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=0, mutation_scale=22),
                        zorder=5)

    # limit cycle of the full neuron (with the spike reset), if there is one
    cyc = _limit_cycle(w0, I_eff)
    ax.axvline(v_th, color='#999999', lw=1.0, ls=':', zorder=2)          # spike threshold
    if cyc is not None:
        path, ((v1, w1), (v2, w2)) = cyc
        ax.plot(path[:, 0], path[:, 1], color=C_CYCLE, lw=4.2, solid_capstyle='round',
                zorder=6)
        ax.annotate('', xy=(v2, w2), xytext=(v1, w1),                      # the reset jump
                    arrowprops=dict(arrowstyle='-|>', color=C_CYCLE, lw=1.8, ls=(0, (2, 2)),
                                    mutation_scale=12, shrinkA=0, shrinkB=3), zorder=6)

    # reset point (c, d)
    ax.plot(c, d, 's', color=C_RESET, ms=8, mec='black', mew=0.8, zorder=7)

    # fixed points
    for (fv, fw, kind) in _fixed_points(w0, I_eff):
        if kind == 'stable':
            ax.plot(fv, fw, 'o', color='black', ms=11, mec='white', mew=1.2, zorder=8)
        elif kind == 'saddle':
            ax.plot(fv, fw, 'X', color=C_SADDLE, ms=12, mec='white', mew=1.0, zorder=8)
        else:   # critical point: stable on one side, unstable on the other
            ax.plot(fv, fw, 'o', color='black', ms=11, fillstyle='left', mfc='black',
                    mfcalt='white', mec='black', mew=1.5, zorder=8)

    tk = [-round(lim * 0.85), 0, round(lim * 0.85)]
    ax.set_xticks(tk); ax.set_yticks(tk)
    return I_eff


def plot_phase_planes(w0=-0.5, I_amp=1.0, lim=3.5, figsize=(7.2, 3.3), savefile=None):
    """Three phase planes: negative, zero, positive input. Purple = ends spiking, green = ends at rest.
    The dark loop is the limit cycle of the full neuron (it includes the spike reset).
    lim = half-width of the window; smaller zooms in on the limit cycle (try 2), larger zooms out."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    cases = [(-I_amp, 'Negative input', r'$I_{ext}<0$', 'rest state appears'),
             (0.0,    'No input',       r'$I_{ext}=0$', 'critical point'),
             (+I_amp, 'Positive input', r'$I_{ext}>0$', 'rest state lost')]
    for ax, (I_ext, title, sub, verdict) in zip(axes, cases):
        _phase_panel(ax, w0, I_ext, lim=lim)
        ax.set_title(f'{title}\n{sub}', pad=6, linespacing=1.3)
        ax.set_xlabel(r'$v$  (voltage)')
        ax.text(0.5, -0.38, verdict, transform=ax.transAxes, ha='center', va='top',
                fontsize=11.5, style='italic', fontweight='bold')
    axes[0].set_ylabel(r'$w$  (recovery)')

    # direct labels (no legend box)
    axes[0].text(0.04, 0.04, rf'$w_0={w0:g}$', transform=axes[0].transAxes, color=C_W0,
                 fontsize=12, fontweight='bold', ha='left', va='bottom')

    # tiny key: what the colours mean
    key = [Line2D([], [], color=C_VNULL, lw=2.6, label=r'$v$ nullcline ($\dot v=0$)'),
           Line2D([], [], color=C_WNULL, lw=2.6, ls=(0, (4, 2)), label=r'$w$ nullcline ($\dot w=0$)'),
           Line2D([], [], color='#999999', lw=1.4, ls=':', label='spike threshold $v_{th}$'),
           Line2D([], [], color=C_CYCLE, lw=3.6, label='limit cycle'),
           Line2D([], [], color=C_TRAJ, lw=2.5, label='spike'),
           Line2D([], [], color=C_REST, lw=2.5, label='at rest'),
           Line2D([], [], marker='o', color='w', mfc='black', ms=9, label='stable point'),
           Line2D([], [], marker='X', color='w', mfc=C_SADDLE, ms=9, label='saddle'),
           Line2D([], [], marker='o', color='w', ms=9, fillstyle='left', mfc='black',
                  mfcalt='white', mec='black', mew=1.2, label='critical (half-stable)'),
           Line2D([], [], marker='s', color='w', mfc=C_RESET, mec='black', ms=8, label='reset point')]
    fig.legend(handles=key, loc='lower center', ncol=3, frameon=False, fontsize=10.5,
               handlelength=1.3, columnspacing=1.6, handletextpad=0.5,
               bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.46, wspace=0.12)

    if savefile:
        fig.savefig(savefile, dpi=300, bbox_inches='tight')
    return fig, axes


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: response to a negative and a positive pulse
# ═════════════════════════════════════════════════════════════════════════════

def simulate_pulses(w0=-0.5, t_start=500, pulse_ms=200, gap_ms=1500, t_end=3500, amp=1.0):
    """
    Negative pulse, silence, positive pulse. All times in ms, one RK4 step per ms
    (same update and reset as the network neurons).
    Returns dict with t (ms), I, v, w, spike_times (ms).
    """
    n = int(t_end)
    t = np.arange(n)
    I = np.zeros(n)
    neg_on, neg_off = t_start, t_start + pulse_ms
    pos_on, pos_off = neg_off + gap_ms, neg_off + gap_ms + pulse_ms
    I[neg_on:neg_off] = -amp
    I[pos_on:pos_off] = +amp

    h = TIME_PER_MS
    itc = I_TC(w0)
    f = lambda v, w, Ii: (v**2 - w**2 + Ii + itc, eps * (a * v - w + w0))
    v = w = 0.0
    V, Wt, spikes = np.zeros(n), np.zeros(n), []
    for k in range(n):
        k1 = f(v, w, I[k])
        k2 = f(v + h/2*k1[0], w + h/2*k1[1], I[k])
        k3 = f(v + h/2*k2[0], w + h/2*k2[1], I[k])
        k4 = f(v + h*k3[0],   w + h*k3[1],   I[k])
        v += h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        w += h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        if v >= v_th:
            spikes.append(k)
            V[k], Wt[k] = v, w          # record the peak, then reset
            v, w = c, d
        else:
            V[k], Wt[k] = v, w
    return dict(t=t, I=I, v=V, w=Wt, spike_times=np.array(spikes),
                neg=(neg_on, neg_off), pos=(pos_on, pos_off), amp=amp)


def plot_pulse_response(w0=-0.5, figsize=(7.2, 2.9), show_w=False, savefile=None, **protocol):
    """Input pulses on top, voltage below. Shading shows the state that persists after each pulse."""
    _style()
    sim = simulate_pulses(w0=w0, **protocol)
    t, I, v, st = sim['t'], sim['I'], sim['v'], sim['spike_times']
    n0, n1 = sim['neg']; p0, p1 = sim['pos']

    # the neuron must really stay silent between the pulses, otherwise the picture would lie
    assert not np.any((st > n0 + 150) & (st < p0)), 'neuron fired during the silent stretch'
    assert np.any(st > p1) and np.any(st < n0), 'neuron should spike before and after'

    rows = 3 if show_w else 2
    ratios = [0.75, 2.2] + ([1.3] if show_w else [])
    fig, axes = plt.subplots(rows, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.08})
    ax_I, ax_v = axes[0], axes[1]

    # state shading, identical on every row: spiking (purple) vs rest (green)
    for ax in axes:
        ax.axvspan(t[0], n0, color=C_SPIKE, alpha=0.10, lw=0)
        ax.axvspan(n0, p0, color=C_REST, alpha=0.18, lw=0)
        ax.axvspan(p0, t[-1], color=C_SPIKE, alpha=0.10, lw=0)

    # input
    ax_I.plot(t, I, color='#333333', lw=2.2, drawstyle='steps-post')
    ax_I.set_yticks([-1, 0, 1]); ax_I.set_yticklabels(['\u22121', '0', '+1'])
    ax_I.set_ylim(-1.5, 1.5)
    ax_I.set_ylabel(r'$I_{ext}$')
    ax_I.text(n1 + 50, -1.0, 'negative pulse', ha='left', va='center', fontsize=11.5)
    ax_I.text(p1 + 50, 1.0, 'positive pulse', ha='left', va='center', fontsize=11.5)
    ax_I.text(0.005, 1.10, rf'$w_0={w0:g}$', transform=ax_I.transAxes, color=C_W0,
              fontsize=12, fontweight='bold', ha='left', va='bottom')

    # voltage with spikes; the state names sit above the spikes
    ax_v.plot(t, v, color='#1f1f1f', lw=1.5)
    ax_v.plot(st, v[st], 'o', color='black', ms=3.5)
    ax_v.set_ylim(-1.1, 1.9)
    ax_v.set_yticks([-1, 0, 1]); ax_v.set_yticklabels(['\u22121', '0', '1'])
    ax_v.set_ylabel(r'$v$  (voltage)')
    y_lab = 1.62
    ax_v.text((t[0] + n0) / 2, y_lab, 'persistent\nspiking', ha='center', va='center',
              color=C_SPIKE, fontsize=11.5, fontweight='bold', linespacing=1.05)
    ax_v.text((n0 + p0) / 2, y_lab, 'persistent rest', ha='center', va='center',
              color=C_REST, fontsize=11.5, fontweight='bold')
    ax_v.text((p0 + t[-1]) / 2, y_lab, 'persistent spiking', ha='center', va='center',
              color=C_SPIKE, fontsize=11.5, fontweight='bold')

    if show_w:
        ax_w = axes[2]
        ax_w.plot(t, sim['w'], color=C_WNULL, lw=1.6)
        ax_w.set_ylabel(r'$w$  (recovery)')
        ax_w.set_yticks([-0.5, 0, 0.5, 1])

    axes[-1].set_xlabel('time (ms)')
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False, bottom=False)
    axes[-1].set_xlim(t[0], t[-1])

    if savefile:
        fig.savefig(savefile, dpi=300, bbox_inches='tight')
    return fig, axes, sim
