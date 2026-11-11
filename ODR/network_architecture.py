"""
network_architecture.py (ODR)

A ring-attractor E/I spiking network for the oculomotor delayed-response
(ODR) task, built on Compte, Brunel, Goldman-Rakic & Wang (2000) and its
EPFL/Brian2 reference implementation (neurodynex3.working_memory_network.
wm_model) -- reimplemented as a
discrete-time, current-based network using THIS project's own neuron
classes (neuron_model.py: LIF_syn / BistableNeuron).

Focus: the neuromorphic-computing angle of the bistable neuron -- can a
very small network of bistable
neurons still correctly encode and hold a spatial cue, under background
noise and a distractor, without any plasticity/training? So this file now
ONLY builds bistable-E networks. The E population's neuron model is no
longer a swappable argument -- it's always BistableNeuron. (The I
population is still unstructured, fixed at a 4:1 E:I ratio, exactly as
before -- interneurons were never the object of the LIF-vs-bistable
question, and aren't the object of this one either. 2026-10-22: the I
neurons are now the MNIST / N-MNIST LIF, see I_INPUT_GAIN below.)

WHAT'S THE SAME AS COMPTE 2000 / THE BRIAN2 CODE
--------------------------------------------------
  - Ring topology: N_E excitatory ("pyramidal") neurons, each with a
    preferred angle theta_i evenly spaced over [0, 360) degrees.
  - N_I inhibitory ("interneuron") neurons, unstructured, fixed 4:1 E:I
    ratio at every network size tested (Compte's own ratio).
  - The E->E "Mexican hat" weight kernel -- exact same formula as
    presyn_weight_kernel() in the uploaded file: a Gaussian bump of width
    sigma_weight_profile around Delta-theta=0, riding on a slightly
    negative baseline (Jneg), normalized so the kernel averages to 1 over
    the ring. Local excitation, i.e. neurons with similar preferred angle
    excite each other most.
  - E->I, I->E, I->I are all UNSTRUCTURED / uniform -- no angular
    dependence, matching the original paper.
  - External background is a literal Poisson bombardment (n_extern_poisson
    sources x bg_rate_hz each per neuron), same mechanism as the paper's
    N_extern_poisson/poisson_firing_rate, just without AMPA kinetics.
  - Stimulus (cue) and distractor delivery: a localized drive to the E
    neurons within stim_width_deg of the target angle (task.py builds the
    RATE profile; see "INPUT IS NOW SPIKING" below for how it becomes
    current).

WHAT'S DIFFERENT, ON PURPOSE
------------------------------
  - No conductance-based AMPA/NMDA/GABA-A synapses. Current-based
    throughout (spikes times a weight, straight into the neuron's input
    current) -- same reason as always: every neuron_model.py class in this
    codebase already uses this substrate, and NMDA's slow kinetics are
    deliberately excluded (the bistable neuron's own slow recovery
    variable, not a slow synapse, is what's meant to carry persistence
    here).
  - THE RECURRENT WEIGHTS ARE NOT TRAINED. W_EE/W_EI/W_IE/W_II are
    hand-built once from the formulas above and never touched by a
    learning rule or an optimizer, at any network size. No plasticity
    anywhere in this file -- persistence has to come from the bistable
    neuron's own dynamics and the fixed connectivity, or not at all.
  - INPUT IS SPIKING, not a raw injected current. The cue and distractor used to be a smooth,
    deterministic current added straight into the neuron's input every
    step. Now task.py builds a RATE profile instead (Hz, not current) --
    the same spatial Gaussian shape as before, just reinterpreted as "how
    fast should external spikes be arriving here" rather than "how much
    current to inject here". Every step, RingSNN.run() draws an actual
    Poisson spike count from that rate (exactly the same machinery already
    used for the background noise -- see _poisson_drive), and converts
    that count to current the same way a background spike is (count x a
    fixed weight-per-spike, stim_weight_e). So the cue, the distractor, AND
    the background noise are now all literally the same kind of thing: a
    Poisson-driven barrage of unitary spikes hitting the E population, at
    different rates in different places and times. This means the cue is
    now genuinely stochastic (irregular from trial to trial, like a real
    afferent volley) rather than a perfectly smooth current, and it shows
    up as actual dots on a raster (stim_spk_e) instead of only as a current
    heatmap.
  - Background drive: n_extern_poisson=1000 sources at bg_rate_hz=1.4Hz
    each, exactly N_extern_poisson/poisson_firing_rate from the uploaded
    Brian2 file, AMPA-free (see _poisson_drive).
  - A network-level absolute refractory period (refrac_i, in ms == steps)
    is applied to the I population only (Compte's t_abs_refract_inhib=1ms),
    as a wrapper around LIF_syn's output -- LIF_syn's own code is never
    touched. The bistable E population has no such wrapper: its own hard
    reset + regenerative/recovery dynamics IS its refractoriness.

NETWORK SIZE
-------------
n_excit/n_inhib are free parameters now, explicitly meant to be swept small
(the neuromorphic-computing question is "how few neurons can still do
this?"). One thing worth knowing going in: sigma_deg (the Mexican-hat
tuning width) is in DEGREES, so it does NOT automatically shrink with N.
The number of E neurons actually inside one sigma-width neighborhood is
roughly n_excit * sigma_deg / 360 -- at n_excit=1024 that's ~57 neurons per
sigma, but at n_excit=64 it's only ~3.5. Very small networks can therefore
fail for a DIFFERENT reason than "not enough total drive": the connectivity
kernel itself becomes too coarsely discretized to describe a smooth bump.
That's a real, reportable limit of "how small can this get", not a bug to
tune away.

CALIBRATION NOTE
-----------------
g_ee_scale/g_ei_scale/g_ie_scale/g_ii_scale, bg_weight_e/i, and
stim_weight_e/stim_rate_hz are calibrated at the full 1024/256 size (see
run_odr_demo.py's SIZE_PRESET / DEFAULT_KWARGS) to give a clean, accurately
located, persistent bump under background noise and a distractor. Whether
the SAME weights still work unchanged at smaller sizes, or need their own
pass, is itself part of what the size sweep is testing.
"""

import math

import numpy as np
import torch

# ---------------------------------------------------------------------------
# I neurons (2026-10-22). The I population uses the MNIST / N-MNIST LIF
# (neuron_model.LIF_simple: v <- 0.5 v + I, spike when v > 1, reset to 0,
# no synaptic filter), so that every LIF in the project is the same neuron.
# Before 2026-10-22 it used LIF_syn (tau_mem 10 ms, tau_syn 5 ms); all ODR
# results up to then were made with it (RingSNN(i_model="lif_syn")).
#
# The MNIST LIF keeps half its voltage per ms and has no synaptic filter, so
# the same synapses drive it far less (a steady input gives 2x its size in
# voltage, against 58x for LIF_syn). With the old weights the I population
# never fires (0 Hz in the reference trial). One factor therefore multiplies
# EVERY synapse onto an I neuron (E->I, I->I, background->I):
#     I_INPUT_GAIN["lif"] = 14
# Rule: the I population fires as many spikes over the reference trial as it
# did with LIF_syn (RF out 0 + pulse, 500 ms cue, 8 trials: 5.0 vs 5.3 Hz).
# Synapses from I onto E, and everything else, are unchanged. Per phase the
# MNIST LIF fires more during the cue (33 vs 19 Hz) and less in the hold
# (3 vs 6 Hz): it answers to coincident input (E volleys), not to the mean.
# ---------------------------------------------------------------------------
I_INPUT_GAIN = {"lif": 14.0, "lif_syn": 1.0}


# ---------------------------------------------------------------------------
# Ring connectivity (Compte 2000's Mexican-hat kernel, ported directly)
# ---------------------------------------------------------------------------

def mexican_hat_kernel(n_excit, sigma_deg=20.0, j_pos=1.6):
    """Exact port of presyn_weight_kernel() from the Brian2 reference file.
    Returns a length-n_excit 1D array: kernel[d] is the (pre-normalization-
    checked) synaptic weight between two E neurons whose ring-index
    distance is d (in either direction -- it's symmetric). Averages to 1.0
    over the full ring by construction (see Jneg derivation)."""
    tmp = (math.sqrt(2. * math.pi) * sigma_deg
           * math.erf(180. / math.sqrt(2.) / sigma_deg) / 360.)
    j_neg = (1. - j_pos * tmp) / (1. - tmp)
    kernel = np.array([
        j_neg + (j_pos - j_neg) * math.exp(
            -0.5 * (360. * min(j, n_excit - j) / n_excit) ** 2 / sigma_deg ** 2)
        for j in range(n_excit)
    ])
    return kernel, j_neg


def build_W_EE(n_excit, sigma_deg=20.0, j_pos=1.6, g_ee_scale=1.0, device="cpu", dtype=torch.float):
    """Dense (n_excit, n_excit) circulant weight matrix from the Mexican-hat
    kernel, scaled by g_ee_scale / n_excit (the /n_excit keeps total
    recurrent drive roughly independent of network size -- see module
    docstring / Compte's own weight_scaling_factor, same role here).
    W_EE[i, j] = weight FROM presynaptic neuron j TO postsynaptic neuron i.
    """
    kernel, j_neg = mexican_hat_kernel(n_excit, sigma_deg, j_pos)
    idx = np.arange(n_excit)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_excit - dist)  # wrap around the ring
    W = kernel[dist] * (g_ee_scale / n_excit)
    return torch.tensor(W, device=device, dtype=dtype), j_neg


def preferred_angles(n_excit):
    """Preferred angle (degrees, [0, 360)) of each E neuron, index order."""
    return np.linspace(0.0, 360.0, n_excit, endpoint=False)


# ---------------------------------------------------------------------------
# Population-vector decode
# ---------------------------------------------------------------------------

def decode_population_vector(spike_or_rate, angles_deg):
    """spike_or_rate: (..., n_excit) tensor/array of activity (spike counts
    or instantaneous spikes) across the ring. angles_deg: (n_excit,) array
    of preferred angles. Returns (decoded_angle_deg, resultant_length) with
    the same leading shape as spike_or_rate minus the last axis.
    resultant_length in [0, 1] is how "peaked" the population vector is --
    near 0 means activity is spread all over the ring (no clear bump), near
    1 means a sharp, unambiguous bump. Standard circular-statistics decode,
    same one used in Wimmer et al. 2014 to read out the bump location."""
    if isinstance(spike_or_rate, torch.Tensor):
        act = spike_or_rate.detach().cpu().numpy()
    else:
        act = np.asarray(spike_or_rate)
    theta = np.deg2rad(angles_deg)
    x = (act * np.cos(theta)).sum(axis=-1)
    y = (act * np.sin(theta)).sum(axis=-1)
    total = act.sum(axis=-1)
    total_safe = np.where(total > 0, total, 1.0)
    resultant_length = np.sqrt(x ** 2 + y ** 2) / total_safe
    resultant_length = np.where(total > 0, resultant_length, 0.0)
    decoded_angle = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
    return decoded_angle, resultant_length


def circular_error_deg(decoded_deg, true_deg):
    """Smallest absolute difference |decoded - true|, wrapped to [0, 180]."""
    diff = (np.asarray(decoded_deg) - np.asarray(true_deg) + 180.0) % 360.0 - 180.0
    return np.abs(diff)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class RingSNN:
    """
    Ring-attractor E/I spiking network for the ODR task. E population is
    ALWAYS BistableNeuron (see module docstring -- the LIF condition has
    been dropped). I population is LIF, unstructured, at a fixed E:I ratio
    (caller picks n_inhib, conventionally n_excit // 4): by default the
    MNIST / N-MNIST LIF (i_model="lif"); i_model="lif_syn" gives the
    LIF_syn of every result before 2026-10-22 (see I_INPUT_GAIN).

    Parameters
    ----------
    n_excit, n_inhib : population sizes. Meant to be swept small -- see
        module docstring's NETWORK SIZE note.
    e_kwargs : extra kwargs for BistableNeuron (the one that matters here:
        w0, the calcium-channel-density / bistability parameter).
    i_kwargs : extra kwargs for the I population's neuron class.
    i_model : "lif" (LIF_simple, the MNIST LIF; default) or "lif_syn".
    i_input_gain : factor on every synapse onto an I neuron (E->I, I->I,
        background->I). None = I_INPUT_GAIN[i_model] (14 for "lif", 1 for
        "lif_syn").
    sigma_deg, j_pos : Mexican-hat kernel shape (defaults match Compte 2000
        / the Brian2 reference: sigma=20 deg, J+=1.6).
    g_ee_scale, g_ei_scale, g_ie_scale, g_ii_scale : overall current-based
        synaptic weight scales (see CALIBRATION NOTE in the module
        docstring).
    n_extern_poisson, bg_rate_hz : the external BACKGROUND noise population
        -- literally Compte's 1000-source, 1.4Hz-each Poisson bombardment.
        Same for E and I (matches the Brian2 code: one shared
        N_extern_poisson/poisson_firing_rate, only the per-spike weight
        differs by target population, same role as g_ext,E vs g_ext,I).
    bg_weight_e, bg_weight_i : current-based stand-in for g_ext,E/g_ext,I --
        the current kicked into a neuron by ONE external background spike.
    stim_weight_e : current-based weight per CUE/DISTRACTOR spike (separate
        knob from bg_weight_e -- the cue pathway doesn't have to use the
        same per-spike weight as the background pathway, even though both
        are now Poisson-generated the same way; see module docstring).
    refrac_i : absolute refractory period in ms (== steps) for the I
        population only (Compte's own t_abs_refract_inhib=1ms).
    dt_eff : RK4 micro-step for the bistable neuron -- decoupled from the
        outer loop's 1-step-per-ms convention, same as everywhere else in
        this codebase (mnist-ms-rebuild.md: "1 network step = 1 ms is a
        relabeling, not a re-derivation").
    """

    def __init__(self, n_excit=1024, n_inhib=256, e_kwargs=None, i_kwargs=None,
                 sigma_deg=20.0, j_pos=1.6,
                 g_ee_scale=1.6, g_ei_scale=1.2, g_ie_scale=1.2, g_ii_scale=0.8,
                 n_extern_poisson=1000, bg_rate_hz=1.4,
                 bg_weight_e=0.0105, bg_weight_i=0.0080,
                 stim_weight_e=0.0105,
                 refrac_i=1,
                 i_model="lif", i_input_gain=None,
                 dt_eff=0.025, device="cpu", dtype=torch.float, seed=None):
        self.n_excit = n_excit
        self.n_inhib = n_inhib
        self.device = device
        self.dtype = dtype
        self.dt_s = 1e-3  # 1 network step = 1 ms, everywhere in this file
        self.angles_deg = preferred_angles(n_excit)

        if seed is not None:
            torch.manual_seed(seed)

        # --- E population: ALWAYS bistable now ---------------------------
        from neuron_model import BistableNeuron
        e_kwargs = dict(e_kwargs or {})
        e_kwargs.setdefault("dt_eff", dt_eff)
        self.e_neuron = BistableNeuron(device=device, dtype=dtype, **e_kwargs)

        # --- I population: LIF (2026-10-22: the MNIST / N-MNIST LIF) -------
        # i_model="lif"     : LIF_simple, beta 0.5 per ms, no synaptic filter (default)
        # i_model="lif_syn" : LIF_syn, tau_mem 10 ms, tau_syn 5 ms (all results before 2026-10-22)
        from neuron_model import LIF_simple, LIF_syn
        i_kwargs = dict(i_kwargs or {})
        if i_model == "lif":
            self.i_neuron = LIF_simple(device=device, dtype=dtype, **i_kwargs)
        elif i_model == "lif_syn":
            self.i_neuron = LIF_syn(device=device, dtype=dtype, **i_kwargs)
        else:
            raise ValueError("i_model must be 'lif' (MNIST LIF) or 'lif_syn'")
        self.i_model = i_model
        # One factor on EVERY synapse onto an I neuron (E->I, I->I, background->I).
        # See I_INPUT_GAIN at the top of this file for why the MNIST LIF needs it.
        if i_input_gain is None:
            i_input_gain = I_INPUT_GAIN[i_model]
        self.i_input_gain = float(i_input_gain)

        # --- connectivity ------------------------------------------------
        self.W_EE, self.j_neg = build_W_EE(n_excit, sigma_deg, j_pos, g_ee_scale, device, dtype)
        # uniform, unstructured E<->I and I<->I -- plain scalar / N, no angular dependence
        self.w_ei = self.i_input_gain * g_ei_scale / n_excit   # each I neuron sums input from all N_E E neurons
        self.w_ie = -g_ie_scale / n_inhib  # each E neuron sums (inhibitory) input from all N_I I neurons
        self.w_ii = -self.i_input_gain * g_ii_scale / n_inhib

        # --- external Poisson BACKGROUND noise ----------------------------
        self.n_extern_poisson = n_extern_poisson
        self.bg_rate_hz = bg_rate_hz
        self.bg_weight_e, self.bg_weight_i = bg_weight_e, self.i_input_gain * bg_weight_i
        # effective aggregate background RATE in Hz per neuron (same for
        # every E/I neuron, every step -- n_extern_poisson * bg_rate_hz).
        # Deliberately dense (~1.4kHz) -- most neurons get at least one
        # background event on most steps, exactly like the paper's
        # relentless bombardment (it's the strong feedback inhibition, not
        # a sparse noise source, that keeps the spontaneous rate low -- see
        # module docstring).
        self._bg_rate_hz = n_extern_poisson * bg_rate_hz

        # --- cue/distractor Poisson weight (rate comes in per-step from task.py) ---
        self.stim_weight_e = stim_weight_e

        # --- network-level absolute refractory period (I population only) ---
        self.refrac_steps_i = int(refrac_i)

    def _poisson_drive(self, rate_hz, weight):
        """One step's worth of Poisson-generated external current for a
        population, given a RATE in Hz (can be a python float applied to
        every unit, or a (batch, n_units) tensor of per-neuron rates --
        used both for the constant background rate and for the
        time/space-varying cue+distractor rate). Returns (current,
        spike_counts) -- spike_counts kept around for the input raster."""
        if not torch.is_tensor(rate_hz):
            rate_hz = torch.as_tensor(rate_hz, device=self.device, dtype=self.dtype)
        lam = torch.clamp(rate_hz * self.dt_s, min=0)
        counts = torch.poisson(lam)
        return counts * weight, counts

    def _apply_refractory(self, state, spk, refrac_counter, refrac_steps):
        """Suppress spikes and freeze membrane voltage for any neuron still
        inside its refractory window, as a wrapper around a LIF_syn step --
        LIF_syn's own step() is untouched. A neuron that just spiked
        (spk_out==1, i.e. it wasn't suppressed) restarts the countdown at
        refrac_steps."""
        if refrac_steps <= 0:
            return state, spk, refrac_counter
        in_refrac = refrac_counter > 0
        spk_out = torch.where(in_refrac, torch.zeros_like(spk), spk)
        state = dict(state)
        if "mem" in state:
            state["mem"] = torch.where(in_refrac, torch.zeros_like(state["mem"]), state["mem"])
        refrac_counter = torch.clamp(refrac_counter - 1, min=0)
        refrac_counter = torch.where(spk_out > 0.5,
                                      torch.full_like(refrac_counter, float(refrac_steps)),
                                      refrac_counter)
        return state, spk_out, refrac_counter

    def run(self, stim_rate_e):
        """
        stim_rate_e : (batch, T, n_excit) tensor -- external cue+distractor
                      RATE in Hz onto the E population at every step (NOT a
                      current -- see module docstring's "INPUT IS NOW
                      SPIKING"). Build with task.py. Cue and distractor are
                      already combined into this one rate tensor (they
                      don't overlap in time), the same way the old
                      current-based version summed them before use.

        Returns recs: dict with
            spk_e, spk_i : (batch, T, n_excit)/(batch, T, n_inhib) spike trains
            mem_e, mem_i : membrane/voltage traces, same shapes
            noise_spk_e, noise_spk_i : background-noise spike COUNTS actually
                drawn at every step, same shapes as spk_e/spk_i.
            stim_spk_e : (batch, T, n_excit) cue+distractor spike COUNTS
                actually drawn at every step (this is genuinely stochastic
                now -- different every trial, even for the same
                stim_rate_e).
            stim_rate_e : the deterministic RATE profile passed in (kept in
                recs for the input raster's heatmap panel).
            decoded_deg, resultant_length : (batch, T) population-vector
                readout of the E population at every step.
            decoded_final_deg, resultant_final : confidence-weighted
                circular-mean readout over the trailing readout_window ms
                (see the long comment below -- this is NOT a raw
                spike-count sum, for good reason).
        """
        batch_size, T, n_excit = stim_rate_e.shape
        assert n_excit == self.n_excit

        state_e = self.e_neuron.init_state(batch_size, self.n_excit)
        state_i = self.i_neuron.init_state(batch_size, self.n_inhib)
        spk_e_prev = torch.zeros(batch_size, self.n_excit, device=self.device, dtype=self.dtype)
        spk_i_prev = torch.zeros(batch_size, self.n_inhib, device=self.device, dtype=self.dtype)
        refrac_counter_i = torch.zeros(batch_size, self.n_inhib, device=self.device, dtype=self.dtype)

        spk_e_rec, spk_i_rec, mem_e_rec, mem_i_rec = [], [], [], []
        noise_spk_e_rec, noise_spk_i_rec, stim_spk_e_rec = [], [], []
        w_e_rec = []

        for t in range(T):
            # recurrent input, one-step delay (standard discrete-time SNN
            # convention: this step's current is built from LAST step's
            # spikes)
            rec_e = spk_e_prev @ self.W_EE.T + spk_i_prev.sum(dim=1, keepdim=True) * self.w_ie
            rec_i = spk_e_prev.sum(dim=1, keepdim=True) * self.w_ei + spk_i_prev.sum(dim=1, keepdim=True) * self.w_ii

            bg_rate_e = torch.full((batch_size, self.n_excit), self._bg_rate_hz, device=self.device, dtype=self.dtype)
            bg_rate_i = torch.full((batch_size, self.n_inhib), self._bg_rate_hz, device=self.device, dtype=self.dtype)
            bg_e, noise_e = self._poisson_drive(bg_rate_e, self.bg_weight_e)
            bg_i, noise_i = self._poisson_drive(bg_rate_i, self.bg_weight_i)
            stim_e, stim_spk = self._poisson_drive(stim_rate_e[:, t], self.stim_weight_e)

            I_e = rec_e + stim_e + bg_e
            I_i = rec_i.expand(-1, self.n_inhib) + bg_i

            mem_e_rec.append(self.e_neuron.voltage(state_e))
            mem_i_rec.append(self.i_neuron.voltage(state_i))
            w_e_rec.append(self.e_neuron.recovery(state_e))

            state_e, spk_e = self.e_neuron.step(state_e, I_e)

            state_i, spk_i = self.i_neuron.step(state_i, I_i)
            state_i, spk_i, refrac_counter_i = self._apply_refractory(
                state_i, spk_i, refrac_counter_i, self.refrac_steps_i)

            spk_e_rec.append(spk_e)
            spk_i_rec.append(spk_i)
            noise_spk_e_rec.append(noise_e)
            noise_spk_i_rec.append(noise_i)
            stim_spk_e_rec.append(stim_spk)
            spk_e_prev, spk_i_prev = spk_e, spk_i

        spk_e_rec = torch.stack(spk_e_rec, dim=1)
        spk_i_rec = torch.stack(spk_i_rec, dim=1)
        mem_e_rec = torch.stack(mem_e_rec, dim=1)
        mem_i_rec = torch.stack(mem_i_rec, dim=1)
        noise_spk_e_rec = torch.stack(noise_spk_e_rec, dim=1)
        noise_spk_i_rec = torch.stack(noise_spk_i_rec, dim=1)
        stim_spk_e_rec = torch.stack(stim_spk_e_rec, dim=1)
        w_e_rec = torch.stack(w_e_rec, dim=1)

        decoded_deg, resultant_length = decode_population_vector(
            spk_e_rec.detach().cpu().numpy(), self.angles_deg)

        # Final readout: a CONFIDENCE-WEIGHTED CIRCULAR MEAN of the per-step
        # decode over the last `readout_window` ms, not a raw spike-count
        # sum. Why: the bistable population fires in periodic collective
        # bursts (~150-165ms apart). Between bursts, background noise alone
        # gives every E neuron a spike count comparable to what a "bump"
        # neuron accumulates from participating in 1-2 bursts, so summing
        # raw spike counts over a long window drowns the (spatially
        # concentrated but numerically modest) bump signal in the (spread
        # out but numerically comparable) background -- verified empirically,
        # it reliably decoded ~180deg off. The per-step decode doesn't have
        # this problem: resultant_length is already ~1.0 exactly AT a burst
        # and ~0 in the gaps, i.e. it already knows which instants to trust.
        # So the final readout is a circular mean of per-step unit vectors,
        # weighted by each step's own resultant_length.
        readout_window = min(400, T)
        dec_w = decoded_deg[:, -readout_window:]
        res_w = resultant_length[:, -readout_window:]
        rad_w = np.deg2rad(dec_w)
        wx = (res_w * np.cos(rad_w)).sum(axis=-1)
        wy = (res_w * np.sin(rad_w)).sum(axis=-1)
        denom = np.clip(res_w.sum(axis=-1), 1e-9, None)
        resultant_final = np.sqrt(wx ** 2 + wy ** 2) / denom
        decoded_final_deg = (np.rad2deg(np.arctan2(wy, wx)) + 360.0) % 360.0

        recs = {
            "spk_e": spk_e_rec, "spk_i": spk_i_rec,
            "mem_e": mem_e_rec, "mem_i": mem_i_rec, "w_e": w_e_rec,
            "noise_spk_e": noise_spk_e_rec, "noise_spk_i": noise_spk_i_rec,
            "stim_spk_e": stim_spk_e_rec, "stim_rate_e": stim_rate_e,
            "decoded_deg": decoded_deg, "resultant_length": resultant_length,
            "decoded_final_deg": decoded_final_deg, "resultant_final": resultant_final,
            "readout_window": readout_window,
            "angles_deg": self.angles_deg, "T": T,
        }
        return recs
