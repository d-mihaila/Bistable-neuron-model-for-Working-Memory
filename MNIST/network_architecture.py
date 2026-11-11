"""
network_architecture.py

A flexible 2-layer (input -> recurrent hidden -> readout) spiking network.
The hidden layer's neuron dynamics are injected as a `hidden_neuron` object
(see neuron_model.py) -- swap LIFNeuron() for BistableNeuron() and nothing
else in the training loop needs to change.

This is an exact copy of ../HSD/network_architecture.py -- fully generic,
no NMNIST/HSD-specific code in it, kept here so this folder's notebooks can
import it locally.

The readout layer is a non-spiking double-exponential filter, same as in
the SpyTorch tutorials: it has no threshold, so classification is done by
max-over-time on its continuous output. It's kept as its own small class
so its time constants can be set independently of the hidden layer if you
ever want that (e.g. bistable hidden layer, LIF-style readout).
"""

import numpy as np
import torch


class ReadoutLayer:
    """
    Non-spiking double-exponential readout filter -- LEGACY / optional.

    Not used by default anymore: the output layer is now spiking, using the
    same neuron model as the hidden layer (see RecurrentSNN below), with
    classification decided by spike count. Kept here in case you want a
    continuous-readout variant again for comparison.
    """

    def __init__(self, tau_mem=10e-3, tau_syn=5e-3, time_step=1e-3,
                 device="cpu", dtype=torch.float):
        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))
        self.device = device
        self.dtype = dtype

    def init_state(self, batch_size, n_units):
        flt = torch.zeros((batch_size, n_units), device=self.device, dtype=self.dtype)
        out = torch.zeros((batch_size, n_units), device=self.device, dtype=self.dtype)
        return flt, out

    def step(self, flt, out, h2_t):
        new_flt = self.alpha * flt + h2_t
        new_out = self.beta * out + flt
        return new_flt, new_out


class SNN:
    """
    Input -> hidden layer -> spiking output layer.

    Both layers use the same neuron model TYPE (pass two separate instances
    with the same hyperparameters -- they need independent state, but should
    share dynamics for a fair comparison). Classification is decided by
    spike count: whichever output unit fires most in the last `T_eval`
    steps wins. That same spike-count vector is what the loss is computed
    on, so training and evaluation use the same quantity.

    Parameters
    ----------
    hidden_neuron, output_neuron : objects with .init_state/.step/.voltage
                                    (see neuron_model.py) -- typically same
                                    class, same hyperparameters
    weight_scale  : fan-in weight init scale (kept as in the original tutorial)
    recurrent     : whether the hidden layer has recurrent weights (v1)
    T_eval        : number of final timesteps used for the spike-count vote
    """

    def __init__(self, nb_inputs, nb_hidden, nb_outputs, nb_steps,
                 hidden_neuron, output_neuron, weight_scale=0.2, T_eval=50, T_wait=0,
                 device="cpu", dtype=torch.float):
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.nb_steps = nb_steps  # stimulus-phase length only -- see T_wait below
        self.hidden_neuron = hidden_neuron
        self.output_neuron = output_neuron
        self.T_eval = T_eval
        self.T_wait = T_wait  # silent steps appended after the stimulus, network keeps running
        self.device = device
        self.dtype = dtype

        self.w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
        # torch.nn.init.normal_(self.w1, mean=0.0, std=weight_scale / np.sqrt(nb_inputs))
        torch.nn.init.normal_(self.w1, mean=0.0, std=0.5)
        print(f"w1 init std: {self.w1.std().item():.4f} (zenke std {weight_scale / np.sqrt(nb_inputs):.4f})")

        self.w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
        # torch.nn.init.normal_(self.w2, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))
        torch.nn.init.normal_(self.w2, mean=0.0, std=0.5)
        print(f"w2 init std: {self.w2.std().item():.4f} (zenke std {weight_scale / np.sqrt(nb_hidden):.4f})")

        self.v1 = None

    def parameters(self):
        params = [self.w1, self.w2]
        return params

    def forward(self, inputs):
        """
        inputs: dense tensor (batch, nb_steps, nb_inputs) -- the STIMULUS
                only. If self.T_wait > 0, that many all-zero timesteps are
                appended here before the simulation runs, so the network
                keeps integrating in silence after the stimulus ends.

        Returns
        -------
        spike_count : (batch, nb_outputs) -- spikes per output unit over the
                      last T_eval steps (of the FULL run, stimulus+wait).
                      This is what the loss and the classification decision
                      both use.
        recs : dict with 'mem'/'spk' (hidden) and 'out_mem'/'out_spk' (output),
               each (batch, nb_steps+T_wait, n_units). Voltages are recorded
               BEFORE each step's update -- i.e. the value actually compared
               against threshold that step -- so a spike's peak is visible in
               the trace, with the reset showing up on the following point.
               If hidden_neuron/output_neuron expose a `.recovery(state)`
               method (bistable models only -- see neuron_model.py), the
               same layer's slow recovery variable is recorded too, as
               'w'/'out_w'. LIF neurons have no such method, so those keys
               are simply absent from recs for LIF runs -- check with
               `"w" in recs` rather than assuming the key exists.
               Also includes 'total_steps' (nb_steps + T_wait), 't_stim_end'
               (step index where the stimulus ends / silence begins -- equal
               to nb_steps) and 't_eval_start' (step index where the T_eval
               vote window begins) -- handy for drawing markers on rasters.
        """
        batch_size = inputs.shape[0]

        if self.T_wait > 0:
            silence = torch.zeros((batch_size, self.T_wait, self.nb_inputs),
                                   device=self.device, dtype=self.dtype)
            inputs = torch.cat([inputs, silence], dim=1)  # (batch, nb_steps+T_wait, nb_inputs)

        total_steps = inputs.shape[1]  # nb_steps + T_wait

        record_hidden_w = hasattr(self.hidden_neuron, "recovery")
        record_output_w = hasattr(self.output_neuron, "recovery")

        # ---- hidden layer -------------------------------------------------
        state = self.hidden_neuron.init_state(batch_size, self.nb_hidden)
        out_spk = torch.zeros((batch_size, self.nb_hidden), device=self.device, dtype=self.dtype)

        mem_rec, spk_rec = [], []
        w_rec = [] if record_hidden_w else None
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))

        for t in range(total_steps):
            h1 = h1_from_input[:, t]

            mem_rec.append(self.hidden_neuron.voltage(state))  # pre-update: value tested this step
            if record_hidden_w:
                w_rec.append(self.hidden_neuron.recovery(state))
            state, out_spk = self.hidden_neuron.step(state, h1)
            spk_rec.append(out_spk)

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        if record_hidden_w:
            w_rec = torch.stack(w_rec, dim=1)

        # ---- output layer (spiking, same neuron type as hidden) ----------
        out_state = self.output_neuron.init_state(batch_size, self.nb_outputs)
        out_mem_rec, out_spk_rec = [], []
        out_w_rec = [] if record_output_w else None
        h2_from_hidden = torch.einsum("abc,cd->abd", (spk_rec, self.w2))

        for t in range(total_steps):
            out_mem_rec.append(self.output_neuron.voltage(out_state))
            if record_output_w:
                out_w_rec.append(self.output_neuron.recovery(out_state))
            out_state, out_spk = self.output_neuron.step(out_state, h2_from_hidden[:, t])
            out_spk_rec.append(out_spk)

        out_mem_rec = torch.stack(out_mem_rec, dim=1)
        out_spk_rec = torch.stack(out_spk_rec, dim=1)
        if record_output_w:
            out_w_rec = torch.stack(out_w_rec, dim=1)

        spike_count = torch.sum(out_spk_rec[:, -self.T_eval:, :], dim=1)  # (batch, nb_outputs)

        recs = {"mem": mem_rec, "spk": spk_rec, "out_mem": out_mem_rec, "out_spk": out_spk_rec,
                "total_steps": total_steps, "t_stim_end": self.nb_steps,
                "t_eval_start": total_steps - self.T_eval}
        if record_hidden_w:
            recs["w"] = w_rec
        if record_output_w:
            recs["out_w"] = out_w_rec
        return spike_count, recs
