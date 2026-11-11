# The single neuron (Sec. II)

The hybrid model of Drion et al. (2012) in NumPy (Eqs. 1–3 of the paper). Used by `../NICE_figures/` for Fig. 1.

| file | content |
|---|---|
| `params.py` | $v_{th}$, $\varepsilon$, $a$, $c$, $d$ (the same values as in the networks) |
| `neuron_figures.py` | nullclines, fixed points and trajectories for the phase planes (`_phase_panel`, `plot_phase_planes`); the response to a negative then a positive pulse (`simulate_pulses`, `plot_pulse_response`) |

```python
import neuron_figures as nf
sim = nf.simulate_pulses(w0=-1, t_start=500, pulse_ms=200, gap_ms=900, t_end=2400)
fig, axes = nf.plot_phase_planes(w0=-1)
```
