# ODR: attention and forgetting in a ring attractor (Sec. IV-A)

The oculomotor delayed-response task: a cue flashes at an angle, a distractor follows, and the network must hold the cue's angle through the delay. The network follows Compte et al. (2000): 1024 bistable excitatory neurons on a ring, ordered by preferred angle, with Mexican-hat connectivity, and 256 LIF inhibitory neurons. The weights are fixed (no training, no plasticity). Cue and distractor are Poisson spikes around one angle, and the angle is decoded with the population vector. `METHODS_original_vs_ours.md` lists the differences from Compte et al.

Two signals act on $w_0$ during the task:

- **Receptive field (attention).** At the end of the cue, neurons within ±42.5° of the decoded angle keep $w_0 = -1$; the others switch to $w_0^{out} = 0$ or $+1$ and can no longer hold activity.
- **Negative pulse (forgetting).** Before each new cue, a brief negative input to every excitatory neuron returns the bistable neurons to rest.

## Run

Open `ODR_w0_gating.ipynb`. Section 1 simulates one trial per network; sections 2 and 3 read their tables from `results/` (set `RERUN = True` to run them again). Then

```bash
python exp_loop_rates.py     # the ten-memory runs again, with the firing rate of each cue presentation (about 15 min on a CPU)
```

## Files

| file | content |
|---|---|
| `neuron_model.py` | bistable neuron and LIF neurons in PyTorch (the same file as `../navigation/neuron_model.py`) |
| `network_architecture.py` | `RingSNN`: the E/I ring, its weights, background Poisson input, population-vector decoding |
| `task.py` | cue and distractor as Poisson spike trains |
| `run_odr_demo.py` | `DEFAULT_KWARGS`: the network settings of every experiment; also a one-trial demo |
| `plots.py` | plots for the demo |
| `w0_gating.py` | $w_0$ per neuron and per time step: receptive field, forgetting pulse, one trial, raster plot |
| `w0_experiments.py` | the six networks (`CUE500_CASES`), sequences of trials, distractor and repeat sweeps, tables |
| `exp_loop_rates.py` | firing rate per cue presentation (Fig. 3C) |
| `ODR_w0_gating.ipynb` | every ODR result of the paper |

## Results

`results/w0_gating_cue500_mnistLIF/` (the inhibitory neurons are the LIF of the MNIST networks):

| table | content | used in |
|---|---|---|
| `cue500_overlap.csv` | memory-1 error against cue–distractor separation, 16 trials per point | Fig. 3B |
| `cue500_repeats.csv` | ten memories per trial, 16 trials per network | notebook, section 3 |
| `cue500_repeats_looprates.csv` | the same, with the firing rate of each cue presentation | Fig. 3C |
| `table_cue500.tex` | error and coding cost of memories 2–10 | – |
