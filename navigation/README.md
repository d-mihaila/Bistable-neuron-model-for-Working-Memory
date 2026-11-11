# Navigation from a single cue (Sec. IV-B)

The ODR memory steers an agent. On a 10 × 10 board the agent sees a target once, for 100 ms, and must walk to it from memory, one step (N, E, S or W) per 200 ms. The code calls the target an *apple*: the task began as a snake game.

The network (22 neurons): a ring of 16 excitatory neurons with Mexican-hat connectivity, the receptive-field gate and the forgetting pulse of `../ODR/` (no inhibitory neurons); four decision neurons N, E, S, W that pool the ring around their direction and inhibit each other (winner-take-all); and two distance counters, one per axis, that block the decision neurons of their axis once the agent has walked far enough along it.

Three networks: all bistable ($w_0 = -1$ in ring and decision neurons), heterogeneous (bistable ring, LIF decision neurons) and all LIF. The score is the number of targets reached per 10,000 spikes.

## Run

```bash
python navigation.py --procs 6      # every experiment; resumes, skips runs already in results/runs.csv
```

or open `navigation.ipynb`, which runs the same experiments and draws their figures and tables. A full run is 340 runs of 30 targets (about 30–40 min with 6 processes).

| experiment | networks | conditions |
|---|---|---|
| `base` | all three | – |
| `weight_noise` | the two bistable rings | every cue, ring and decision synapse × $(1 + \mathrm{CV}\,\xi)$, CV 10–50 % |
| `rf_forget` | the two bistable rings | no RF, no forgetting, neither |
| `input_noise` | all three | cue rate of each ring neuron × its own gain, CV 10–50 % (not in the paper) |

10 runs (seeds 0–9) per condition.

## Files

| file | content |
|---|---|
| `neuron_model.py` | bistable neuron and LIF neurons (the same file as `../ODR/neuron_model.py`) |
| `network.py` | the ring, cue, decision neurons and counters (`SnakeNet`) |
| `board.py` | the board and the target positions drawn from the seed |
| `navigation.py` | the task, the experiments, the tables and their figures |
| `navigation.ipynb` | runs everything, with explanations |

## Results

`results/runs.csv` (one row per run, every metric), `results/tables/` (one table per experiment) and the notebook's figures. Fig. 4 of the paper is drawn from `runs.csv` by `../NICE_figures/fig4_navigation.ipynb`.
