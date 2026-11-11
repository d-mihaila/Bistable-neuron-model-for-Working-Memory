# N-MNIST: working memory with a time-varying input (Sec. III)

The same networks and protocol as `../MNIST/`, on N-MNIST (Orchard et al., 2015): MNIST digits recorded by an event camera that makes three saccades over each digit. The 2 × 34 × 34 = 2312 inputs carry the ON and OFF events, binned by Tonic into 100 time steps (one network step, 1 ms, each). A trial is 100 ms of input followed by a silent delay.

## Run

```bash
bash run_paper_experiments.sh      # trains the missing networks, then every post-hoc experiment
```

The training and experiment scripts take the same arguments as in `../MNIST/` (e.g. `python run_const_w0.py --w0 -1 --nb_epochs 50`; the default here is 25 epochs). N-MNIST is downloaded by Tonic into `data/` the first time, and the binned frames are cached in `cache/`.

## Files

| file | content |
|---|---|
| `neuron_model.py`, `network_architecture.py` | as in `../MNIST/` (same files) |
| `nmnist_w0_lib.py` | constants, data loading (Tonic `ToFrame`), the network variants, the training loop |
| `save_experiment.py`, `rebuild_network.py` | saving runs; rebuilding trained networks from `results/` |
| `run_const_w0.py`, `run_lif_hidden.py`, `run_distribution_w0.py`, `run_all_lif.py` | training, one script per network type |
| `continue_training.py` | resumes saved runs to more epochs (used for some of the shipped networks; see `resume_log` in their `config.json`) |
| `exp_waiting.py`, `exp_delay_activity.py` | accuracy against delay; spikes during a 5 s trial |
| `exp_noise_weights_cv.py` | every trained weight scaled by $1 + \mathrm{CV}\,\xi$ |
| `exp_noise_input_cv.py` | one factor $f = 1 + \mathrm{CV}\,\xi$ per digit and input channel: for $f < 1$ each event is kept with probability $f$, for $f > 1$ each event is copied with probability $f - 1$ |
| `run_noise_cv.sh` | both noise sweeps, CV 0–50 %, 5 noise draws, the same 1280 test digits for every network |
| `run_paper_experiments.sh` | everything above, in order |

## Results

`results/nmnist_w0_experiments/` holds the ten networks of the paper and the post-hoc tables, laid out as in `../MNIST/results/`. Fig. 2C reads `post_hoc/waiting/twait_sweep.csv`.
