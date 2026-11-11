# MNIST: working memory in feedforward networks (Sec. III)

A network of 784 inputs, 100 hidden and 10 output neurons, all-to-all and feedforward, with no recurrent connections. Each input spikes in each 1 ms step with a probability equal to its pixel intensity. A trial is 100 ms of input followed by a silent delay, and the class is the output neuron that fires most in the last 100 ms. The networks are trained with surrogate gradients and a 100 ms delay, then tested with delays up to 5 s.

## Run

```bash
bash run_paper_experiments.sh      # trains the missing networks, then every post-hoc experiment
```

Or step by step:

```bash
python run_const_w0.py --w0 -1                                             # constant w0 in both layers
python run_lif_hidden.py --w0 -1                                           # heterogeneous: LIF hidden, bistable output
python run_distribution_w0.py --dist two_tailed --mean -1 --std 0.5 --learnable
python run_distribution_w0.py --dist uniform --mean -1 --std 0.5          # w0 fixed after sampling
python run_all_lif.py --t_wait 100 --nb_epochs 20
python exp_waiting.py                                                       # accuracy against delay
python exp_delay_activity.py                                                # spikes during a 5 s trial
bash run_noise_cv.sh                                                        # weight- and input-noise robustness
```

Add `--max_batches_per_epoch 2 --nb_epochs 1` to a training script for a quick test. MNIST is downloaded into `data/` the first time.

## Files

| file | content |
|---|---|
| `neuron_model.py` | the bistable neuron in PyTorch (RK4 step, surrogate spike function) |
| `network_architecture.py` | the feedforward SNN (input → hidden → output) |
| `mnist_lib.py` | constants, data loading, Poisson encoding, the network variants, the training loop |
| `save_experiment.py` | saving runs and rebuilding trained networks from `results/` |
| `run_const_w0.py`, `run_lif_hidden.py`, `run_distribution_w0.py`, `run_all_lif.py` | training, one script per network type |
| `exp_waiting.py` | accuracy after delays of 0.1, 0.2, 0.5, 1, 2 and 5 s (1280 test digits) |
| `exp_delay_activity.py` | spikes per 100 ms over a 5 s trial (512 test digits) |
| `exp_noise_weights_cv.py` | every trained weight scaled by its own factor $1 + \mathrm{CV}\,\xi$, $\xi \sim \mathcal{N}(0,1)$ |
| `exp_noise_input_cv.py` | every pixel rate scaled the same way, frozen over the 100 ms input |
| `run_noise_cv.sh` | both noise sweeps, CV 0–50 %, 5 noise draws per level |
| `run_paper_experiments.sh` | everything above, in order |

## Results

`results/mnist_ms_experiments/` holds the ten networks of the paper (`<run>/weights.pt`, `config.json`, `history.json`) and the post-hoc tables:

| table | used in |
|---|---|
| `post_hoc/waiting/twait_sweep.csv` | Fig. 2A |
| `post_hoc/delay_activity/delay_activity.csv` | Fig. 2B (spike rates) |
| `post_hoc/weight_noise_cv/weight_noise_cv_sweep.csv`, `post_hoc/input_noise_cv/input_noise_cv_sweep.csv` | robustness, Sec. III (supplementary figure) |

`neuron_model.py` and `network_architecture.py` are the same files as in `../N_MNIST/`.
