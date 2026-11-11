# Bistability for working memory

Code for the paper

> **Spike-efficient working memory with bistability through neuromodulation for neuromorphic hardware**
> Daria Mihaila, Loris Mendolia and Guillaume Drion
> Neuroengineering Laboratory, University of Liège, Belgium
> Submitted to NICE 2027

## In short

Working memory holds information after its source has gone. On neuromorphic hardware every spike costs energy, so a memory should be held with as few spikes as possible. We take a neuron model in which one parameter, $w_0$, stands for the density of calcium channels. Negative $w_0$ makes the neuron bistable: a brief input switches it between rest and persistent firing. Non-negative $w_0$ makes it behave like an integrate-and-fire (IF) neuron. With this one parameter we show that:

1. **Feedforward networks remember without recurrent connections.** Trained on MNIST and N-MNIST with a 100 ms silent delay, networks of bistable neurons hold their answer over delays fifty times longer.
2. **Heterogeneity saves spikes.** LIF hidden neurons with bistable output neurons hold the answer with 34 times fewer spikes than an all-bistable network.
3. **The memory is robust.** It survives multiplicative noise of up to 50 % on the inputs and on the trained weights.
4. **Switching $w_0$ gives attention and forgetting.** In a ring attractor for the oculomotor delayed-response (ODR) task, a receptive field of bistable neurons protects the memory from distractors, and a brief negative pulse clears it. The cue is held with about 1 % error, at three times fewer spikes or more.
5. **The memory steers behaviour.** The same ring guides an agent to a target it has seen only once.
6. **The mechanism exists in silicon.** In a mixed-feedback silicon neuron, the gain of the slow positive feedback plays the role of $w_0$ (Fig. 1B).

## The neuron model

We use the transcritical hybrid model of Drion et al. (2012), with the reset of Izhikevich (2003):

$$\dot v = v^2 - w^2 + I_{ext} + I_{TC}, \qquad \dot w = \varepsilon\,(a v - w + w_0), \qquad \text{if } v \ge v_{th}:\ v \leftarrow c,\ w \leftarrow d,$$

with $I_{TC} = w_0^2/(1-a^2)$, $v_{th} = 1$, $\varepsilon = 0.5$, $a = 0.1$, $c = 0$ and $d = 1$. The term $-w^2$ captures slow voltage-gated calcium channels, and $w_0$ is an image of their conductance:

- $w_0 < 0$: high calcium-channel density, rest-spike bistability (the neuron can hold a memory);
- $w_0 \ge 0$: low density, IF-like behaviour (the neuron forgets).

One network step is 1 ms, integrated with one RK4 step of 0.025 model time units. The model is written in NumPy in `single_neuron/` and in PyTorch, with a surrogate gradient, in each `neuron_model.py`.

## Repository layout

| folder | paper | content |
|---|---|---|
| [`single_neuron/`](single_neuron) | Sec. II, Fig. 1 | the hybrid model in NumPy: phase planes, response to pulses |
| [`MNIST/`](MNIST) | Sec. III, Fig. 2 | feedforward SNNs on MNIST: training, delay, activity and noise experiments, trained networks |
| [`N_MNIST/`](N_MNIST) | Sec. III, Fig. 2 | the same on N-MNIST (event-camera digits) |
| [`ODR/`](ODR) | Sec. IV-A, Fig. 3 | ring attractor for the ODR task: receptive-field gating and forgetting |
| [`navigation/`](navigation) | Sec. IV-B, Fig. 4 | the ring steers an agent to a target seen once |
| [`NICE_figures/`](NICE_figures) | Figs. 1–4 | one notebook per figure, supplementary figures, the rendered figures |

Each experiment folder is self-contained. It has its own copy of the neuron model and is run from inside the folder. Each folder has its own README.

The on-chip network of Fig. 5 (neuromodulatory synapses on mixed-feedback neurons) is hardware work and is not part of this repository.

## Installation

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

Training uses CUDA or Apple MPS when available, otherwise the CPU. Everything else runs on a laptop CPU.

## Reproducing the figures

The trained networks and all result tables are included, so no figure needs training.

```bash
cd NICE_figures
jupyter lab
```

| notebook | figure | needs |
|---|---|---|
| `fig1_neuron_model.ipynb` | Fig. 1 | NumPy and Matplotlib only |
| `fig2_mnist_nmnist.ipynb` | Fig. 2 | PyTorch and the MNIST test set (downloaded by torchvision) for panel B, the first time |
| `fig3_odr.ipynb` | Fig. 3 | PyTorch: panel A simulates one trial (under a minute) |
| `fig4_navigation.ipynb` | Fig. 4 | PyTorch: panels B and C simulate one trial (seconds) |
| `supplementary.ipynb` | robustness, rasters, learnt $w_0$, ODR network | as Fig. 2 |

The rendered figures are in [`NICE_figures/figures/`](NICE_figures/figures).

## Re-running the experiments

| folder | command, from inside the folder |
|---|---|
| `MNIST/` | `bash run_paper_experiments.sh` |
| `N_MNIST/` | `bash run_paper_experiments.sh` |
| `ODR/` | `ODR_w0_gating.ipynb` with `RERUN = True`, then `python exp_loop_rates.py` |
| `navigation/` | `python navigation.py --procs 6`, or `navigation.ipynb` |

The scripts skip networks and runs that already exist. Move a run out of `results/` to compute it again.

## Notes

- **Networks in the paper.** On both datasets: LIF; constant $w_0 \in \{-2, -1, -0.5, +0.5, +1, +2\}$; heterogeneous (LIF hidden, bistable output, $w_0 = -1$); two-tailed learnt $w_0$ (hidden layer half at $-1$, half at $+1$, s.d. 0.5; output layer around $-1$); fixed uniform $w_0 \sim \mathcal{U}[-1.87, -0.13]$ (mean $-1$, s.d. 0.5).
- **Epochs.** Every network was trained for 50 epochs, except the MNIST LIF network (20 epochs; its accuracy had plateaued). Some N-MNIST networks were trained in two stages with `N_MNIST/continue_training.py`; see `resume_log` in their `config.json`.
- **Names.** In `navigation/` the target is called an *apple*: the task began as a snake game. The MNIST results folder is `mnist_ms_experiments` (ms: one network step is one millisecond).
- **Hardware trace.** The silicon-neuron recording of Fig. 1B (`NICE_figures/fig1_c.csv`) was made by Loris Mendolia on the current-mode mixed-feedback neuron of Mendolia et al. (2026).

## (potential future) Citation

```bibtex
@inproceedings{mihaila2027bistability,
  title     = {Spike-efficient working memory with bistability through neuromodulation for neuromorphic hardware},
  author    = {Mihaila, Daria and Mendolia, Loris and Drion, Guillaume},
  journal = {},
  year      = {2027},
  note      = {Submitted}
}
```

The neuron model comes from:

> G. Drion, A. Franci, V. Seutin and R. Sepulchre. A novel phase portrait for neuronal excitability. *PLoS ONE* 7(8): e41806, 2012. https://doi.org/10.1371/journal.pone.0041806

The code also builds on the hybrid reset of Izhikevich (2003), the ring network of Compte et al. (2000), the N-MNIST dataset of Orchard et al. (2015), surrogate-gradient learning (Neftci, Mostafa and Zenke, 2019), [snnTorch](https://github.com/jeshraghian/snntorch) and [Tonic](https://github.com/neuromorphs/tonic).

## Licence

MIT Licence. Please cite this work when using models, results and investigations presented in this project. 
