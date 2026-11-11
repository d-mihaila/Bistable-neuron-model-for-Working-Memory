# Figures of the paper

One notebook per figure. Every notebook reads the results of the experiment folders and runs nothing long; see the table in the main README for what each one needs. Figures are saved to `figures/` as PDF and 300 dpi PNG.

| notebook | saves | |
|---|---|---|
| `fig1_neuron_model.ipynb` | `fig1_neuron_model` | Fig. 1: phase planes; simulated and silicon neuron under input pulses |
| `fig2_mnist_nmnist.ipynb` | `fig2_mnist_nmnist` | Fig. 2: accuracy against delay (MNIST, N-MNIST), MNIST rasters and spike rates; the numbers of Sec. III |
| `fig3_odr.ipynb` | `fig3_odr` | Fig. 3: one ODR trial, distractor immunity, error against firing rate |
| `fig4_navigation.ipynb` | `fig4_navigation` | Fig. 4: network, one trial, targets per 10,000 spikes |
| `supplementary.ipynb` | `supp_*` | robustness to weight and input noise (Sec. III, text only in the paper), full MNIST rasters, learnt $w_0$, the ODR network |

## Modules

| file | content |
|---|---|
| `style.py` | sizes, fonts, colours and line styles shared by every figure (`style.NETWORKS`: one style per network); `S.save(fig, name)` |
| `shared_plots.py` | drawing helpers first written for the Cosyne 2027 abstract: phase-plane pair, MNIST example traces, the ODR trial figure |
| `neuron_model_figure.py` | Fig. 1 (also runs as a script: `python neuron_model_figure.py`) |
| `mnist_figures.py` | Fig. 2 and the MNIST / N-MNIST supplementary figures |
| `odr_figures.py` | Fig. 3 and the ODR network schematic |
| `nav_figures.py` | Fig. 4 |
| `fig1_c.csv` | silicon-neuron recording of Fig. 1B (Loris Mendolia): time (s), input voltage (V), neuron output (V; 1 V = 1 nA) |

Simulated spikes (MNIST rasters, the ODR and navigation trials) are cached in `cache/` with a fingerprint of the weights or code that produced them, and simulated again when those change.
