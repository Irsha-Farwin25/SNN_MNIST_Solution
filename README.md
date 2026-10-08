# IT5092 — Spiking Neural Network for MNIST Digit Classification

An unsupervised spiking neural network (SNN) that learns to recognise handwritten MNIST digits using
**spike-timing-dependent plasticity (STDP)**, built with the [Brian2](https://brian2.readthedocs.io/) simulator.
The design follows Diehl & Cook (2015), *"Unsupervised learning of digit recognition using
spike-timing-dependent plasticity"*, Frontiers in Computational Neuroscience.

Coursework for **IT5092 — Neuroscience & Neurocomputing**.

## Results

Trained on 1,000 MNIST images and tested on 50 (reduced from the full dataset to keep simulation time manageable):

| Metric | Value |
|---|---|
| Training accuracy (running, final) | 63.6% |
| **Test accuracy** | **64.0%** |
| Training time | ~39 min (Brian2 numpy backend, CPU) |

Chance level for 10 classes is 10%. The network learns without ever seeing a label during training;
labels are only used afterwards to name each neuron by the digit it responds to most.

## Network architecture

```
 784 Poisson input neurons  ──(plastic, STDP, all-to-all)──▶  400 excitatory LIF neurons
   (28×28 pixels, rate-coded)                                        │        ▲
                                                           one-to-one│        │ lateral inhibition
                                                                     ▼        │ (all except partner)
                                                                400 inhibitory LIF neurons
```

- **Input encoding** — each pixel becomes a Poisson spike train with rate `pixel / 8 × 2.0` Hz (max ≈ 63.75 Hz).
  Each image is shown for 350 ms, followed by 150 ms of silence so synaptic traces decay.
- **Excitatory neurons** — conductance-based leaky integrate-and-fire (LIF) with an **adaptive threshold θ**
  that rises with every spike. This homeostasis stops a few neurons from dominating and lets
  different neurons specialise on different digits.
- **Inhibitory neurons** — fast LIF neurons that, once triggered by their excitatory partner, suppress all
  other excitatory neurons, producing **winner-take-all** competition.
- **Learning** — trace-based STDP on the input → excitatory synapses (LTP rate 0.01, LTD rate 0.0001),
  weights clipped to [0, 1] and column-normalised before each image.
- **Robustness** — if an image produces fewer than 5 excitatory spikes, it is re-presented with boosted
  input intensity (up to 5 attempts).
- **Classification** — after training, each neuron is assigned to the digit class it fired most for;
  a test image is classified by the class whose assigned neurons fire most on average.

## Repository structure

```
├── IT5092_Assignment_MNIST.ipynb   # Main notebook: build, train, test, visualise, written analysis
├── config.py                       # All hyperparameters (network size, neuron, STDP, encoding, training)
└── utils/
    ├── mnist_loader.py             # Loads MNIST via scikit-learn, sample image display
    ├── metrics.py                  # Neuron label assignment, prediction, accuracy, per-class report
    └── plotting.py                 # Spike rasters, receptive fields, firing rates, training curves
```

## Getting started

**Requirements:** Python 3.9+ and the packages below. No C++ compiler is needed, because Brian2 runs on its numpy backend.

```bash
git clone https://github.com/Irsha-Farwin25/IT5092_SNN_MNIST_Solution.git
cd IT5092_SNN_MNIST_Solution
pip install brian2 scikit-learn matplotlib numpy jupyter
jupyter notebook IT5092_Assignment_MNIST.ipynb
```

Run the cells from top to bottom. MNIST downloads automatically on first use through scikit-learn.

To change the experiment, edit `config.py`, not the notebook. For example, increase `N_TRAIN` / `N_TEST`
for higher accuracy at the cost of runtime, or change `N_EXCITATORY` to resize the network.
Training time scales roughly linearly with `N_TRAIN` (~2.3 s per image on the default settings).

## Notebook outline

0. Brian2 quick introduction (single LIF neuron toy example)
1. Load MNIST and visualise rate encoding
2–3. Build the network: excitatory group, inhibitory group, STDP synapses, fixed Exc↔Inh synapses
4. Training loop with weight normalisation and adaptive input intensity
5. Testing with STDP frozen, plus per-class accuracy report
6. Visualisation: spike raster, firing-rate distribution, learned receptive fields, neuron assignments
7. Written analysis

## Reference

P. U. Diehl and M. Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity,"
*Frontiers in Computational Neuroscience*, vol. 9, 2015. doi:10.3389/fncom.2015.00099
