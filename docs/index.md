---
layout: default
title: GPU Workload Dynamics in Probabilistic Tumour Growth
---

# GPU Workload Dynamics in Probabilistic Tumour-Growth Simulations

This site provides source code, experimental datasets, profiling results,
analysis scripts, figures, and reproducibility information associated with
our research on irregular GPU workloads arising from probabilistic
cellular-automaton simulations of tumour growth.

The repository currently contains experimental material associated with
two complementary studies:

- **Euro-Par 2026:** *Hotspot-Induced GPU Imbalance in Irregular
  Probabilistic Workloads*.
- **DataMod 2026:** *Predicting Computational Hotspots from
  Spatio-Temporal Dynamics in Probabilistic Tumor Growth Models*
  (submitted article).

## Authors

- Manuel I. Capel — University of Granada
- Alberto Salguero-Hidalgo — Universidad de Málaga

## Research overview

The underlying application is a stochastic cellular-automaton model of
tumour growth executed on GPUs. As the tumour evolves, its spatial
distribution changes substantially, producing an irregular and
time-dependent computational workload.

Our Euro-Par 2026 study focused on how these spatial dynamics affect GPU
load imbalance and the relative performance of alternative workload
distribution strategies. The subsequent DataMod 2026 work extends this
analysis by treating the evolving simulation state as a source of data
from which future computational behaviour can be characterised and
predicted.

The resulting research workflow is:

**Tumour simulation → spatio-temporal state → computational workload →
predictive model → workload prediction**

## Euro-Par 2026: GPU imbalance and workload distribution

The Euro-Par 2026 experiments compare three spatial workload mappings:

- **CRM:** Contiguous Row Mapping
- **FRM:** Fine-Grained Regional Mapping
- **HWD:** Hybrid Workload Dispersion

A central observation is that the relative FRM overhead follows a
non-monotonic temporal pattern. It reaches a maximum at an intermediate
stage of tumour growth and subsequently decreases, even though tumour
size and absolute transition-kernel execution time continue to increase.

### Main Euro-Par results

- Maximum GPU speedup over the sequential implementation: **218×**
- Transition kernel contribution to runtime: approximately **81%**
- Pearson correlation between tumour size and kernel time: **> 0.99**
- Maximum smoothed FRM/CRM ratio: **4.53**
- Iteration at maximum ratio: **1987**
- Tumour cells at maximum ratio: **38,491**

![Temporal evolution of the FRM/CRM ratio](assets/images/frm_crm_ratio_peak.png)

## DataMod 2026: spatio-temporal workload prediction

The DataMod 2026 study extends the previous profiling analysis by
investigating whether the current tumour state contains information about
future GPU computational behaviour.

In addition to tumour-cell count \(N(t)\), the analysis considers spatial
descriptors including:

- hotspot gap \(H(t)\);
- regional workload variability \(CV_{\mathrm{reg}}(t)\);
- active-region fraction \(f_{\mathrm{act}}(t)\); and
- regional tumour-density information.

The new material published in this repository includes the instrumented
CUDA implementation, fine-grained measurements, analysis scripts, and
results used in the DataMod 2026 submitted article.

### Experiment 2 — Fine-grained temporal analysis

A complete 150-day stochastic simulation was instrumented every
10 iterations, producing **360 fine-grained observations**. The
\(1024\times1024\) cellular-automaton lattice is analysed using an
\(8\times8\) spatial decomposition comprising 64 regions.

The published material includes:

- fine-grained tumour-state and transition-kernel measurements;
- temporal and lagged-correlation analyses;
- direct future-cost prediction;
- size-residualised future-cost analysis;
- future workload-increment prediction;
- fold-level results and summary tables; and
- figures used to analyse temporal workload evolution.

Prediction horizons of **20, 50, and 120 iterations** are evaluated using
temporally separated validation.

### Experiment 3 — Generalisation across stochastic realisations

To determine whether the relationships identified in a single trajectory
generalise, the DataMod 2026 study includes **10 independent stochastic
realisations**, corresponding to seeds **1001–1010**.

Each realisation contains 360 fine-grained observations, giving a combined
experimental dataset of **3,600 observations**.

Cross-realisation validation is performed using:

- **5-fold GroupKFold**, with complete stochastic realisations used as
  groups; and
- **Leave-One-Seed-Out (LOSO)** validation, in which models are trained on
  nine realisations and evaluated on the remaining unseen realisation.

The repository contains the ten raw fine-grained datasets together with
fold-level metrics and summary results for these validation experiments.

### Main DataMod 2026 findings

The experiments distinguish two complementary aspects of computational
behaviour:

- **Tumour size primarily determines workload magnitude and overall
  workload growth.**
- **Spatial organisation contains additional information about
  size-independent computational behaviour.**

In LOSO validation, the complete spatial descriptor model predicts the
size-independent component of future transition-kernel cost with mean
\(R^2\) values between **0.510 and 0.568**, reducing MAE by approximately
**36.6–40.6%** relative to the zero-residual baseline.

Future workload increments also generalise across unseen stochastic
realisations. Relative to persistence, MAE reductions increase with the
prediction horizon:

| Prediction horizon | MAE reduction |
|-------------------:|--------------:|
| 20 iterations      | 5.7% |
| 50 iterations      | 21.0% |
| 120 iterations     | 48.5% |

The workload-increment models improve upon persistence in **all 10
held-out stochastic realisations** at all three prediction horizons.

These results indicate that the evolving biomedical model state can be
used not only to describe tumour evolution, but also to anticipate the
computational behaviour of the simulation.

## Reproducibility material

The repository provides the experimental artefacts required to reproduce
the principal analyses, including:

- CUDA implementations of the probabilistic tumour-growth model;
- the multi-seed DataMod 2026 simulation implementation;
- the script used to execute the independent stochastic realisations;
- raw fine-grained measurements for seeds 1001–1010;
- Experiment 2 temporal-analysis scripts;
- Experiment 3 grouped and leave-one-seed-out validation scripts;
- fold-level GroupKFold and LOSO metrics;
- statistical summary tables; and
- figures and intermediate analysis results used in the submitted study.

In particular, the DataMod 2026 material can be found under:

```text
Cuda/SOURCES/crecimientoTumoralDataMod_seeds.cu
Cuda/BIN/ct_10_seeds.sh

DataMod2026/
├── scripts/
│   ├── analyze_experiment2.py
│   ├── analyze_experiment2_prediction.py
│   ├── analyze_experiment2_residual.py
│   ├── analyze_experiment3_multiseed.py
│   └── analyze_experiment3_README.txt
│
├── results/
│   ├── run001/analysis/
│   └── multiseed/
│       ├── seed_1001/
│       ├── ...
│       ├── seed_1010/
│       ├── groupkfold_metrics.csv
│       ├── loso_metrics.csv
│       ├── summary_groupkfold.csv
│       ├── summary_loso.csv
│       └── loso_seed_success.csv
│
└── figures/
    └── experiment2/
