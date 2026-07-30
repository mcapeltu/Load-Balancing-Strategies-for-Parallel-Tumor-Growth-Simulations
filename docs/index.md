---
layout: default
title: Hotspot-Induced GPU Imbalance
---

# Hotspot-Induced GPU Imbalance in Irregular Probabilistic Workloads

This site provides the source code, benchmark results, profiling data,
figures, and reproducibility information associated with our Euro-Par 2026
poster paper.

## Authors

- Manuel I. Capel — University of Granada
- Alberto Salguero-Hidalgo — Universidad de Málaga

## Overview

This work studies the temporal evolution of GPU workload imbalance in a
probabilistic cellular-automaton model of tumour growth.

Three spatial mappings are evaluated:

- **CRM:** Contiguous Row Mapping
- **FRM:** Fine-Grained Regional Mapping
- **HWD:** Hybrid Workload Dispersion

The main observation is that the relative FRM overhead follows a
non-monotonic temporal pattern: it reaches a maximum at an intermediate
tumour size and subsequently decreases, even though tumour size and absolute
kernel execution time continue to increase.

## Main results

- Maximum GPU speedup over the sequential implementation: **218×**
- Transition kernel contribution to runtime: approximately **81%**
- Pearson correlation between tumour size and kernel time: **> 0.99**
- Maximum smoothed FRM/CRM ratio: **4.53**
- Iteration at maximum ratio: **1987**
- Tumour cells at maximum ratio: **38,491**

![Temporal evolution of the FRM/CRM ratio](assets/images/frm_crm_ratio_peak.png)

## Documentation

- [Implementation and mapping strategies](implementations.md)
- [Benchmark results](benchmarks.md)
- [Profiling methodology](profiling.md)
- [Reproducibility instructions](reproducibility.md)

## Repository

The complete source code and experimental artefacts are available in this
repository.
