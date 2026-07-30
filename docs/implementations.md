---
layout: default
title: Implementations
---

# Spatial Mapping Strategies

## Contiguous Row Mapping — CRM

Source file:

CRM uses a one-dimensional CUDA grid composed of:

1024 thread blocks;
1024 threads per block;
one spatial position per CUDA thread;
one pass over the complete 1024 × 1024 lattice.

The mapping preserves spatial locality but may concentrate active tumour
regions in a limited subset of thread blocks.

Cuda/SOURCES/crecimientoTumoral.cu

## Fine-Grained Regional Mapping - FRM
FRM uses:

16 thread blocks;
1024 threads per block;
64 spatial positions processed sequentially by each thread;
64 iterations per complete lattice update.

Cuda/SOURCES/crecimientoTumoralDLB.cu

## Hybrid Workload Dispersion - HDW (non implemented)
HWD combines contiguous-row locality with cyclic spatial dispersion.

Only the assignment between spatial regions and thread blocks changes.
The probabilistic update semantics remain unchanged.


