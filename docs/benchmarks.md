
## 4. Página de benchmarks

En `docs/benchmarks.md`:

```markdown
---
layout: default
title: Benchmarks
---

# Benchmark Results
NVIDIA RTX 30390
| Property                  |        Value |
| ------------------------- | -----------: |
| Architecture              | Ampere GA102 |
| Compute capability        |          8.6 |
| Streaming multiprocessors |           82 |
| Memory                    |       24 GiB |
| CUDA version              |         11.2 |
| Lattice size              |  1024 × 1024 |
| Simulation steps          |         3600 |

Execution times:
| GPU        | Mapping | Execution time |
| ---------- | ------- | -------------: |
| RTX 3090   | CRM     |      19,499 ms |
| RTX 3090   | FRM     |      58,697 ms |
| Tesla V100 | CRM     |      37,045 ms |
| Tesla V100 | FRM     |      80,235 ms |

Relative overhead:
| GPU        | FRM / CRM |
| ---------- | --------: |
| RTX 3090   |     3.01× |
| Tesla V100 |     2.17× |

Both implementations produced the same final tumour: 143,972 cells
The instrumented CRM(Cuda/SOURCES/crecimientoTumoral_inst.cu) and FRM (Cuda/SOURCES/crecimientoTumoralDLB_inst.cu) 
executions also generated identical tumour
evolution sequences for all 3600 simulation iterations.

## Experimental platform

The main experiments were conducted on the UGR computing server:

lsicomputing.ugr.es
