
## 5. Página de profiling


---
layout: default
title: Profiling
---

# Kernel Profiling and Temporal Analysis

## Nsight Systems

The following profiles were collected:

crm_3090.qdrep
frm_3090.qdrep
crm_3090.sqlite
frm_3090.sqlite

Nsight Systems was used to analyse:

kernel execution time;
temporal kernel evolution;
CUDA API activity;
memory-transfer activity.

1) Kernel execution times
CRM:

| Kernel              | Accumulated time |
| ------------------- | ---------------: |
| `funcionTransicion` |          16.07 s |
| `actualizarVecinos` |           3.01 s |
| `comprobarMadre`    |          18.8 ms |

FRM:
| Kernel              | Accumulated time |
| ------------------- | ---------------: |
| `funcionTransicion` |          55.24 s |
| `actualizarVecinos` |           3.01 s |
| `comprobarMadre`    |          18.8 ms |

##Therefore, the performance difference is therefore concentrated almost entirely in
funcionTransicion.

Temporal evolution
-------------------------
The transition-kernel time is strongly correlated with tumour size:
| Mapping | Pearson correlation | Spearman correlation |
| ------- | ------------------: | -------------------: |
| CRM     |              0.9897 |               0.9967 |
| FRM     |              0.9945 |               0.9963 |

Conclusion: The underlying microarchitectural cause cannot be determined from Nsight
Systems alone and remains a subject for future Nsight Compute analysis.

