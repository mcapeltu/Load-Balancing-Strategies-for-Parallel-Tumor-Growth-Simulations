
## 6. Página de reproducibilidad

---
layout: default
title: Reproducibility
---

# Reproducing the Experiments

## Connect to the server

ssh -p 2222 USERNAME@lsicomputing.ugr.es

Clone the repository:
git clone https://github.com/mcapeltu/Load-Balancing-Strategies-for-Parallel-Tumor-Growth-Simulations.git
cd Load-Balancing-Strategies-for-Parallel-Tumor-Growth-Simulations

Compile CRM:
nvcc -O3 -arch=sm_86 \
  Cuda/SOURCES/crecimientoTumoral.cu \
  -o crecimientoTumoral

Compile FRM:
nvcc -O3 -arch=sm_86 \
  Cuda/SOURCES/crecimientoTumoralDLB.cu \
  -o crecimientoTumoralDLB

Execute both simulations:
./crecimientoTumoral
./crecimientoTumoralDLB

Profile:
nsys profile \
  --output=crm_3090 \
  ./crecimientoTumoral

nsys profile \
  --output=frm_3090 \
  ./crecimientoTumoralDLB

Note:Exact compilation flags and commands should be adapted to the Makefile and
CUDA environment used on the server: lsicomputing.ugr.es
