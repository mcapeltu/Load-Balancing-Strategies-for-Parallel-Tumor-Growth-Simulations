Experiment 3 analysis for DataMod 2026.

This script:
  1) discovers and concatenates the 10 independent-seed CSV files;
  2) creates all_seeds_metrics.csv;
  3) creates future absolute-cost and future cost-increment targets for
     horizons Δ={20,50,120} iterations;
  4) evaluates the predefined models with:
       - GroupKFold(n_splits=5), grouped by complete seed;
       - Leave-One-Seed-Out (LOSO);
  5) evaluates:
       - future absolute transition cost;
       - future residual cost after fold-specific size-cost residualisation;
       - future transition-cost increment;
  6) writes fold-level metrics, predictions, and summary tables.

IMPORTANT:
Residual targets are deliberately NOT precomputed globally.  The size-cost
model used to define residual cost is fitted inside each training fold, so
that no information from held-out seeds leaks into residualisation.

Expected raw columns:
iteration,day,step,tumor_cells,rho_global,rho_max,hotspot_gap,
active_regions,active_region_fraction,regional_cv,transition_ms

Typical repository layout:
DataMod2026/results/multiseed/
    seed_1001/fine_grained_metrics.csv
    ...
    seed_1010/fine_grained_metrics.csv

Flat files such as fine_grained_metrics_1002.csv are also supported.
If exactly one plain fine_grained_metrics.csv is found together with
files 1002..1010, it is interpreted as seed 1001. 
RESULTS are in:
DataMod2026/results/multiseed/
│
├── all_seeds_metrics.csv
│
├── groupkfold_metrics.csv
├── groupkfold_predictions.csv
├── summary_groupkfold.csv
│
├── loso_metrics.csv
├── loso_predictions.csv
├── summary_loso.csv
└── loso_seed_success.csv
