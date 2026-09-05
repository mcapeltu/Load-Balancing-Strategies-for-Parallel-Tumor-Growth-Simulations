from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# Paths
# ============================================================

INPUT = Path(
    "DataMod2026/results/run001/fine_grained_metrics.csv"
)

OUT_RESULTS = Path(
    "DataMod2026/results/run001/analysis"
)

OUT_FIGURES = Path(
    "DataMod2026/figures/experiment2"
)

OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT)

print("Shape:", df.shape)
print(df.head())
print(df.tail())

assert len(df) == 360
assert df["iteration"].iloc[0] == 10
assert df["iteration"].iloc[-1] == 3600

# More convenient names
df["N"] = df["tumor_cells"]
df["H"] = df["hotspot_gap"]
df["active_fraction"] = df["active_region_fraction"]
df["T"] = df["transition_ms"]


# ============================================================
# 1. Descriptive statistics
# ============================================================

variables = [
    "N",
    "H",
    "regional_cv",
    "active_fraction",
    "T"
]

desc = df[variables].describe().T
desc.to_csv(OUT_RESULTS / "descriptive_statistics.csv")

print("\nDescriptive statistics")
print(desc)


# ============================================================
# 2. Pearson and Spearman correlations
# ============================================================

pearson = df[variables].corr(method="pearson")
spearman = df[variables].corr(method="spearman")

pearson.to_csv(OUT_RESULTS / "correlations_pearson.csv")
spearman.to_csv(OUT_RESULTS / "correlations_spearman.csv")

print("\nPearson correlations")
print(pearson)

print("\nSpearman correlations")
print(spearman)


# ============================================================
# 3. Temporal evolution
# ============================================================

for var, ylabel in [
    ("N", "Tumour cells N(t)"),
    ("H", "Hotspot descriptor H(t)"),
    ("regional_cv", "Regional CV"),
    ("active_fraction", "Active-region fraction"),
    ("T", "Transition-kernel time (ms)")
]:
    plt.figure(figsize=(7, 4))
    plt.plot(df["iteration"], df[var])
    plt.xlabel("Simulation iteration")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        OUT_FIGURES / f"temporal_{var}.pdf",
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 4. Lagged correlations
# ============================================================

# lag_samples = number of 10-iteration sampling intervals
max_lag_samples = 30       # 300 iterations = 12.5 days

lag_rows = []

predictors = [
    "N",
    "H",
    "regional_cv",
    "active_fraction"
]

for predictor in predictors:

    for lag in range(0, max_lag_samples + 1):

        # X(t) versus T(t + lag)
        x = df[predictor].iloc[:-lag] if lag > 0 else df[predictor]
        y = df["T"].iloc[lag:] if lag > 0 else df["T"]

        r = np.corrcoef(
            x.to_numpy(),
            y.to_numpy()
        )[0, 1]

        lag_rows.append({
            "predictor": predictor,
            "lag_samples": lag,
            "lag_iterations": lag * 10,
            "pearson_r": r
        })

lag_df = pd.DataFrame(lag_rows)

lag_df.to_csv(
    OUT_RESULTS / "lagged_correlations.csv",
    index=False
)

plt.figure(figsize=(7, 4))

for predictor in predictors:
    sub = lag_df[lag_df["predictor"] == predictor]

    plt.plot(
        sub["lag_iterations"],
        sub["pearson_r"],
        label=predictor
    )

plt.xlabel(r"Prediction horizon $\Delta$ (iterations)")
plt.ylabel(
    r"corr$(X(t),T_{\rm transition}(t+\Delta))$"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "lagged_correlations.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. Time-series predictive validation
# ============================================================

# Horizons must be multiples of the 10-iteration sampling interval.
horizons = [20, 50, 120]

models = {
    "N": [
        "N"
    ],
    "N+H": [
        "N",
        "H"
    ],
    "Spatial": [
        "N",
        "H",
        "regional_cv",
        "active_fraction"
    ]
}

results = []


for horizon_iterations in horizons:

    horizon_samples = horizon_iterations // 10

    data = df.copy()

    # Future target
    data["target"] = data["T"].shift(-horizon_samples)

    data = data.dropna().reset_index(drop=True)

    # Chronological cross-validation
    #
    # gap prevents observations very close to the future target
    # from appearing on opposite sides of a train/test boundary.
    tscv = TimeSeriesSplit(
        n_splits=5,
        gap=horizon_samples
    )

    for model_name, features in models.items():

        fold = 0

        for train_idx, test_idx in tscv.split(data):

            fold += 1

            X_train = data.loc[
                train_idx,
                features
            ]

            X_test = data.loc[
                test_idx,
                features
            ]

            y_train = data.loc[
                train_idx,
                "target"
            ]

            y_test = data.loc[
                test_idx,
                "target"
            ]

            # Ridge is preferable here to unregularised
            # multiple regression because the spatial
            # descriptors are correlated.
            model = Pipeline([
                ("scale", StandardScaler()),
                ("regression", Ridge(alpha=1.0))
            ])

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(
                mean_squared_error(y_test, pred)
            )

            results.append({
                "horizon_iterations":
                    horizon_iterations,
                "horizon_samples":
                    horizon_samples,
                "model":
                    model_name,
                "fold":
                    fold,
                "n_train":
                    len(train_idx),
                "n_test":
                    len(test_idx),
                "R2":
                    r2,
                "MAE_ms":
                    mae,
                "RMSE_ms":
                    rmse
            })


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT_RESULTS / "future_prediction_folds.csv",
    index=False
)


# ============================================================
# 6. Summary over temporal folds
# ============================================================

summary = (
    results_df
    .groupby(
        ["horizon_iterations", "model"]
    )
    .agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        MAE_mean=("MAE_ms", "mean"),
        MAE_std=("MAE_ms", "std"),
        RMSE_mean=("RMSE_ms", "mean"),
        RMSE_std=("RMSE_ms", "std")
    )
    .reset_index()
)

summary.to_csv(
    OUT_RESULTS / "future_prediction_summary.csv",
    index=False
)

print("\nFuture prediction")
print(summary.to_string(index=False))


# ============================================================
# 7. Incremental value of spatial information
# ============================================================

pivot = summary.pivot(
    index="horizon_iterations",
    columns="model",
    values="R2_mean"
)

if "N" in pivot.columns:
    if "N+H" in pivot.columns:
        pivot["delta_R2_N_H"] = (
            pivot["N+H"] - pivot["N"]
        )

    if "Spatial" in pivot.columns:
        pivot["delta_R2_spatial"] = (
            pivot["Spatial"] - pivot["N"]
        )

pivot.to_csv(
    OUT_RESULTS / "incremental_spatial_value.csv"
)

print("\nIncremental spatial predictive value")
print(pivot)


# ============================================================
# 8. Predictive-performance figure
# ============================================================

plt.figure(figsize=(7, 4))

for model_name in ["N", "N+H", "Spatial"]:

    sub = summary[
        summary["model"] == model_name
    ]

    plt.plot(
        sub["horizon_iterations"],
        sub["R2_mean"],
        marker="o",
        label=model_name
    )

plt.axhline(0, linewidth=0.8)

plt.xlabel(
    r"Prediction horizon $\Delta$ (iterations)"
)
plt.ylabel(
    r"Mean time-series CV $R^2$"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "future_prediction_r2.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. RESIDUAL FUTURE COST
# ============================================================
#
# Question:
# Does spatial state at time t predict deviations of future
# transition cost from the cost expected from tumour size alone?
#
# IMPORTANT:
# The N -> T baseline is fitted separately inside each training
# fold to avoid temporal leakage.
# ============================================================

residual_results = []

residual_models = {
    "H": [
        "H"
    ],
    "Spatial_residual": [
        "H",
        "regional_cv",
        "active_fraction"
    ]
}

for horizon_iterations in horizons:

    h = horizon_iterations // 10

    # At row i:
    # predictors correspond to t
    # future_N and future_T correspond to t + Delta
    data = df.copy()

    data["future_N"] = df["N"].shift(-h)
    data["future_T"] = df["T"].shift(-h)

    data = data.dropna().reset_index(drop=True)

    tscv = TimeSeriesSplit(
        n_splits=5,
        gap=h
    )

    for fold, (train_idx, test_idx) in enumerate(
        tscv.split(data),
        start=1
    ):

        # ----------------------------------------------------
        # Stage 1:
        # Learn expected future cost from future tumour size
        # using TRAINING DATA ONLY.
        #
        # T(t+Delta) = f(N(t+Delta))
        # ----------------------------------------------------

        size_model = LinearRegression()

        X_size_train = data.loc[
            train_idx,
            ["future_N"]
        ]

        y_size_train = data.loc[
            train_idx,
            "future_T"
        ]

        size_model.fit(
            X_size_train,
            y_size_train
        )

        # Expected cost for train and test
        expected_train = size_model.predict(
            data.loc[train_idx, ["future_N"]]
        )

        expected_test = size_model.predict(
            data.loc[test_idx, ["future_N"]]
        )

        # Future residual:
        # observed cost - expected cost from N alone
        residual_train = (
            data.loc[train_idx, "future_T"].to_numpy()
            - expected_train
        )

        residual_test = (
            data.loc[test_idx, "future_T"].to_numpy()
            - expected_test
        )

        # ----------------------------------------------------
        # Zero-residual baseline
        # ----------------------------------------------------
        zero_pred = np.zeros_like(residual_test)

        baseline_mae = mean_absolute_error(
            residual_test,
            zero_pred
        )

        baseline_rmse = np.sqrt(
            mean_squared_error(
                residual_test,
                zero_pred
            )
        )

        # ----------------------------------------------------
        # Stage 2:
        # Can spatial state at t predict residual at t+Delta?
        # ----------------------------------------------------

        for model_name, features in residual_models.items():

            X_train = data.loc[
                train_idx,
                features
            ]

            X_test = data.loc[
                test_idx,
                features
            ]

            spatial_model = Pipeline([
                ("scale", StandardScaler()),
                ("regression", Ridge(alpha=1.0))
            ])

            spatial_model.fit(
                X_train,
                residual_train
            )

            pred = spatial_model.predict(X_test)

            r2 = r2_score(
                residual_test,
                pred
            )

            mae = mean_absolute_error(
                residual_test,
                pred
            )

            rmse = np.sqrt(
                mean_squared_error(
                    residual_test,
                    pred
                )
            )

            residual_results.append({
                "horizon_iterations":
                    horizon_iterations,
                "model":
                    model_name,
                "fold":
                    fold,
                "R2":
                    r2,
                "MAE_ms":
                    mae,
                "RMSE_ms":
                    rmse,
                "baseline_MAE_ms":
                    baseline_mae,
                "baseline_RMSE_ms":
                    baseline_rmse,
                "MAE_improvement":
                    baseline_mae - mae,
                "RMSE_improvement":
                    baseline_rmse - rmse
            })


residual_df = pd.DataFrame(residual_results)

residual_df.to_csv(
    OUT_RESULTS / "residual_future_cost_folds.csv",
    index=False
)


# Summary
residual_summary = (
    residual_df
    .groupby(
        ["horizon_iterations", "model"]
    )
    .agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),

        MAE_mean=("MAE_ms", "mean"),
        MAE_std=("MAE_ms", "std"),

        RMSE_mean=("RMSE_ms", "mean"),
        RMSE_std=("RMSE_ms", "std"),

        baseline_MAE_mean=(
            "baseline_MAE_ms",
            "mean"
        ),

        baseline_RMSE_mean=(
            "baseline_RMSE_ms",
            "mean"
        ),

        MAE_improvement_mean=(
            "MAE_improvement",
            "mean"
        ),

        RMSE_improvement_mean=(
            "RMSE_improvement",
            "mean"
        )
    )
    .reset_index()
)

residual_summary.to_csv(
    OUT_RESULTS /
    "residual_future_cost_summary.csv",
    index=False
)

print("\nResidual future cost")
print(
    residual_summary.to_string(index=False)
)


print("\nAnalysis completed.")
