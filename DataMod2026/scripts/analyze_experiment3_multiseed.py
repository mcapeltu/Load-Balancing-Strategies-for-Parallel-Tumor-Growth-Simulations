#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Pre-specified experimental design
# ---------------------------------------------------------------------------

HORIZONS_ITER = (20, 50, 120)
SAMPLE_INTERVAL = 10

ABSOLUTE_MODELS: Dict[str, List[str]] = {
    "N": ["tumor_cells"],
    "N+H": ["tumor_cells", "hotspot_gap"],
    "Spatial": [
        "tumor_cells",
        "hotspot_gap",
        "regional_cv",
        "active_region_fraction",
    ],
}

INCREMENT_MODELS = ABSOLUTE_MODELS

RESIDUAL_MODELS: Dict[str, List[str]] = {
    "H": ["hotspot_gap"],
    "SpatialResidual": [
        "hotspot_gap",
        "regional_cv",
        "active_region_fraction",
    ],
}

EXPECTED_COLUMNS = [
    "iteration",
    "day",
    "step",
    "tumor_cells",
    "rho_global",
    "rho_max",
    "hotspot_gap",
    "active_regions",
    "active_region_fraction",
    "regional_cv",
    "transition_ms",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def extract_seed(path: Path) -> int | None:
    """
    Extract seed from either:
      .../seed_1007/fine_grained_metrics.csv
      .../fine_grained_metrics_1007.csv
      .../fine_grained_metrics_seed_1007.csv
    """
    candidates = [path.stem, path.parent.name]
    patterns = [
        r"(?:seed[_-]?)(\d+)",
        r"fine_grained_metrics[_-](\d+)$",
        r"(\d{4,})$",
    ]

    for text in candidates:
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    return None


def discover_csvs(input_dir: Path) -> List[Tuple[int, Path]]:
    files = sorted(input_dir.rglob("fine_grained_metrics*.csv"))

    # Do not accidentally ingest files produced by this script.
    files = [
        p for p in files
        if p.name not in {
            "all_seeds_metrics.csv",
            "groupkfold_metrics.csv",
            "groupkfold_predictions.csv",
            "loso_metrics.csv",
            "loso_predictions.csv",
            "summary_groupkfold.csv",
            "summary_loso.csv",
        }
    ]

    if not files:
        raise FileNotFoundError(
            f"No fine_grained_metrics*.csv files found under {input_dir}"
        )

    identified: List[Tuple[int, Path]] = []
    ambiguous: List[Path] = []

    for path in files:
        seed = extract_seed(path)
        if seed is None:
            ambiguous.append(path)
        else:
            identified.append((seed, path))

    # Convenience for the uploaded / flat layout used in this experiment:
    # one unlabelled fine_grained_metrics.csv + labelled 1002..1010.
    if len(ambiguous) == 1:
        existing = {s for s, _ in identified}
        if set(range(1002, 1011)).issubset(existing) and 1001 not in existing:
            identified.append((1001, ambiguous[0]))
            ambiguous = []

    if ambiguous:
        names = "\n  ".join(str(p) for p in ambiguous)
        raise ValueError(
            "Could not infer seed from these file paths:\n  "
            + names
            + "\nRename them or place each file under seed_<N>/."
        )

    # Detect duplicate seeds.
    by_seed: Dict[int, List[Path]] = {}
    for seed, path in identified:
        by_seed.setdefault(seed, []).append(path)

    duplicates = {s: ps for s, ps in by_seed.items() if len(ps) > 1}
    if duplicates:
        msg = []
        for s, ps in sorted(duplicates.items()):
            msg.append(f"seed {s}: " + ", ".join(str(p) for p in ps))
        raise ValueError(
            "Multiple raw CSV files were found for the same seed:\n"
            + "\n".join(msg)
        )

    return sorted(identified, key=lambda x: x[0])


def load_and_validate(input_dir: Path) -> pd.DataFrame:
    seed_files = discover_csvs(input_dir)

    frames = []
    for seed, path in seed_files:
        df = pd.read_csv(path)

        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: missing required columns: {missing}"
            )

        df = df[EXPECTED_COLUMNS].copy()
        df.insert(0, "seed", seed)
        df["source_file"] = str(path)

        # Structural checks.
        if df["iteration"].duplicated().any():
            raise ValueError(f"{path}: duplicate iteration values")

        df = df.sort_values("iteration").reset_index(drop=True)

        expected_step = np.diff(df["iteration"].to_numpy())
        if len(expected_step) and not np.all(expected_step == SAMPLE_INTERVAL):
            raise ValueError(
                f"{path}: iterations are not spaced by {SAMPLE_INTERVAL}"
            )

        if df[EXPECTED_COLUMNS].isna().any().any():
            bad = df[EXPECTED_COLUMNS].isna().sum()
            bad = bad[bad > 0].to_dict()
            raise ValueError(f"{path}: missing values detected: {bad}")

        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["seed", "iteration"]).reset_index(drop=True)

    print("Discovered seeds:", sorted(all_df["seed"].unique().tolist()))
    print("Rows per seed:")
    print(all_df.groupby("seed").size().to_string())

    return all_df


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds future T, future N, and ΔT for each horizon.
    Residual targets are NOT created here because they must be
    residualised separately inside each CV training fold.
    """
    out = df.copy()

    for h in HORIZONS_ITER:
        if h % SAMPLE_INTERVAL != 0:
            raise ValueError(
                f"Horizon {h} is not divisible by sample interval "
                f"{SAMPLE_INTERVAL}"
            )

        k = h // SAMPLE_INTERVAL
        g = out.groupby("seed", sort=False)

        out[f"future_transition_ms_d{h}"] = g["transition_ms"].shift(-k)
        out[f"future_tumor_cells_d{h}"] = g["tumor_cells"].shift(-k)
        out[f"delta_transition_ms_d{h}"] = (
            out[f"future_transition_ms_d{h}"] - out["transition_ms"]
        )

    return out


def split_label(test_seeds: Sequence[int]) -> str:
    return ",".join(str(int(s)) for s in sorted(test_seeds))


# ---------------------------------------------------------------------------
# Evaluation: absolute future cost and future increment
# ---------------------------------------------------------------------------

def evaluate_standard_target(
    df_h: pd.DataFrame,
    splitter,
    validation_name: str,
    horizon: int,
    target_kind: str,
    target_col: str,
    models: Dict[str, List[str]],
) -> Tuple[List[dict], List[dict]]:
    rows_metrics: List[dict] = []
    rows_pred: List[dict] = []

    X_dummy = np.zeros((len(df_h), 1))
    groups = df_h["seed"].to_numpy()

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X_dummy, groups=groups), start=1
    ):
        train = df_h.iloc[train_idx]
        test = df_h.iloc[test_idx]

        train_seeds = sorted(train["seed"].unique().tolist())
        test_seeds = sorted(test["seed"].unique().tolist())

        y_train = train[target_col].to_numpy()
        y_test = test[target_col].to_numpy()

        # Operational baseline.
        if target_kind == "absolute":
            # Persistence: future cost ~= current cost.
            baseline_pred = test["transition_ms"].to_numpy()
            baseline_name = "Persistence"
        elif target_kind == "increment":
            # Persistence is ΔT = 0.
            baseline_pred = np.zeros(len(test), dtype=float)
            baseline_name = "Persistence"
        else:
            raise ValueError(target_kind)

        base_m = metrics(y_test, baseline_pred)

        rows_metrics.append(
            {
                "validation": validation_name,
                "target": target_kind,
                "horizon": horizon,
                "fold": fold,
                "train_seeds": split_label(train_seeds),
                "test_seeds": split_label(test_seeds),
                "model": baseline_name,
                **base_m,
                "mae_reduction_vs_baseline_pct": 0.0,
                "rmse_reduction_vs_baseline_pct": 0.0,
            }
        )

        for j, (_, row) in enumerate(test.reset_index(drop=False).iterrows()):
            rows_pred.append(
                {
                    "validation": validation_name,
                    "target": target_kind,
                    "horizon": horizon,
                    "fold": fold,
                    "model": baseline_name,
                    "seed": int(row["seed"]),
                    "iteration": int(row["iteration"]),
                    "y_true": float(y_test[j]),
                    "y_pred": float(baseline_pred[j]),
                }
            )

        for model_name, features in models.items():
            model = ridge_pipeline(alpha=1.0)
            model.fit(train[features], y_train)
            pred = model.predict(test[features])

            m = metrics(y_test, pred)

            mae_red = 100.0 * (base_m["mae"] - m["mae"]) / base_m["mae"]
            rmse_red = 100.0 * (base_m["rmse"] - m["rmse"]) / base_m["rmse"]

            rows_metrics.append(
                {
                    "validation": validation_name,
                    "target": target_kind,
                    "horizon": horizon,
                    "fold": fold,
                    "train_seeds": split_label(train_seeds),
                    "test_seeds": split_label(test_seeds),
                    "model": model_name,
                    **m,
                    "mae_reduction_vs_baseline_pct": mae_red,
                    "rmse_reduction_vs_baseline_pct": rmse_red,
                }
            )

            for j, (_, row) in enumerate(test.reset_index(drop=False).iterrows()):
                rows_pred.append(
                    {
                        "validation": validation_name,
                        "target": target_kind,
                        "horizon": horizon,
                        "fold": fold,
                        "model": model_name,
                        "seed": int(row["seed"]),
                        "iteration": int(row["iteration"]),
                        "y_true": float(y_test[j]),
                        "y_pred": float(pred[j]),
                    }
                )

    return rows_metrics, rows_pred


# ---------------------------------------------------------------------------
# Evaluation: fold-specific residual future cost
# ---------------------------------------------------------------------------

def evaluate_residual_target(
    df_h: pd.DataFrame,
    splitter,
    validation_name: str,
    horizon: int,
) -> Tuple[List[dict], List[dict]]:
    rows_metrics: List[dict] = []
    rows_pred: List[dict] = []

    X_dummy = np.zeros((len(df_h), 1))
    groups = df_h["seed"].to_numpy()

    future_n_col = f"future_tumor_cells_d{horizon}"
    future_t_col = f"future_transition_ms_d{horizon}"

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X_dummy, groups=groups), start=1
    ):
        train = df_h.iloc[train_idx]
        test = df_h.iloc[test_idx]

        train_seeds = sorted(train["seed"].unique().tolist())
        test_seeds = sorted(test["seed"].unique().tolist())

        # -------------------------------------------------------
        # 1. Fit size -> future cost ONLY on training seeds.
        # -------------------------------------------------------
        size_model = ridge_pipeline(alpha=1.0)
        size_model.fit(
            train[[future_n_col]],
            train[future_t_col].to_numpy(),
        )

        train_expected = size_model.predict(train[[future_n_col]])
        test_expected = size_model.predict(test[[future_n_col]])

        residual_train = train[future_t_col].to_numpy() - train_expected
        residual_test = test[future_t_col].to_numpy() - test_expected

        # Zero-residual baseline.
        baseline_pred = np.zeros(len(test), dtype=float)
        base_m = metrics(residual_test, baseline_pred)

        rows_metrics.append(
            {
                "validation": validation_name,
                "target": "residual",
                "horizon": horizon,
                "fold": fold,
                "train_seeds": split_label(train_seeds),
                "test_seeds": split_label(test_seeds),
                "model": "ZeroResidual",
                **base_m,
                "mae_reduction_vs_baseline_pct": 0.0,
                "rmse_reduction_vs_baseline_pct": 0.0,
            }
        )

        for j, (_, row) in enumerate(test.reset_index(drop=False).iterrows()):
            rows_pred.append(
                {
                    "validation": validation_name,
                    "target": "residual",
                    "horizon": horizon,
                    "fold": fold,
                    "model": "ZeroResidual",
                    "seed": int(row["seed"]),
                    "iteration": int(row["iteration"]),
                    "y_true": float(residual_test[j]),
                    "y_pred": 0.0,
                }
            )

        # -------------------------------------------------------
        # 2. Predict fold-specific residual using current spatial state.
        # -------------------------------------------------------
        for model_name, features in RESIDUAL_MODELS.items():
            model = ridge_pipeline(alpha=1.0)
            model.fit(train[features], residual_train)
            pred = model.predict(test[features])

            m = metrics(residual_test, pred)
            mae_red = 100.0 * (base_m["mae"] - m["mae"]) / base_m["mae"]
            rmse_red = 100.0 * (base_m["rmse"] - m["rmse"]) / base_m["rmse"]

            rows_metrics.append(
                {
                    "validation": validation_name,
                    "target": "residual",
                    "horizon": horizon,
                    "fold": fold,
                    "train_seeds": split_label(train_seeds),
                    "test_seeds": split_label(test_seeds),
                    "model": model_name,
                    **m,
                    "mae_reduction_vs_baseline_pct": mae_red,
                    "rmse_reduction_vs_baseline_pct": rmse_red,
                }
            )

            for j, (_, row) in enumerate(test.reset_index(drop=False).iterrows()):
                rows_pred.append(
                    {
                        "validation": validation_name,
                        "target": "residual",
                        "horizon": horizon,
                        "fold": fold,
                        "model": model_name,
                        "seed": int(row["seed"]),
                        "iteration": int(row["iteration"]),
                        "y_true": float(residual_test[j]),
                        "y_pred": float(pred[j]),
                    }
                )

    return rows_metrics, rows_pred


# ---------------------------------------------------------------------------
# Full validation suite
# ---------------------------------------------------------------------------

def run_validation(
    df: pd.DataFrame,
    splitter,
    validation_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics: List[dict] = []
    all_preds: List[dict] = []

    for h in HORIZONS_ITER:
        needed = [
            f"future_transition_ms_d{h}",
            f"future_tumor_cells_d{h}",
            f"delta_transition_ms_d{h}",
        ]
        df_h = df.dropna(subset=needed).reset_index(drop=True)

        # Absolute future cost
        m, p = evaluate_standard_target(
            df_h=df_h,
            splitter=splitter,
            validation_name=validation_name,
            horizon=h,
            target_kind="absolute",
            target_col=f"future_transition_ms_d{h}",
            models=ABSOLUTE_MODELS,
        )
        all_metrics.extend(m)
        all_preds.extend(p)

        # Future residual cost
        m, p = evaluate_residual_target(
            df_h=df_h,
            splitter=splitter,
            validation_name=validation_name,
            horizon=h,
        )
        all_metrics.extend(m)
        all_preds.extend(p)

        # Future cost increment
        m, p = evaluate_standard_target(
            df_h=df_h,
            splitter=splitter,
            validation_name=validation_name,
            horizon=h,
            target_kind="increment",
            target_col=f"delta_transition_ms_d{h}",
            models=INCREMENT_MODELS,
        )
        all_metrics.extend(m)
        all_preds.extend(p)

    return pd.DataFrame(all_metrics), pd.DataFrame(all_preds)


def make_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["validation", "target", "horizon", "model"]

    summary = (
        metrics_df.groupby(group_cols, as_index=False)
        .agg(
            folds=("fold", "nunique"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_reduction_mean_pct=(
                "mae_reduction_vs_baseline_pct",
                "mean",
            ),
            mae_reduction_std_pct=(
                "mae_reduction_vs_baseline_pct",
                "std",
            ),
            rmse_reduction_mean_pct=(
                "rmse_reduction_vs_baseline_pct",
                "mean",
            ),
            rmse_reduction_std_pct=(
                "rmse_reduction_vs_baseline_pct",
                "std",
            ),
        )
        .sort_values(["target", "horizon", "model"])
        .reset_index(drop=True)
    )
    return summary


def make_loso_seed_success_table(
    loso_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    For LOSO, count how many held-out seeds each model improves over its
    operational baseline in MAE.
    """
    x = loso_metrics.copy()

    baseline_by_target = {
        "absolute": "Persistence",
        "increment": "Persistence",
        "residual": "ZeroResidual",
    }

    rows = []
    for (target, horizon), part in x.groupby(["target", "horizon"]):
        baseline_name = baseline_by_target[target]
        base = part[part["model"] == baseline_name][
            ["fold", "test_seeds", "mae"]
        ].rename(columns={"mae": "baseline_mae"})

        for model_name in sorted(part["model"].unique()):
            if model_name == baseline_name:
                continue

            model = part[part["model"] == model_name][
                ["fold", "test_seeds", "mae"]
            ].merge(base, on=["fold", "test_seeds"], how="inner")

            model["improved"] = model["mae"] < model["baseline_mae"]

            rows.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "model": model_name,
                    "seeds_improved": int(model["improved"].sum()),
                    "seeds_total": int(len(model)),
                    "fraction_improved": float(model["improved"].mean()),
                    "mean_mae": float(model["mae"].mean()),
                    "mean_baseline_mae": float(model["baseline_mae"].mean()),
                    "mean_mae_reduction_pct": float(
                        (
                            100.0
                            * (model["baseline_mae"] - model["mae"])
                            / model["baseline_mae"]
                        ).mean()
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target", "horizon", "model"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "DataMod 2026 Experiment 3 multi-seed analysis "
            "(GroupKFold + Leave-One-Seed-Out)"
        )
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../DataMod2026/results/multiseed"),
        help=(
            "Directory containing seed subdirectories or flat "
            "fine_grained_metrics*.csv files."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: same as --input-dir."
        ),
    )
    p.add_argument(
        "--expect-seeds",
        type=int,
        default=10,
        help="Expected number of independent seeds (default: 10).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DataMod 2026 -- Experiment 3 multi-seed analysis")
    print("=" * 72)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    raw = load_and_validate(input_dir)

    n_seeds = raw["seed"].nunique()
    if n_seeds != args.expect_seeds:
        raise ValueError(
            f"Expected {args.expect_seeds} seeds, found {n_seeds}: "
            f"{sorted(raw['seed'].unique().tolist())}"
        )

    # Generate future targets within each seed only.
    data = add_future_targets(raw)

    all_path = output_dir / "all_seeds_metrics.csv"
    data.to_csv(all_path, index=False)
    print(f"\nWrote: {all_path}")
    print(f"Total rows: {len(data)}")

    # -----------------------------------------------------------
    # GroupKFold: complete seeds are the groups.
    # -----------------------------------------------------------
    print("\nRunning GroupKFold(n_splits=5) ...")
    gkf = GroupKFold(n_splits=5)
    gkf_metrics, gkf_preds = run_validation(
        data, gkf, "GroupKFold5"
    )

    gkf_metrics_path = output_dir / "groupkfold_metrics.csv"
    gkf_preds_path = output_dir / "groupkfold_predictions.csv"
    gkf_summary_path = output_dir / "summary_groupkfold.csv"

    gkf_metrics.to_csv(gkf_metrics_path, index=False)
    gkf_preds.to_csv(gkf_preds_path, index=False)
    gkf_summary = make_summary(gkf_metrics)
    gkf_summary.to_csv(gkf_summary_path, index=False)

    # -----------------------------------------------------------
    # Leave-One-Seed-Out.
    # -----------------------------------------------------------
    print("Running Leave-One-Seed-Out ...")
    logo = LeaveOneGroupOut()
    loso_metrics, loso_preds = run_validation(
        data, logo, "LOSO"
    )

    loso_metrics_path = output_dir / "loso_metrics.csv"
    loso_preds_path = output_dir / "loso_predictions.csv"
    loso_summary_path = output_dir / "summary_loso.csv"
    loso_success_path = output_dir / "loso_seed_success.csv"

    loso_metrics.to_csv(loso_metrics_path, index=False)
    loso_preds.to_csv(loso_preds_path, index=False)
    loso_summary = make_summary(loso_metrics)
    loso_summary.to_csv(loso_summary_path, index=False)

    loso_success = make_loso_seed_success_table(loso_metrics)
    loso_success.to_csv(loso_success_path, index=False)

    print("\nGenerated files:")
    for p in [
        all_path,
        gkf_metrics_path,
        gkf_preds_path,
        gkf_summary_path,
        loso_metrics_path,
        loso_preds_path,
        loso_summary_path,
        loso_success_path,
    ]:
        print(f"  {p}")

    # -----------------------------------------------------------
    # Compact console view of the operationally most relevant
    # future-increment results.
    # -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("LOSO summary -- future cost increment")
    print("=" * 72)

    cols = [
        "horizon",
        "model",
        "r2_mean",
        "mae_mean",
        "rmse_mean",
        "mae_reduction_mean_pct",
    ]
    view = loso_summary[
        loso_summary["target"] == "increment"
    ][cols].copy()

    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("LOSO held-out seed success counts")
    print("=" * 72)

    view2 = loso_success[
        loso_success["target"] == "increment"
    ][
        [
            "horizon",
            "model",
            "seeds_improved",
            "seeds_total",
            "mean_mae_reduction_pct",
        ]
    ]
    print(view2.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nAnalysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
