from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DAY_RE = re.compile(r"(?P<day>\d+)dia\.txt$", re.IGNORECASE)


def parse_snapshot(path: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return (grid_size, rows, cols, ro) from one *dia.txt file."""
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        if not first:
            raise ValueError(f"{path}: empty file")
        grid_size = int(first)

        rows, cols, ros = [], [], []
        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{lineno}: expected 'row col ro', got: {line!r}"
                )
            r, c, ro = int(parts[0]), int(parts[1]), float(parts[2])
            if not (0 <= r < grid_size and 0 <= c < grid_size):
                raise ValueError(
                    f"{path}:{lineno}: coordinate {(r, c)} outside 0..{grid_size-1}"
                )
            rows.append(r)
            cols.append(c)
            ros.append(ro)

    return (
        grid_size,
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(ros, dtype=np.float64),
    )


def perimeter_4n(rows: np.ndarray, cols: np.ndarray) -> int:
    """
    Approximate tumour perimeter in lattice-edge units using 4-neighbour edges.

    P = number of exposed N/S/E/W edges of occupied sites.
    """
    occupied = set(zip(rows.tolist(), cols.tolist()))
    perimeter = 0
    for r, c in occupied:
        perimeter += (r - 1, c) not in occupied
        perimeter += (r + 1, c) not in occupied
        perimeter += (r, c - 1) not in occupied
        perimeter += (r, c + 1) not in occupied
    return int(perimeter)


def regional_statistics(
    rows: np.ndarray,
    cols: np.ndarray,
    grid_size: int,
    n_regions_axis: int,
) -> Dict[str, float]:
    """
    Partition the whole lattice into n_regions_axis x n_regions_axis regions.

    Regional density rho_r = cancer cells in region r / number of lattice sites
    in region r.

    Returns rho_global, rho_max, hotspot measures, active-region fraction,
    entropy and regional imbalance descriptors.
    """
    # array_split gives exact coverage even if grid_size is not divisible by n
    row_edges = np.linspace(0, grid_size, n_regions_axis + 1, dtype=int)
    col_edges = np.linspace(0, grid_size, n_regions_axis + 1, dtype=int)

    # Region index for each tumour cell
    r_idx = np.searchsorted(row_edges[1:-1], rows, side="right")
    c_idx = np.searchsorted(col_edges[1:-1], cols, side="right")

    counts = np.zeros((n_regions_axis, n_regions_axis), dtype=np.int64)
    np.add.at(counts, (r_idx, c_idx), 1)

    region_areas = np.empty_like(counts, dtype=np.int64)
    for i in range(n_regions_axis):
        for j in range(n_regions_axis):
            region_areas[i, j] = (
                (row_edges[i + 1] - row_edges[i])
                * (col_edges[j + 1] - col_edges[j])
            )

    densities = counts / region_areas
    flat_counts = counts.ravel()
    flat_densities = densities.ravel()

    n_cells = int(len(rows))
    n_regions = n_regions_axis**2

    rho_global = n_cells / float(grid_size**2)
    rho_max = float(flat_densities.max()) if n_cells else 0.0
    rho_mean_regions = float(flat_densities.mean())
    rho_std_regions = float(flat_densities.std(ddof=0))

    hotspot_gap = rho_max - rho_global
    hotspot_ratio = (
        rho_max / rho_global if rho_global > 0 else 0.0
    )
    hotspot_excess_norm = (
        hotspot_gap / rho_global if rho_global > 0 else 0.0
    )

    active_mask = flat_counts > 0
    n_active = int(active_mask.sum())
    active_fraction = n_active / n_regions

    # Distribution of tumour cells across regions (not density).
    if n_cells > 0:
        p = flat_counts[flat_counts > 0] / n_cells
        entropy = float(-(p * np.log(p)).sum())
        entropy_norm_all = entropy / math.log(n_regions) if n_regions > 1 else 0.0
        entropy_norm_active = (
            entropy / math.log(n_active) if n_active > 1 else 0.0
        )
        # Herfindahl concentration: 1 = all tumour cells in one region.
        hhi = float(np.sum(p**2))
    else:
        entropy = entropy_norm_all = entropy_norm_active = hhi = 0.0

    # Coefficient of variation across regional densities.
    cv_regions = (
        rho_std_regions / rho_mean_regions if rho_mean_regions > 0 else 0.0
    )

    # A simple regional imbalance index comparable across times.
    # 0 means identical regional densities; larger values mean stronger imbalance.
    imbalance_linf = (
        (rho_max - rho_mean_regions) / rho_mean_regions
        if rho_mean_regions > 0 else 0.0
    )

    hotspot_flat = int(np.argmax(flat_densities)) if n_cells else 0
    hotspot_row_region = hotspot_flat // n_regions_axis
    hotspot_col_region = hotspot_flat % n_regions_axis

    return {
        "regions_axis": n_regions_axis,
        "n_regions": n_regions,
        "active_regions": n_active,
        "active_region_fraction": active_fraction,
        "rho_global": rho_global,
        "rho_region_mean": rho_mean_regions,
        "rho_region_std": rho_std_regions,
        "rho_max": rho_max,
        "hotspot_gap": hotspot_gap,
        "hotspot_ratio": hotspot_ratio,
        "hotspot_excess_norm": hotspot_excess_norm,
        "regional_density_cv": cv_regions,
        "regional_imbalance_linf": imbalance_linf,
        "spatial_entropy": entropy,
        "spatial_entropy_norm_all": entropy_norm_all,
        "spatial_entropy_norm_active": entropy_norm_active,
        "regional_hhi": hhi,
        "hotspot_region_row": hotspot_row_region,
        "hotspot_region_col": hotspot_col_region,
    }


def shape_statistics(
    rows: np.ndarray, cols: np.ndarray, grid_size: int
) -> Dict[str, float]:
    """Compute morphology/shape descriptors from occupied lattice coordinates."""
    n = len(rows)
    if n == 0:
        return {
            "tumour_cells": 0,
            "area_lattice": 0,
            "r_eff": 0.0,
            "centroid_row": np.nan,
            "centroid_col": np.nan,
            "bbox_height": 0,
            "bbox_width": 0,
            "bbox_area": 0,
            "rho_tumour_bbox": 0.0,
            "radius_gyration": 0.0,
            "max_radius_centroid": 0.0,
            "anisotropy": 0.0,
            "perimeter_4n": 0,
            "compactness": 0.0,
        }

    area = int(n)
    r_eff = math.sqrt(area / math.pi)

    cr = float(rows.mean())
    cc = float(cols.mean())

    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    bh = rmax - rmin + 1
    bw = cmax - cmin + 1
    bbox_area = bh * bw
    rho_bbox = area / bbox_area if bbox_area else 0.0

    dr = rows.astype(float) - cr
    dc = cols.astype(float) - cc
    d2 = dr**2 + dc**2
    radius_gyration = float(np.sqrt(np.mean(d2)))
    max_radius = float(np.sqrt(d2.max()))

    # 2D covariance eigenvalue anisotropy:
    # 0 ~ isotropic; approaches 1 as spatial distribution becomes elongated.
    if n >= 2:
        cov = np.cov(np.vstack([rows, cols]), bias=True)
        eig = np.linalg.eigvalsh(cov)
        lam_min, lam_max = float(eig[0]), float(eig[-1])
        anisotropy = (
            (lam_max - lam_min) / (lam_max + lam_min)
            if (lam_max + lam_min) > 0 else 0.0
        )
    else:
        anisotropy = 0.0

    perim = perimeter_4n(rows, cols)
    compactness = (
        4.0 * math.pi * area / (perim**2)
        if perim > 0 else 0.0
    )

    return {
        "tumour_cells": area,
        "area_lattice": area,
        "r_eff": r_eff,
        "centroid_row": cr,
        "centroid_col": cc,
        "centroid_offset_from_grid_center": math.hypot(
            cr - (grid_size - 1) / 2.0,
            cc - (grid_size - 1) / 2.0,
        ),
        "bbox_height": bh,
        "bbox_width": bw,
        "bbox_area": bbox_area,
        "rho_tumour_bbox": rho_bbox,
        "radius_gyration": radius_gyration,
        "max_radius_centroid": max_radius,
        "anisotropy": anisotropy,
        "perimeter_4n": perim,
        "compactness": compactness,
    }


def ro_statistics(ro: np.ndarray) -> Dict[str, float]:
    """
    Descriptors of the stored 'ro' field.

    In the current code the stem cell uses a very large ro (e.g. 100000),
    whereas ordinary tumour cells commonly have much smaller values.
    We do NOT hard-code a biological interpretation; these are raw descriptors.
    """
    if len(ro) == 0:
        return {
            "ro_min": np.nan,
            "ro_mean": np.nan,
            "ro_median": np.nan,
            "ro_max": np.nan,
            "ro_std": np.nan,
            "ro_gt_10_fraction": np.nan,
        }

    return {
        "ro_min": float(np.min(ro)),
        "ro_mean": float(np.mean(ro)),
        "ro_median": float(np.median(ro)),
        "ro_max": float(np.max(ro)),
        "ro_std": float(np.std(ro, ddof=0)),
        "ro_gt_10_fraction": float(np.mean(ro > 10.0)),
    }


def descriptors_for_file(path: Path, n_regions_axis: int) -> Dict[str, float]:
    m = DAY_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract day from filename: {path.name}")
    day = int(m.group("day"))

    grid_size, rows, cols, ro = parse_snapshot(path)

    out: Dict[str, float] = {
        "snapshot_file": path.name,
        "day": day,
        # crecimientoTumoralTiempos.cu runs 24 simulation steps per day.
        # This is the natural key for joining a day snapshot to step-level profiles.
        "iteration_end_of_day": day * 24,
        "simulated_hour_end_of_day": day * 24,
        "grid_size": grid_size,
        "grid_cells_total": grid_size**2,
    }
    out.update(shape_statistics(rows, cols, grid_size))
    out.update(regional_statistics(rows, cols, grid_size, n_regions_axis))
    out.update(ro_statistics(ro))
    return out


def read_optional_profiling(
    path: Path,
    profile_key: str,
) -> pd.DataFrame:
    """
    Read an optional Euro-Par profiling CSV.

    The profiling file must contain `profile_key`, e.g.
      day
    or
      iteration_end_of_day

    All other profiling columns are retained.
    """
    df = pd.read_csv(path)
    if profile_key not in df.columns:
        raise ValueError(
            f"{path} does not contain profiling key {profile_key!r}. "
            f"Columns: {list(df.columns)}"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract spatio-temporal tumour descriptors from Cuda/TABLES/*dia.txt"
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("Cuda/TABLES"),
        help="Directory containing *dia.txt snapshots (default: Cuda/TABLES)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tumour_spatiotemporal_descriptors.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--regions-axis",
        type=int,
        default=8,
        help="Regions per axis; 8 means 8x8=64 regions (default: 8)",
    )
    parser.add_argument(
        "--profiling-csv",
        type=Path,
        default=None,
        help="Optional Euro-Par profiling CSV to merge into the output",
    )
    parser.add_argument(
        "--profile-key",
        default="day",
        choices=["day", "iteration_end_of_day"],
        help="Key used to merge the optional profiling CSV (default: day)",
    )
    args = parser.parse_args()

    if args.regions_axis < 1:
        raise ValueError("--regions-axis must be >= 1")

    files = []
    for p in args.tables_dir.glob("*dia.txt"):
        m = DAY_RE.search(p.name)
        if m:
            files.append((int(m.group("day")), p))

    files.sort(key=lambda x: x[0])

    if not files:
        raise FileNotFoundError(
            f"No *dia.txt snapshots found in {args.tables_dir.resolve()}"
        )

    records = []
    for day, path in files:
        print(f"[{day:3d} d] {path}")
        records.append(descriptors_for_file(path, args.regions_axis))

    df = pd.DataFrame(records).sort_values("day").reset_index(drop=True)

    # Add temporal changes between saved snapshots. These become useful predictors.
    temporal_cols = [
        "tumour_cells",
        "r_eff",
        "rho_tumour_bbox",
        "rho_max",
        "hotspot_gap",
        "hotspot_ratio",
        "hotspot_excess_norm",
        "active_region_fraction",
        "spatial_entropy_norm_all",
        "regional_hhi",
        "regional_density_cv",
        "regional_imbalance_linf",
        "compactness",
        "anisotropy",
    ]
    day_delta = df["day"].diff()
    for col in temporal_cols:
        df[f"d_{col}_per_day"] = df[col].diff() / day_delta

    if args.profiling_csv is not None:
        prof = read_optional_profiling(args.profiling_csv, args.profile_key)
        # Validate one-to-one profiling key where possible.
        if prof[args.profile_key].duplicated().any():
            print(
                f"WARNING: profiling CSV has duplicate {args.profile_key!r} values; "
                "merge may create multiple rows per snapshot."
            )
        df = df.merge(prof, on=args.profile_key, how="left", suffixes=("", "_profile"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print()
    print(f"Wrote {len(df)} snapshots to: {args.output.resolve()}")
    print(f"Columns: {len(df.columns)}")
    print()
    print("Core predictors for DataMod:")
    for c in [
        "day",
        "iteration_end_of_day",
        "tumour_cells",
        "r_eff",
        "rho_tumour_bbox",
        "rho_max",
        "hotspot_gap",
        "hotspot_ratio",
        "active_region_fraction",
        "spatial_entropy_norm_all",
        "regional_hhi",
        "regional_density_cv",
        "regional_imbalance_linf",
        "compactness",
        "anisotropy",
    ]:
        print(f"  - {c}")


if __name__ == "__main__":
    main()

