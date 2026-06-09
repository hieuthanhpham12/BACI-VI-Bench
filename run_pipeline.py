#!/usr/bin/env python3
"""
BACI-VI-Bench v0.3 — local build, validation, and package-synchronisation pipeline
================================================================================

Run this script from the BACI-VI-Bench_Zenodo_v0.3/ folder, or pass
--output-dir explicitly. It rebuilds the NPZ instances, normalises the year-level
sector order, validates the package, updates reproducibility metadata, and then
recomputes a package-wide SHA-256 file index.

Usage
-----
    cd BACI-VI-Bench_Zenodo_v0.3
    python run_pipeline.py --baci-dir "D:/BACI_HS17_V202601"

The --baci-dir folder must contain files named, for example:
    BACI_HS17_Y2017_V202601.csv
    BACI_HS17_Y2018_V202601.csv
    ...
    BACI_HS17_Y2024_V202601.csv
    country_codes_V202601.csv    (optional, used for country labels)

Requirements
------------
    pip install numpy pandas matplotlib
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATASET_VERSION = "v0.3"
YEARS = [str(y) for y in range(2017, 2025)]  # 2017–2024
M_EXPORTERS = 10
N_IMPORTERS = 10
K_SECTORS = 5
L_ROUTES = 1
SECTOR_YEAR = "2022"
N_SECTORS = 5

# Canonical order used in the manuscript/README for longitudinal readability.
# The raw top-sector ranking may swap Chemicals and Transport across years;
# this pipeline rewrites year-level NPZ arrays to this common sector order.
CANONICAL_SECTORS = ["Machinery", "Minerals", "Chemicals", "Transport", "Metals"]

EXPECTED_YEAR_INSTANCES = len(YEARS)
EXPECTED_SECTOR_INSTANCES = N_SECTORS


# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], step: str) -> None:
    """Run a subprocess command and raise on failure."""
    print(f"\n{'=' * 78}")
    print(f"  STEP: {step}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'=' * 78}")
    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        sys.exit(f"\n[ERROR] Step '{step}' failed (exit code {result.returncode}).")
    print(f"  Done in {elapsed:.1f}s\n")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default.copy()
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_file(path: Path, message: str) -> None:
    if not path.exists():
        sys.exit(f"[ERROR] {message}: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Repository and data checks
# ──────────────────────────────────────────────────────────────────────────────

def verify_repository_layout(root: Path) -> None:
    """Check the files needed by the pipeline before running expensive steps."""
    required = [
        root / "code" / "build_baci_vi_bench.py",
        root / "code" / "build_sector_instances.py",
        root / "code" / "validate_baci_vi_bench.py",
        root / "code" / "trade_vi_instance.py",
        root / "metadata",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        sys.exit(
            "[ERROR] Repository layout is incomplete. Run from the package root "
            f"or pass --output-dir correctly. Missing:\n{msg}"
        )


def verify_baci_inputs(baci_dir: Path) -> None:
    """Warn early if expected BACI annual files are missing."""
    missing = []
    for year in YEARS:
        candidates = [
            baci_dir / f"BACI_HS17_Y{year}_V202601.csv",
            baci_dir / f"BACI_HS17_Y{year}.csv",
            baci_dir / f"BACI_HS17_Y{year}_V202601.CSV",
            baci_dir / f"BACI_HS17_Y{year}.CSV",
        ]
        if not any(p.exists() for p in candidates):
            missing.append(year)
    if missing:
        sys.exit(
            "[ERROR] Missing BACI annual files for years: "
            + ", ".join(missing)
            + f"\n        Checked folder: {baci_dir}"
        )


def verify_instance_counts(root: Path) -> None:
    year_files = sorted((root / "processed_instances" / "year_instances").glob("*.npz"))
    sector_files = sorted((root / "processed_instances" / "sector_instances").glob("*.npz"))
    if len(year_files) != EXPECTED_YEAR_INSTANCES:
        sys.exit(
            f"[ERROR] Expected {EXPECTED_YEAR_INSTANCES} year-level NPZ files, "
            f"found {len(year_files)}."
        )
    if len(sector_files) != EXPECTED_SECTOR_INSTANCES:
        sys.exit(
            f"[ERROR] Expected {EXPECTED_SECTOR_INSTANCES} sector-level NPZ files, "
            f"found {len(sector_files)}."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Sector-order standardisation
# ──────────────────────────────────────────────────────────────────────────────

def standardize_year_sector_order(root: Path) -> None:
    """
    Reorder year-level NPZ arrays to the canonical sector order.

    This avoids an ambiguity in longitudinal use: the top five sectors are the
    same labels, but the raw ranking may place Chemicals and Transport in a
    different order in different years. Only arrays whose first dimension is the
    sector dimension are reordered.
    """
    import numpy as np

    year_dir = root / "processed_instances" / "year_instances"
    files = sorted(year_dir.glob("*.npz"))
    if not files:
        sys.exit(f"[ERROR] No year-level NPZ files found in {year_dir}")

    changed = 0
    for path in files:
        data = np.load(path, allow_pickle=True)
        arrays: dict[str, Any] = {key: data[key] for key in data.files}
        sectors = [str(x) for x in arrays["sector_names"].tolist()]

        if set(sectors) != set(CANONICAL_SECTORS):
            sys.exit(
                f"[ERROR] {path.name} has sector labels {sectors}, "
                f"but expected exactly {CANONICAL_SECTORS}."
            )

        permutation = [sectors.index(sec) for sec in CANONICAL_SECTORS]
        if permutation == list(range(len(CANONICAL_SECTORS))):
            continue

        k_dim = len(CANONICAL_SECTORS)
        for key, value in list(arrays.items()):
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == k_dim:
                arrays[key] = value[permutation, ...]

        arrays["sector_names"] = np.array(CANONICAL_SECTORS, dtype="U32")
        if "metadata_json" in arrays:
            try:
                meta = json.loads(str(arrays["metadata_json"][0]))
            except Exception:
                meta = {}
            meta["sector_order"] = CANONICAL_SECTORS
            meta["sector_order_note"] = (
                "Year-level NPZ arrays are stored in a canonical sector order "
                "for longitudinal comparison. Country selection remains per-year."
            )
            arrays["metadata_json"] = np.array([json.dumps(meta, ensure_ascii=False)], dtype="U4096")

        np.savez_compressed(path, **arrays)
        changed += 1
        print(f"  Standardized sector order: {relpath(path, root)}")

    print(f"  Sector-order standardisation complete ({changed} file(s) rewritten).")


def verify_sector_policy(root: Path) -> None:
    """Verify that all year-level instances now use the same sector order."""
    import numpy as np

    for path in sorted((root / "processed_instances" / "year_instances").glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        sectors = [str(x) for x in data["sector_names"].tolist()]
        if sectors != CANONICAL_SECTORS:
            sys.exit(
                f"[ERROR] Non-canonical sector order in {path.name}: {sectors}. "
                f"Expected {CANONICAL_SECTORS}."
            )
    print("  Verified fixed sector labels/order across all year-level instances.")


# ──────────────────────────────────────────────────────────────────────────────
# Calibration consistency checks and metadata sync
# ──────────────────────────────────────────────────────────────────────────────

def calibration_errors_for_instance(npz_path: Path) -> dict[str, float | str]:
    """Check NPZ operator arrays against the actual calibration formula used in code."""
    import numpy as np

    d = np.load(npz_path, allow_pickle=True)
    x = d["x_obs"].astype(float)
    a_cost = d["a_cost"].astype(float)
    p_price = d["p_price"].astype(float)
    d_demand = d["d_demand"].astype(float)
    b = float(d["b_congestion"][0])
    tau = float(d["tau_transport"][0])
    uv_exporter_norm = d["uv_exporter_norm"].astype(float)

    expected_a = np.clip(uv_exporter_norm, 0.2, 5.0) * 0.5
    expected_p = np.zeros_like(p_price)
    expected_d = np.zeros_like(d_demand)
    K, m, n, _ = x.shape

    for k in range(K):
        for j in range(n):
            import_total = float(x[k, :, j, :].sum())
            costs = []
            for i in range(m):
                if float(x[k, i, j, :].sum()) > 0.0:
                    export_total = float(x[k, i, :, :].sum())
                    costs.append(expected_a[k, i] * (1.0 + b * export_total) + tau * expected_a[k, i])
            avg_cost = float(np.mean(costs)) if costs else float(expected_a[k].mean())
            d_kj = 0.4 / max(import_total, 0.01)
            expected_p[k, j] = max(avg_cost * (1.0 + d_kj * import_total), avg_cost * 0.8)
            expected_d[k, j] = d_kj

    return {
        "file": npz_path.name,
        "max_abs_err_a_cost": float(np.max(np.abs(a_cost - expected_a))),
        "max_abs_err_p_price": float(np.max(np.abs(p_price - expected_p))),
        "max_abs_err_d_demand": float(np.max(np.abs(d_demand - expected_d))),
    }


def verify_calibration_consistency(root: Path, tol: float = 1e-10) -> None:
    """Write an audit CSV and fail if calibration arrays do not match code formula."""
    files = sorted((root / "processed_instances").glob("**/*.npz"))
    audit_rows = [calibration_errors_for_instance(path) for path in files]
    out_path = root / "metadata" / "calibration_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    bad = [
        row for row in audit_rows
        if max(
            float(row["max_abs_err_a_cost"]),
            float(row["max_abs_err_p_price"]),
            float(row["max_abs_err_d_demand"]),
        ) > tol
    ]
    if bad:
        sys.exit(
            "[ERROR] Calibration consistency check failed. See "
            f"{relpath(out_path, root)}. First failing row: {bad[0]}"
        )
    print(f"  Calibration audit passed for {len(audit_rows)} NPZ files: {relpath(out_path, root)}")


def sync_benchmark_config(root: Path) -> None:
    """Update benchmark_config_default.json with the actual v0.3 formulas."""
    path = root / "metadata" / "benchmark_config_default.json"
    config = load_json(path)
    config["dataset_version"] = DATASET_VERSION
    config["source_dataset"] = "CEPII-BACI HS17 V202601"
    config["years"] = YEARS
    config["default_network"] = {
        "m_exporters": M_EXPORTERS,
        "n_importers": N_IMPORTERS,
        "k_sectors": K_SECTORS,
        "l_routes": L_ROUTES,
        "dimension": M_EXPORTERS * N_IMPORTERS * K_SECTORS * L_ROUTES,
    }
    config["sector_network"] = {
        "year": SECTOR_YEAR,
        "m_exporters": M_EXPORTERS,
        "n_importers": N_IMPORTERS,
        "k_sectors": 1,
        "l_routes": L_ROUTES,
        "dimension": M_EXPORTERS * N_IMPORTERS * L_ROUTES,
    }
    config["sector_policy"] = {
        "year_level_country_selection": "Top 10 exporters and top 10 importers are selected independently for each year by aggregate bilateral trade value.",
        "year_level_sector_labels": CANONICAL_SECTORS,
        "year_level_sector_order": "Fixed canonical order in NPZ files after running run_pipeline.py.",
        "sector_level_scope": f"Five single-sector instances are generated for year {SECTOR_YEAR}.",
    }
    config["cost_operator"] = {
        "operator_form": "F[k,i,j,l] = a_cost[k,i]*(1 + b_congestion*sum_{j,l} x[k,i,j,l]) + tau_transport*a_cost[k,i] - p_price[k,j]/(1 + d_demand[k,j]*sum_{i,l} x[k,i,j,l])",
        "b_congestion": 0.25,
        "tau_transport": 0.05,
        "fixed_defaults_note": "b_congestion and tau_transport are reproducible benchmark defaults, not structurally estimated parameters.",
        "calibration_formulas_used_by_code": {
            "a_cost[k,i]": "0.5 * clip(uv_exporter_norm[k,i], 0.2, 5.0)",
            "import_total[k,j]": "sum_{i,l} x_obs[k,i,j,l]",
            "export_total[k,i]": "sum_{j,l} x_obs[k,i,j,l]",
            "d_demand[k,j]": "0.4 / max(import_total[k,j], 0.01)",
            "avg_cost[k,j]": "mean over active exporters i with x_obs[k,i,j,:].sum()>0 of a_cost[k,i]*(1+b_congestion*export_total[k,i]) + tau_transport*a_cost[k,i]; otherwise mean_i a_cost[k,i]",
            "p_price[k,j]": "max(avg_cost[k,j]*(1+d_demand[k,j]*import_total[k,j]), 0.8*avg_cost[k,j])",
        },
    }
    config["operator_properties"] = {
        "lipschitz_continuity": "F is Lipschitz continuous on compact C under the stored finite parameter arrays.",
        "lipschitz_bound": "L_F <= max_{k,i,j} { b_congestion*a_cost[k,i]*K*n + p_price[k,j]*d_demand[k,j]*K*m }",
        "pseudomonotonicity_note": "The instances are intended as pseudomonotone network-equilibrium benchmarks; strict monotonicity and uniqueness are not guaranteed.",
        "reference_solution_note": "Pre-computed high-accuracy reference solutions are not stored in v0.3; users may approximate them with included solvers and a tight residual tolerance such as G(x)<1e-8.",
    }
    config["feasible_set"] = "nonnegative flows with exporter-wise capacity sum_{k,j,l} x[k,i,j,l] <= supply_cap[i]"
    config["residual"] = "natural projection residual ||x-P_C(x-F(x))||"
    write_json(path, config)
    print(f"  Synced calibration metadata: {relpath(path, root)}")


def sync_instance_schema(root: Path) -> None:
    """Update the schema version and clarify calibrated-parameter meanings."""
    path = root / "metadata" / "instance_schema.json"
    schema = load_json(path)
    schema["dataset_name"] = "BACI-VI-Bench"
    schema["version"] = DATASET_VERSION
    variables = schema.setdefault("npz_variables", {})
    variables["sector_names"] = "Commodity-sector labels. Year-level NPZ files are stored in the canonical order: Machinery, Minerals, Chemicals, Transport, Metals."
    variables["a_cost"] = "Production-cost coefficients calibrated as 0.5*clip(uv_exporter_norm,0.2,5.0), shape (K,m)."
    variables["p_price"] = "Demand-price scale parameters calibrated from observed active-route average costs, shape (K,n)."
    variables["d_demand"] = "Demand-sensitivity parameters calibrated as 0.4/max(import_total,0.01), shape (K,n)."
    schema["reference_solution"] = "No x_ref is stored in v0.3; approximate references can be computed by running included solvers to a tight residual tolerance."
    write_json(path, schema)
    print(f"  Synced instance schema: {relpath(path, root)}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure/table mapping and README sync
# ──────────────────────────────────────────────────────────────────────────────

def mapping_rows() -> list[dict[str, str]]:
    y2022_npz = "processed_instances/year_instances/baci_vi_HS17_Y2022_V202601_m10_n10_K5_L1.npz"
    sector_npz = "processed_instances/sector_instances/*.npz"
    return [
        {
            "item": "Table 1",
            "archive_file": "manuscript table",
            "source_data_files": "Repository directory tree; metadata/file_index_sha256.csv",
            "source_arrays_or_columns": "folder/file names; file sizes; checksums",
            "reproduction_script": "run_pipeline.py",
            "notes": "Repository structure and reuse purposes.",
        },
        {
            "item": "Table 2",
            "archive_file": "manuscript table",
            "source_data_files": "metadata/instance_schema.json; processed_instances/**/*.npz",
            "source_arrays_or_columns": "NPZ variable names, shapes, units, and descriptions",
            "reproduction_script": "code/validate_baci_vi_bench.py",
            "notes": "NPZ variable schema.",
        },
        {
            "item": "Fig1",
            "archive_file": "figures/Fig1_BACI_HS17_Y2022_convergence.png",
            "source_data_files": f"benchmark_outputs/benchmark_BACI_HS17_all_years.csv; {y2022_npz}",
            "source_arrays_or_columns": "iteration, residual/gap columns by solver; 2022 instance arrays",
            "reproduction_script": "code/saise_baci_experiment.py",
            "notes": "Convergence characterization for the 2022 year-level instance.",
        },
        {
            "item": "Fig2",
            "archive_file": "figures/Fig2_BACI_HS17_Y2022_flow_heatmaps.png",
            "source_data_files": y2022_npz,
            "source_arrays_or_columns": "x_obs, sector_names, exporter_names, importer_names",
            "reproduction_script": "code/saise_baci_experiment.py",
            "notes": "Observed-flow heat maps for the 2022 year-level instance.",
        },
        {
            "item": "Fig3",
            "archive_file": "figures/Fig3_BACI_HS17_Y2022_network_graph.png",
            "source_data_files": y2022_npz,
            "source_arrays_or_columns": "x_obs, exporter_names, importer_names",
            "reproduction_script": "code/saise_baci_experiment.py",
            "notes": "Directed weighted trade-network graph for 2022.",
        },
        {
            "item": "Fig4",
            "archive_file": "figures/Fig4_BACI_HS17_multiyear_comparison.png",
            "source_data_files": "benchmark_outputs/benchmark_BACI_HS17_all_years.csv; processed_instances/year_instances/*.npz",
            "source_arrays_or_columns": "year, method, final residual/gap, iteration counts, CPU time",
            "reproduction_script": "code/saise_baci_experiment.py",
            "notes": "Multi-year instance characterization.",
        },
        {
            "item": "Fig5",
            "archive_file": "figures/Fig5_BACI_HS17_Y2022_sector_speedup.png",
            "source_data_files": f"benchmark_outputs/benchmark_BACI_HS17_Y2022_sector.csv; {sector_npz}",
            "source_arrays_or_columns": "sector, method, final residual/gap, iteration counts, CPU time",
            "reproduction_script": "code/saise_baci_experiment.py",
            "notes": "Sector-level characterization for year 2022.",
        },
    ]


def write_figure_table_mapping(root: Path) -> None:
    """Write explicit source mapping requested by reviewers."""
    rows = mapping_rows()
    csv_path = root / "metadata" / "figure_table_source_mapping.csv"
    md_path = root / "docs" / "figure_table_source_mapping.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Figure and table source mapping",
        "",
        "This file records the direct mapping between manuscript figures/tables and the CSV/NPZ files in the Zenodo package.",
        "",
        "| Item | Archive file | Source data files | Source arrays/columns | Reproduction script | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {item} | `{archive_file}` | `{source_data_files}` | {source_arrays_or_columns} | `{reproduction_script}` | {notes} |".format(**row)
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  Wrote source mapping: {relpath(csv_path, root)}")
    print(f"  Wrote source mapping: {relpath(md_path, root)}")


def replace_or_append_section(text: str, heading: str, body: str) -> str:
    """Append a section if it is not already present."""
    if heading in text:
        return text
    return text.rstrip() + "\n\n" + heading + "\n\n" + body.strip() + "\n"


def sync_readme(root: Path) -> None:
    """Patch README version/tree notes and add package-level reviewer-facing details."""
    path = root / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace("BACI-VI-Bench_Zenodo_v0.1/", "BACI-VI-Bench_Zenodo_v0.3/")
    text = text.replace("BACI-VI-Bench_Zenodo_v0.2/", "BACI-VI-Bench_Zenodo_v0.3/")
    text = text.replace(
        "top 10 importers, and top 5 commodity sectors are selected independently\nbased on aggregate trade value in that year. The selected countries and\nsectors may therefore differ across years.",
        "top 10 importers are selected independently based on aggregate trade value\nin that year. The five sector labels are fixed as Machinery, Minerals,\nChemicals, Transport, and Metals, and the NPZ arrays are stored in this\ncanonical order. The selected countries may therefore differ across years,\nbut the sector structure is common across year-level instances.",
    )

    calibration_body = """
The stored v0.3 NPZ files use the calibration implemented in
`code/build_baci_vi_bench.py` and documented in
`metadata/benchmark_config_default.json`:

- `a_cost[k,i] = 0.5 * clip(uv_exporter_norm[k,i], 0.2, 5.0)`;
- `d_demand[k,j] = 0.4 / max(import_total[k,j], 0.01)`;
- `p_price[k,j]` is calibrated from the active-exporter average cost at the
  observed flow and then lower-bounded by `0.8 * avg_cost`;
- `b_congestion = 0.25` and `tau_transport = 0.05` are fixed benchmark
  defaults rather than structurally estimated parameters.

A machine-readable audit of these formulas is written to
`metadata/calibration_audit.csv` by `run_pipeline.py`.
"""
    text = replace_or_append_section(text, "## Operator calibration formulas", calibration_body)

    mapping_body = """
The explicit mapping between manuscript figures/tables and the archive files is
provided in:

- `metadata/figure_table_source_mapping.csv`
- `docs/figure_table_source_mapping.md`

These files identify the CSV/NPZ source files and the relevant arrays or columns
for Fig1--Fig5 and Tables 1--2.
"""
    text = replace_or_append_section(text, "## Figure/table source mapping", mapping_body)

    path.write_text(text, encoding="utf-8")
    print(f"  Synced README: {relpath(path, root)}")


# ──────────────────────────────────────────────────────────────────────────────
# File-index creation and verification
# ──────────────────────────────────────────────────────────────────────────────

def include_in_file_index(path: Path, root: Path) -> bool:
    """Return True if a file should be included in the reproducibility index."""
    if not path.is_file():
        return False
    relative = relpath(path, root)
    excluded_exact = {
        "metadata/file_index_sha256.csv",
    }
    excluded_parts = {"__pycache__", ".pytest_cache"}
    if relative in excluded_exact:
        return False
    if any(part in excluded_parts for part in path.parts):
        return False
    if path.suffix.lower() in {".pyc", ".pyo", ".tmp"}:
        return False
    return True


def update_file_index(root: Path) -> None:
    """Recompute SHA-256 for every package file except the index itself."""
    files = sorted(p for p in root.rglob("*") if include_in_file_index(p, root))
    out_path = root / "metadata" / "file_index_sha256.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for p in files:
            writer.writerow([relpath(p, root), p.stat().st_size, sha256_file(p)])

    print(f"  File index written: {relpath(out_path, root)} ({len(files)} files indexed)")


def verify_file_index(root: Path) -> None:
    """Verify that metadata/file_index_sha256.csv matches the current files."""
    index_path = root / "metadata" / "file_index_sha256.csv"
    require_file(index_path, "Missing file-index CSV")
    with index_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    errors = []
    for row in rows:
        rel = row.get("relative_path") or row.get("file")
        if not rel:
            errors.append("row without relative_path/file column")
            continue
        path = root / rel
        if not path.exists():
            errors.append(f"missing file listed in index: {rel}")
            continue
        expected_size = row.get("size_bytes")
        if expected_size and int(expected_size) != path.stat().st_size:
            errors.append(f"size mismatch: {rel}")
        if sha256_file(path) != row["sha256"]:
            errors.append(f"sha256 mismatch: {rel}")

    indexed = {row.get("relative_path") or row.get("file") for row in rows}
    actual = {relpath(p, root) for p in root.rglob("*") if include_in_file_index(p, root)}
    missing_from_index = sorted(actual - indexed)
    if missing_from_index:
        errors.append("files missing from index: " + ", ".join(missing_from_index[:10]))

    if errors:
        sys.exit("[ERROR] File-index verification failed:\n  - " + "\n  - ".join(errors[:20]))
    print(f"  Verified file index: {len(rows)} files.")


# ──────────────────────────────────────────────────────────────────────────────
# Optional zip creation
# ──────────────────────────────────────────────────────────────────────────────

def create_zip(root: Path, zip_path: Path) -> None:
    """Create a clean zip archive from the repository root."""
    import zipfile

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    base = root.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            arcname = str(Path(base) / path.relative_to(root)).replace("\\", "/")
            zf.write(path, arcname)
    print(f"  Zip archive written: {zip_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="BACI-VI-Bench v0.3 local build/validation/package pipeline.",
        epilog=textwrap.dedent(
            """
            Typical use:
              python run_pipeline.py --baci-dir "D:/BACI_HS17_V202601"

            Fast metadata refresh without rebuilding NPZ files:
              python run_pipeline.py --baci-dir "D:/BACI_HS17_V202601" --skip-build
            """
        ),
    )
    parser.add_argument("--baci-dir", required=True, help="Folder containing BACI HS17 V202601 annual CSV files")
    parser.add_argument("--output-dir", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--skip-build", action="store_true", help="Skip instance building; run post-build checks, validation, docs, and index only")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Rows per chunk when reading BACI CSV")
    parser.add_argument("--skip-baci-input-check", action="store_true", help="Do not pre-check for all annual BACI CSV files")
    parser.add_argument("--make-zip", action="store_true", help="Create a zip archive after successful completion")
    parser.add_argument("--zip-path", default="", help="Optional output zip path; default is ../BACI-VI-Bench_Zenodo_v0.3.zip")
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    baci_dir = Path(args.baci_dir).resolve()
    code_dir = root / "code"
    py = sys.executable

    print("\nBACI-VI-Bench v0.3 — Build/Validation/Package Pipeline")
    print(f"  Repository root : {root}")
    print(f"  BACI source     : {baci_dir}")
    print(f"  Python          : {py}")

    if not baci_dir.exists():
        sys.exit(f"[ERROR] --baci-dir not found: {baci_dir}")
    verify_repository_layout(root)
    if not args.skip_baci_input_check and not args.skip_build:
        verify_baci_inputs(baci_dir)

    if not args.skip_build:
        run(
            [
                py,
                str(code_dir / "build_baci_vi_bench.py"),
                "--baci-dir",
                str(baci_dir),
                "--output-dir",
                str(root),
                "--years",
                *YEARS,
                "--m-exporters",
                str(M_EXPORTERS),
                "--n-importers",
                str(N_IMPORTERS),
                "--k-sectors",
                str(K_SECTORS),
                "--l-routes",
                str(L_ROUTES),
                "--chunk-size",
                str(args.chunk_size),
            ],
            "Build year-level instances (2017–2024, d=500)",
        )

        run(
            [
                py,
                str(code_dir / "build_sector_instances.py"),
                "--baci-dir",
                str(baci_dir),
                "--output-dir",
                str(root),
                "--year",
                SECTOR_YEAR,
                "--n-sectors",
                str(N_SECTORS),
                "--m-exporters",
                str(M_EXPORTERS),
                "--n-importers",
                str(N_IMPORTERS),
                "--chunk-size",
                str(args.chunk_size),
            ],
            f"Build sector-level instances (Y{SECTOR_YEAR}, K=1, d=100)",
        )

    print(f"\n{'=' * 78}")
    print("  STEP: Post-build package synchronisation")
    print(f"{'=' * 78}")
    verify_instance_counts(root)
    standardize_year_sector_order(root)
    verify_sector_policy(root)
    verify_calibration_consistency(root)
    sync_benchmark_config(root)
    sync_instance_schema(root)
    write_figure_table_mapping(root)
    sync_readme(root)

    run(
        [py, str(code_dir / "validate_baci_vi_bench.py"), "--root", str(root), "--tol", "1e-8"],
        "Validate all NPZ instances",
    )

    print(f"\n{'=' * 78}")
    print("  STEP: Recompute and verify package-wide SHA-256 file index")
    print(f"{'=' * 78}")
    update_file_index(root)
    verify_file_index(root)

    if args.make_zip:
        zip_path = Path(args.zip_path).resolve() if args.zip_path else root.parent / f"{root.name}.zip"
        create_zip(root, zip_path)

    print(f"\n{'=' * 78}")
    print("  ALL STEPS COMPLETED SUCCESSFULLY")
    print(f"  Processed instances : {root / 'processed_instances'}")
    print(f"  Validation log      : {root / 'logs' / 'validation_log.txt'}")
    print(f"  Calibration audit   : {root / 'metadata' / 'calibration_audit.csv'}")
    print(f"  Source mapping      : {root / 'metadata' / 'figure_table_source_mapping.csv'}")
    print(f"  File index          : {root / 'metadata' / 'file_index_sha256.csv'}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
