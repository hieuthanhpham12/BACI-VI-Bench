# BACI-VI-Bench: BACI-derived benchmark data for VI and MARL trade-network equilibrium

**Version:** v0.3  
**Prepared:** 2026-06-09  
**Authors:** Pham Thanh Hieu and Nguyen Kieu Linh  
**Primary source:** CEPII-BACI HS17 V202601, years 2017--2024

## Purpose

BACI-VI-Bench is a BACI-derived benchmark layer for variational inequality (VI) and multi-agent reinforcement learning (MARL) experiments on international trade-network equilibrium problems.

The original BACI database is not created by this project. BACI-VI-Bench transforms local CEPII-BACI annual trade-flow files into processed VI instances with:

- observed trade-flow tensors `x_obs`,
- exporter-wise capacity sets `C`,
- calibrated trade-network operators `F`,
- natural projection residuals `R(x)=||x-P_C(x-F(x))||`,
- MARL-compatible equilibrium proximity scores `rho_eq(x)=1/(1+R(x))`,
- benchmark solver logs and figures.

## Repository structure

```text
BACI-VI-Bench_Zenodo_v0.3/
├── README.md
├── ZENODO_METADATA.md
├── CITATION.cff
├── LICENSE_CODE_MIT.txt
├── run_pipeline.py                        (end-to-end build + validate runner)
├── code/
│   ├── build_baci_vi_bench.py             (year-level instance builder)
│   ├── build_sector_instances.py          (sector-level instance builder)
│   ├── trade_vi_instance.py               (VI instance loader / evaluator)
│   ├── validate_baci_vi_bench.py          (validation checks)
│   ├── saise_baci_experiment.py           (companion algorithm experiment)
│   ├── requirements.txt
│   └── original_saise_baci_local_hs17.py  (reference original script)
├── raw_source_manifest/
│   ├── baci_source_manifest.csv
│   ├── baci_variables.md
│   └── source_license.md
├── metadata/
│   ├── benchmark_config_default.json      (calibration parameter defaults)
│   ├── calibration_audit.csv             (per-instance calibration audit)
│   ├── figure_table_source_mapping.csv   (figure/table → archive file mapping)
│   ├── hs2_to_sector_mapping.csv
│   ├── instance_schema.json
│   ├── processed_instance_manifest.csv
│   ├── sector_instance_manifest.csv
│   └── file_index_sha256.csv             (SHA-256 checksums of all indexed files)
├── docs/
│   └── figure_table_source_mapping.md    (human-readable figure/table mapping)
├── logs/
│   └── validation_log.txt
├── processed_instances/
│   ├── year_instances/        (8 NPZ files, years 2017-2024, d=500)
│   └── sector_instances/      (5 NPZ files, Y2022 per sector, d=100)
├── benchmark_outputs/
│   ├── benchmark_BACI_HS17_all_years.csv
│   ├── benchmark_BACI_HS17_Y2022_ablation.csv
│   ├── benchmark_BACI_HS17_Y2022_sector.csv
│   └── benchmark_BACI_HS17_Y2022_tuning.csv
└── figures/
    ├── Fig1_BACI_HS17_Y2022_convergence.png
    ├── Fig2_BACI_HS17_Y2022_flow_heatmaps.png
    ├── Fig3_BACI_HS17_Y2022_network_graph.png
    ├── Fig4_BACI_HS17_multiyear_comparison.png
    └── Fig5_BACI_HS17_Y2022_sector_speedup.png
```

## Quick start

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

**Step 1 — Build year-level instances (d=500, years 2017–2024):**

```bash
python code/build_baci_vi_bench.py \
  --baci-dir "/path/to/BACI_HS17_V202601" \
  --output-dir . \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 \
  --m-exporters 10 --n-importers 10 --k-sectors 5 --l-routes 1
```

**Step 2 — Build sector-level instances (d=100, top 5 sectors, year 2022):**

```bash
python code/build_sector_instances.py \
  --baci-dir "/path/to/BACI_HS17_V202601" \
  --output-dir .
```

**Step 3 — Validate all instances:**

```bash
python code/validate_baci_vi_bench.py --root .
```

Replace `/path/to/BACI_HS17_V202601` with the folder on your machine that contains the CEPII-BACI annual CSV files (e.g. `BACI_HS17_Y2022_V202601.csv`).

Expected output files:

```text
processed_instances/year_instances/baci_vi_HS17_Y2017_V202601_m10_n10_K5_L1.npz
...
processed_instances/year_instances/baci_vi_HS17_Y2024_V202601_m10_n10_K5_L1.npz
processed_instances/sector_instances/baci_vi_HS17_Y2022_Machinery_m10_n10_K1_L1.npz
processed_instances/sector_instances/baci_vi_HS17_Y2022_Minerals_m10_n10_K1_L1.npz
processed_instances/sector_instances/baci_vi_HS17_Y2022_Chemicals_m10_n10_K1_L1.npz
processed_instances/sector_instances/baci_vi_HS17_Y2022_Transport_m10_n10_K1_L1.npz
processed_instances/sector_instances/baci_vi_HS17_Y2022_Metals_m10_n10_K1_L1.npz
metadata/processed_instance_manifest.csv
logs/validation_log.txt
```

## VI formulation

Each instance defines a finite-dimensional variational inequality:

\[
\text{Find } x^\ast \in C \text{ such that } \langle F(x^\ast),x-x^\ast\rangle \geq 0,\quad \forall x\in C.
\]

The feasible set is the nonnegative exporter-wise capacity set:

\[
C = \left\{x\ge 0:\sum_{k,j,\ell}x_{kij\ell}\le s_i,\ i=1,\ldots,m\right\}.
\]

The default calibrated operator is:

\[
F_{kij\ell}(x)=a_{ki}\left(1+b\sum_{j,\ell}x_{kij\ell}\right)+\tau a_{ki}-\frac{p_{kj}}{1+d_{kj}\sum_{i,\ell}x_{kij\ell}}.
\]

## Country and sector selection

**Year-level instances** (K=5, d=500): For each year, the top 10 exporters,
top 10 importers are selected independently based on aggregate trade value
in that year. The five sector labels are fixed as Machinery, Minerals,
Chemicals, Transport, and Metals, and the NPZ arrays are stored in this
canonical order. The selected countries may therefore differ across years,
but the sector structure is common across year-level instances.

**Sector-level instances** (K=1, d=100): One sector is specified explicitly
(Machinery, Minerals, Chemicals, Transport, Metals). The top 10 exporters
and importers are selected based on trade within that sector for year 2022.
The five sector names are fixed across the sector instance set.

## Operator properties

Under the calibrated parameter ranges used in BACI-VI-Bench:

- **Lipschitz continuity**: F is Lipschitz continuous on C. A conservative
  bound is L_F ≤ max_{k,i,j} { b·a[k,i]·K·n + p[k,j]·d[k,j]·K·m }.
- **Pseudomonotonicity**: F is pseudomonotone on C under Nagurney (1999)
  network-equilibrium theory. Strict monotonicity is not guaranteed.
- **Solution existence**: Guaranteed by the Hartman–Stampacchia theorem
  (continuous pseudomonotone operator, compact convex feasible set).

Note: A reference equilibrium x_ref is not stored in v0.3 instances.
It may be approximated by running an extragradient or subgradient method
from x_obs until G(x) < chosen tolerance.

## File naming convention

`baci_vi_HS17_Y2022_V202601_m10_n10_K5_L1.npz` means:

- `HS17`: Harmonized System 2017 revision,
- `Y2022`: data year,
- `V202601`: BACI source release,
- `m10`: top 10 exporters,
- `n10`: top 10 importers,
- `K5`: top 5 commodity sectors,
- `L1`: one route, because BACI does not provide route-level data.

## Recommended citation text

Please cite the Zenodo DOI after publication and cite the original BACI source:

- CEPII-BACI database.
- Gaulier, G., & Zignago, S. (2010). BACI: International Trade Database at the Product-Level. CEPII Working Paper No. 2010-23.

## Reproducibility notes

- Raw BACI files are large annual CSV files and are not included in this repository.
- The builder reads the raw files in chunks to reduce memory usage.
- All derived arrays are stored in compressed `.npz` files.
- The code assumes the BACI columns `t`, `i`, `j`, `k`, `v`, `q`.
- `L=1` is used by default because route-level information is unavailable in BACI.

## Operator calibration formulas

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

## Figure/table source mapping

The explicit mapping between manuscript figures/tables and the archive files is
provided in:

- `metadata/figure_table_source_mapping.csv`
- `docs/figure_table_source_mapping.md`

These files identify the CSV/NPZ source files and the relevant arrays or columns
for Fig1--Fig5 and Tables 1--2.
