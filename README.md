# BACI-VI-Bench: BACI-derived benchmark data for VI and MARL trade-network equilibrium

**Version:** v2  
**Prepared:** 2026-05-04  
**Authors:** Pham Thanh Hieu  
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
BACI-VI-Bench_Zenodo_v0.1/
├── README.md
├── ZENODO_METADATA.md
├── CITATION.cff
├── LICENSE_CODE_MIT.txt
├── code/
│   ├── build_baci_vi_bench.py
│   ├── build_sector_instances.py
│   ├── trade_vi_instance.py
│   ├── validate_baci_vi_bench.py
│   ├── requirements.txt
│   └── original_saise_baci_local_hs17.py
├── raw_source_manifest/
│   ├── baci_source_manifest.csv
│   ├── baci_variables.md
│   └── source_license.md
├── metadata/
│   ├── benchmark_config_default.json
│   ├── hs2_to_sector_mapping.csv
│   ├── instance_schema.json
│   ├── processed_instance_manifest.csv
│   └── file_index_sha256.csv
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

Replace `/path/to/BACI_HS17_V202601` with the folder on your machine that contains the CEPII-BACI annual CSV files (e.g. `BACI_HS17_V202601_Y2022.csv`).

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
