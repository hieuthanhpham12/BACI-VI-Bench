#!/usr/bin/env python3
"""
Build sector-level BACI-VI-Bench instances (K=1, d=100) for year 2022.

This script generates one NPZ file per commodity sector for the top 5 sectors
in year 2022, with sector name embedded in the filename.

Example:
    python code/build_sector_instances.py \
        --baci-dir "/path/to/BACI_HS17_V202601" \
        --output-dir . \
        --year 2022 \
        --n-sectors 5 \
        --m-exporters 10 --n-importers 10

Output files (in processed_instances/sector_instances/):
    baci_vi_HS17_Y2022_Machinery_m10_n10_K1_L1.npz
    baci_vi_HS17_Y2022_Minerals_m10_n10_K1_L1.npz
    baci_vi_HS17_Y2022_Chemicals_m10_n10_K1_L1.npz
    baci_vi_HS17_Y2022_Transport_m10_n10_K1_L1.npz
    baci_vi_HS17_Y2022_Metals_m10_n10_K1_L1.npz
    metadata/sector_instance_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reuse shared helpers from build_baci_vi_bench.py
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from build_baci_vi_bench import (  # type: ignore
    HS2_TO_SECTION,
    BACI_HS,
    BACI_VERSION,
    hs2sec,
    load_country_names,
    read_baci_year,
    calibrate_operator,
    write_instance,
)


def aggregate_single_sector(
    trades,
    countries,
    year: str,
    sector_name: str,
    m_exporters: int = 10,
    n_importers: int = 10,
) -> dict:
    """Build a K=1 VI instance restricted to a single named sector."""
    exp_total: dict[str, float] = defaultdict(float)
    imp_total: dict[str, float] = defaultdict(float)
    raw = defaultdict(lambda: [0.0, 0.0])

    # Filter flows to the requested sector only
    for i, j, hs2, v, q in trades:
        sec = hs2sec(hs2)
        if sec != sector_name:
            continue
        exp_total[i] += v
        imp_total[j] += v
        raw[(i, j)][0] += v
        if not np.isnan(q) and q > 0:
            raw[(i, j)][1] += q

    if not exp_total:
        raise ValueError(f"No trade data found for sector '{sector_name}' in year {year}.")

    exporters = [c for c, _ in sorted(exp_total.items(), key=lambda z: -z[1])[:m_exporters]]
    importers = [c for c, _ in sorted(imp_total.items(), key=lambda z: -z[1])[:n_importers]]

    K, m, n, L = 1, len(exporters), len(importers), 1
    ei = {c: idx for idx, c in enumerate(exporters)}
    ii_map = {c: idx for idx, c in enumerate(importers)}

    flow_v = np.zeros((K, m, n, L), dtype=float)
    flow_q = np.zeros((K, m, n, L), dtype=float)

    for (i, j), (v, q) in raw.items():
        if i in ei and j in ii_map:
            flow_v[0, ei[i], ii_map[j], 0] += v
            if q > 0:
                flow_q[0, ei[i], ii_map[j], 0] += q

    flow_v_musd = flow_v / 1000.0
    scale = max(float(flow_v_musd.max()), 1.0)
    x_obs = flow_v_musd / scale

    uv_exp = np.ones((K, m), dtype=float)
    uv_imp = np.ones((K, n), dtype=float)
    for i in range(m):
        vv = flow_v[0, i, :, :].sum()
        qq = flow_q[0, i, :, :].sum()
        if qq > 1:
            uv_exp[0, i] = vv / qq
    for j in range(n):
        vv = flow_v[0, :, j, :].sum()
        qq = flow_q[0, :, j, :].sum()
        if qq > 1:
            uv_imp[0, j] = vv / qq
    uv_exp[0] /= max(float(np.nanmedian(uv_exp[0])), 1e-9)
    uv_imp[0] /= max(float(np.nanmedian(uv_imp[0])), 1e-9)

    supply_cap = x_obs.sum(axis=(0, 2, 3)) * 1.5
    demand_total = x_obs.sum(axis=(0, 1, 3))

    def cname(code: str) -> str:
        return countries.get(str(code), f"C{str(code)[-3:]}").upper()[:3]

    metadata = {
        "dataset_name": "BACI-VI-Bench",
        "source_dataset": "CEPII-BACI",
        "hs_revision": BACI_HS,
        "baci_version": BACI_VERSION,
        "year": str(year),
        "sector": sector_name,
        "m_exporters": m,
        "n_importers": n,
        "k_sectors": K,
        "l_routes": L,
        "dimension": int(K * m * n * L),
        "normalization": "million_USD divided by maximum retained flow value",
        "route_note": "BACI does not include route-level information; default empirical route count is L=1.",
    }

    return {
        "x_obs": x_obs,
        "flow_value_musd": flow_v_musd,
        "flow_quantity_ton": flow_q,
        "exporter_codes": np.array(exporters, dtype="U16"),
        "importer_codes": np.array(importers, dtype="U16"),
        "exporter_names": np.array([cname(c) for c in exporters], dtype="U32"),
        "importer_names": np.array([cname(c) for c in importers], dtype="U32"),
        "sector_names": np.array([sector_name], dtype="U32"),
        "uv_exporter_norm": uv_exp,
        "uv_importer_norm": uv_imp,
        "supply_cap": supply_cap,
        "demand_total": demand_total,
        "normalization_scale_musd": np.array([scale], dtype=float),
        "metadata_json": np.array([json.dumps(metadata, ensure_ascii=False)], dtype="U4096"),
    }


def identify_top_sectors(trades, n: int = 5) -> list[str]:
    """Return the top N sector names by total trade value."""
    sec_total: dict[str, float] = defaultdict(float)
    for _, _, hs2, v, _ in trades:
        sec = hs2sec(hs2)
        if sec != "Other":
            sec_total[sec] += v
    return [s for s, _ in sorted(sec_total.items(), key=lambda z: -z[1])[:n]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build K=1 sector-level VI instances for BACI-VI-Bench."
    )
    parser.add_argument(
        "--baci-dir", required=True,
        help="Folder containing BACI HS17 V202601 annual CSV files"
    )
    parser.add_argument("--output-dir", default=".", help="Repository root folder")
    parser.add_argument("--year", default="2022", help="Year for sector instances (default: 2022)")
    parser.add_argument("--n-sectors", type=int, default=5, help="Number of top sectors (default: 5)")
    parser.add_argument("--m-exporters", type=int, default=10)
    parser.add_argument("--n-importers", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    baci_dir = Path(args.baci_dir)
    root = Path(args.output_dir)
    sector_dir = root / "processed_instances" / "sector_instances"
    sector_dir.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    print(f"Loading BACI data for year {args.year}...")
    trades = read_baci_year(baci_dir, args.year, chunk_size=args.chunk_size)
    countries = load_country_names(baci_dir)

    print(f"Identifying top {args.n_sectors} sectors...")
    top_sectors = identify_top_sectors(trades, n=args.n_sectors)
    print(f"  Top sectors: {top_sectors}")

    manifest_rows = []
    for sector in top_sectors:
        print(f"\nBuilding sector instance: {sector} ...")
        try:
            inst = aggregate_single_sector(
                trades, countries, args.year, sector,
                m_exporters=args.m_exporters,
                n_importers=args.n_importers,
            )
            inst = calibrate_operator(inst)

            fname = (
                f"baci_vi_HS17_Y{args.year}_{sector}"
                f"_m{args.m_exporters}_n{args.n_importers}_K1_L1.npz"
            )
            out_path = sector_dir / fname
            write_instance(inst, out_path)

            dim = int(inst["x_obs"].size)
            print(f"  ✓ Saved: {out_path.relative_to(root)}  (dim={dim})")

            manifest_rows.append({
                "instance_id": fname.replace(".npz", ""),
                "year": args.year,
                "sector": sector,
                "file": str(out_path.relative_to(root)).replace("\\", "/"),
                "dimension": dim,
                "n_exporters": int(inst["x_obs"].shape[1]),
                "n_importers": int(inst["x_obs"].shape[2]),
                "n_sectors": 1,
                "n_routes": 1,
            })
        except ValueError as exc:
            print(f"  ⚠ Skipped {sector}: {exc}")

    # Write sector manifest
    manifest_path = root / "metadata" / "sector_instance_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(manifest_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"\n✓ Manifest written: metadata/sector_instance_manifest.csv")

    print(f"\n{'='*60}")
    print(f"Sector instances built: {len(manifest_rows)} / {len(top_sectors)}")
    print("Run validate_baci_vi_bench.py to verify all instances.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
