#!/usr/bin/env python3
"""
Build BACI-VI-Bench processed instances from local CEPII-BACI HS17 CSV files.

This script constructs a BACI-derived benchmark layer for variational inequality
(VI) and multi-agent reinforcement learning (MARL) experiments. It does not
redistribute or replace the original CEPII-BACI database. Users must download
BACI from CEPII and point --baci-dir to the folder containing the annual files.

Expected BACI columns: t, i, j, k, v, q
where v is trade value in thousands of current USD and q is quantity in tonnes.

Example:
    python code/build_baci_vi_bench.py \
        --baci-dir "/path/to/BACI_HS17_V202601" \
        --output-dir . \
        --years 2017 2018 2019 2020 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

BACI_HS = "HS17"
BACI_VERSION = "V202601"
DEFAULT_YEARS = [str(y) for y in range(2017, 2025)]
ROW_CODES = {
    "0", "899", "97", "290", "471", "837", "838", "879", "849", "111",
    "568", "577", "636", "697", "839"
}

HS2_TO_SECTION: Dict[str, str] = {}
_SECTION_MAP = {
    "Animals": list(range(1, 6)),
    "Vegetables": list(range(6, 15)),
    "Fats_Oils": [15],
    "Food_Bev": list(range(16, 25)),
    "Minerals": list(range(25, 28)),
    "Chemicals": list(range(28, 39)),
    "Plastics": [39, 40],
    "Leather": list(range(41, 44)),
    "Wood_Paper": list(range(44, 50)),
    "Textiles": list(range(50, 64)),
    "Footwear": list(range(64, 68)),
    "Stone_Glass": list(range(68, 72)),
    "Precious": [71],
    "Metals": list(range(72, 84)),
    "Machinery": [84, 85],
    "Transport": list(range(86, 90)),
    "Instruments": list(range(90, 93)),
    "Misc_Manuf": list(range(94, 97)),
}
for sec, chapters in _SECTION_MAP.items():
    for ch in chapters:
        HS2_TO_SECTION[f"{ch:02d}"] = sec


def hs2sec(hs2: str) -> str:
    return HS2_TO_SECTION.get(str(hs2).zfill(2), "Other")


def safe_int_code(value) -> str:
    try:
        return str(int(float(str(value).strip())))
    except Exception:
        return str(value).strip()


def try_encodings(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            with path.open("r", encoding=enc) as f:
                f.read(4096)
            return enc
        except UnicodeError:
            continue
    return "latin-1"


def find_baci_file(baci_dir: Path, year: str) -> Path | None:
    stems = [f"BACI_{BACI_HS}_Y{year}_{BACI_VERSION}", f"BACI_{BACI_HS}_Y{year}"]
    for stem in stems:
        for ext in [".csv", ".CSV", ".txt", ""]:
            path = baci_dir / f"{stem}{ext}"
            if path.exists():
                return path
    return None


def find_metadata_file(baci_dir: Path, prefix: str) -> Path | None:
    for stem in [f"{prefix}_{BACI_VERSION}", prefix]:
        for ext in [".csv", ".CSV", ".xlsx"]:
            path = baci_dir / f"{stem}{ext}"
            if path.exists():
                return path
    return None


def load_country_names(baci_dir: Path) -> Dict[str, str]:
    path = find_metadata_file(baci_dir, "country_codes")
    if path is None:
        return {}
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, encoding=try_encodings(path))
    df.columns = [c.strip().lower() for c in df.columns]
    code_col = next((c for c in df.columns if "code" in c and ("country" in c or c == "code")), df.columns[0])
    iso_col = next((c for c in df.columns if "iso" in c or "alpha" in c), None)
    name_col = next((c for c in df.columns if "name" in c or "country" in c), None)
    out = {}
    for _, row in df.iterrows():
        code = safe_int_code(row.get(code_col, ""))
        if not code:
            continue
        if iso_col and str(row.get(iso_col, "")).strip().lower() not in ["", "nan", "none"]:
            out[code] = str(row[iso_col]).strip()[:3].upper()
        elif name_col:
            nm = str(row.get(name_col, "")).strip()
            out[code] = nm[:3].upper() if nm and nm.lower() not in ["nan", "none"] else code
    return out


def read_baci_year(baci_dir: Path, year: str, chunk_size: int = 500_000) -> List[Tuple[str, str, str, float, float]]:
    path = find_baci_file(baci_dir, year)
    if path is None:
        raise FileNotFoundError(f"Cannot find BACI_{BACI_HS}_Y{year}_{BACI_VERSION}.csv in {baci_dir}")
    enc = try_encodings(path)
    trades: List[Tuple[str, str, str, float, float]] = []
    reader = pd.read_csv(
        path,
        dtype={"t": str, "i": str, "j": str, "k": str},
        chunksize=chunk_size,
        encoding=enc,
        low_memory=True,
        on_bad_lines="skip",
    )
    for chunk in reader:
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        required = {"i", "j", "k", "v"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"Missing required BACI columns in {path.name}. Found {list(chunk.columns)}")
        if "t" in chunk.columns:
            chunk = chunk[chunk["t"].astype(str).str.strip() == str(year)]
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            try:
                i = safe_int_code(row["i"])
                j = safe_int_code(row["j"])
                if i == j or i in ROW_CODES or j in ROW_CODES:
                    continue
                v = float(row["v"])
                if v <= 0:
                    continue
                q_raw = row.get("q", np.nan)
                q_str = str(q_raw).strip()
                q = float(q_str) if q_str not in ["", "nan", "NaN", "NA", "None"] else np.nan
                hs6 = str(row["k"]).strip().zfill(6)
                trades.append((i, j, hs6[:2], v, q))
            except Exception:
                continue
    return trades


def aggregate_network(
    trades: List[Tuple[str, str, str, float, float]],
    countries: Dict[str, str],
    year: str,
    m_exporters: int,
    n_importers: int,
    k_sectors: int,
    l_routes: int,
) -> dict:
    from collections import defaultdict

    exp_total = defaultdict(float)
    imp_total = defaultdict(float)
    sec_total = defaultdict(float)
    raw = defaultdict(lambda: [0.0, 0.0])

    for i, j, hs2, v, q in trades:
        sec = hs2sec(hs2)
        if sec == "Other":
            continue
        exp_total[i] += v
        imp_total[j] += v
        sec_total[sec] += v
        raw[(i, j, sec)][0] += v
        if not np.isnan(q) and q > 0:
            raw[(i, j, sec)][1] += q

    exporters = [c for c, _ in sorted(exp_total.items(), key=lambda z: -z[1])[:m_exporters]]
    importers = [c for c, _ in sorted(imp_total.items(), key=lambda z: -z[1])[:n_importers]]
    sectors = [s for s, _ in sorted(sec_total.items(), key=lambda z: -z[1])[:k_sectors]]
    K, m, n, L = len(sectors), len(exporters), len(importers), l_routes

    ei = {c: idx for idx, c in enumerate(exporters)}
    ii = {c: idx for idx, c in enumerate(importers)}
    si = {s: idx for idx, s in enumerate(sectors)}
    flow_v = np.zeros((K, m, n, L), dtype=float)
    flow_q = np.zeros((K, m, n, L), dtype=float)

    for (i, j, sec), (v, q) in raw.items():
        if i in ei and j in ii and sec in si:
            k, a, b = si[sec], ei[i], ii[j]
            flow_v[k, a, b, 0] += v
            if q > 0:
                flow_q[k, a, b, 0] += q

    flow_v_musd = flow_v / 1000.0
    scale = max(float(flow_v_musd.max()), 1.0)
    x_obs = flow_v_musd / scale

    uv_exp = np.ones((K, m), dtype=float)
    uv_imp = np.ones((K, n), dtype=float)
    for k in range(K):
        for i in range(m):
            vv = flow_v[k, i, :, :].sum()
            qq = flow_q[k, i, :, :].sum()
            if qq > 1:
                uv_exp[k, i] = vv / qq
        for j in range(n):
            vv = flow_v[k, :, j, :].sum()
            qq = flow_q[k, :, j, :].sum()
            if qq > 1:
                uv_imp[k, j] = vv / qq
        uv_exp[k] /= max(np.nanmedian(uv_exp[k]), 1e-9)
        uv_imp[k] /= max(np.nanmedian(uv_imp[k]), 1e-9)

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
        "sector_names": np.array(sectors, dtype="U32"),
        "uv_exporter_norm": uv_exp,
        "uv_importer_norm": uv_imp,
        "supply_cap": supply_cap,
        "demand_total": demand_total,
        "normalization_scale_musd": np.array([scale], dtype=float),
        "metadata_json": np.array([json.dumps(metadata, ensure_ascii=False)], dtype="U4096"),
    }


def calibrate_operator(instance: dict, b_congestion: float = 0.25, tau_transport: float = 0.05) -> dict:
    x = instance["x_obs"]
    K, m, n, L = x.shape
    a_cost = np.clip(instance["uv_exporter_norm"], 0.2, 5.0) * 0.5
    p_price = np.zeros((K, n), dtype=float)
    d_demand = np.zeros((K, n), dtype=float)
    for k in range(K):
        for j in range(n):
            import_total = x[k, :, j, :].sum()
            costs = []
            for i in range(m):
                if x[k, i, j, :].sum() > 0:
                    export_total = x[k, i, :, :].sum()
                    costs.append(a_cost[k, i] * (1.0 + b_congestion * export_total) + tau_transport * a_cost[k, i])
            avg_cost = float(np.mean(costs)) if costs else float(a_cost[k].mean())
            d_kj = 0.4 / max(float(import_total), 0.01)
            p_price[k, j] = max(avg_cost * (1.0 + d_kj * import_total), avg_cost * 0.8)
            d_demand[k, j] = d_kj
    instance["a_cost"] = a_cost
    instance["b_congestion"] = np.array([b_congestion], dtype=float)
    instance["tau_transport"] = np.array([tau_transport], dtype=float)
    instance["p_price"] = p_price
    instance["d_demand"] = d_demand
    return instance


def write_instance(instance: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **instance)


def write_hs_mapping(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hs2_chapter", "sector"])
        for hs2 in sorted(HS2_TO_SECTION):
            w.writerow([hs2, HS2_TO_SECTION[hs2]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baci-dir", required=True, help="Folder containing BACI HS17 V202601 annual CSV files")
    parser.add_argument("--output-dir", default=".", help="Repository root or output folder")
    parser.add_argument("--years", nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--m-exporters", type=int, default=10)
    parser.add_argument("--n-importers", type=int, default=10)
    parser.add_argument("--k-sectors", type=int, default=5)
    parser.add_argument("--l-routes", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    baci_dir = Path(args.baci_dir)
    root = Path(args.output_dir)
    year_dir = root / "processed_instances" / "year_instances"
    meta_dir = root / "metadata"
    write_hs_mapping(meta_dir / "hs2_to_sector_mapping.csv")
    countries = load_country_names(baci_dir)

    manifest_rows = []
    for year in args.years:
        print(f"Building year instance {year}...")
        trades = read_baci_year(baci_dir, str(year), chunk_size=args.chunk_size)
        inst = aggregate_network(trades, countries, str(year), args.m_exporters, args.n_importers, args.k_sectors, args.l_routes)
        inst = calibrate_operator(inst)
        fname = f"baci_vi_HS17_Y{year}_V202601_m{args.m_exporters}_n{args.n_importers}_K{args.k_sectors}_L{args.l_routes}.npz"
        out_path = year_dir / fname
        write_instance(inst, out_path)
        manifest_rows.append({
            "instance_id": fname.replace(".npz", ""),
            "year": year,
            "file": str(out_path.relative_to(root)),
            "dimension": int(args.k_sectors * args.m_exporters * args.n_importers * args.l_routes),
            "n_exporters": args.m_exporters,
            "n_importers": args.n_importers,
            "n_sectors": args.k_sectors,
            "n_routes": args.l_routes,
        })

    manifest_path = root / "metadata" / "processed_instance_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
