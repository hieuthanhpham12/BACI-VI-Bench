#!/usr/bin/env python3
"""Validate generated BACI-VI-Bench NPZ instances."""
from __future__ import annotations

import argparse
from pathlib import Path

from trade_vi_instance import BACIVIInstance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Dataset root folder")
    args = parser.parse_args()
    root = Path(args.root)
    files = sorted((root / "processed_instances").glob("**/*.npz"))
    if not files:
        raise SystemExit("No .npz processed instances found. Run build_baci_vi_bench.py first.")
    for path in files:
        inst = BACIVIInstance(path)
        print(f"{path.relative_to(root)} | dim={inst.dim} | residual(x_obs)={inst.residual():.6e} | rho_eq={inst.equilibrium_proximity():.6f}")


if __name__ == "__main__":
    main()
