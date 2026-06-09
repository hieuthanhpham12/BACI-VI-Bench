#!/usr/bin/env python3
"""
Validate generated BACI-VI-Bench NPZ instances.

Dataset version: v0.3
Revision notes (v0.3): Expanded validation to include nonnegativity check,
capacity-constraint check, projection idempotency check, operator output
dimension check, and structured log output.

Usage
-----
    python code/validate_baci_vi_bench.py --root .

Output
------
  Console: per-instance summary table
  File:    logs/validation_log.txt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow running from any working directory
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from trade_vi_instance import BACIVIInstance  # noqa: E402


# --------------------------------------------------------------------------- #
# Validation helpers                                                           #
# --------------------------------------------------------------------------- #

def check_nonnegativity(inst: BACIVIInstance) -> float:
    """Maximum negative entry in x_obs (0.0 means fully nonneg.)."""
    return float(max(0.0, -inst.x_obs.min()))


def check_capacity(inst: BACIVIInstance, tol: float = 1e-8) -> float:
    """Maximum capacity constraint violation across all exporters."""
    x = inst.x_obs
    viol = 0.0
    for i in range(inst.m):
        total = float(x[:, i, :, :].sum())
        cap   = float(inst.supply_cap[i])
        excess = total - cap
        if excess > tol:
            viol = max(viol, excess)
    return viol


def check_projection_idempotency(inst: BACIVIInstance, tol: float = 1e-8) -> float:
    """
    Verify P_C(P_C(x)) approx= P_C(x).
    Returns max absolute difference; should be < tol for a correct projection.
    """
    x_flat = inst.x_obs.reshape(-1)
    px  = inst.project(x_flat)
    ppx = inst.project(px)
    return float(np.abs(px - ppx).max())


def check_operator_dimension(inst: BACIVIInstance) -> bool:
    """Check that F(x_obs) output has the same shape as x_obs."""
    x_flat = inst.x_obs.reshape(-1)
    Fx = inst.F(x_flat)
    return Fx.shape == x_flat.shape


def check_finite(inst: BACIVIInstance) -> bool:
    """Check that x_obs, F(x_obs), supply_cap contain only finite values."""
    x_flat = inst.x_obs.reshape(-1)
    return (
        np.all(np.isfinite(x_flat))
        and np.all(np.isfinite(inst.F(x_flat)))
        and np.all(np.isfinite(inst.supply_cap))
    )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate BACI-VI-Bench NPZ instances."
    )
    parser.add_argument("--root", default=".", help="Dataset root folder")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Tolerance for idempotency / feasibility checks (default: 1e-8)")
    args = parser.parse_args()

    root = Path(args.root)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "validation_log.txt"

    files = sorted((root / "processed_instances").glob("**/*.npz"))
    if not files:
        raise SystemExit(
            "No .npz instances found under processed_instances/. "
            "Run build_baci_vi_bench.py first."
        )

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"BACI-VI-Bench Validation Log\n"
        f"Generated: {timestamp}\n"
        f"Tolerance: {args.tol}\n"
        f"Instances found: {len(files)}\n"
        + "=" * 100 + "\n"
    )

    col_fmt = "{:<55} {:>6} {:>12} {:>12} {:>12} {:>10} {:>8} {:>8} {:>8}"
    col_head = col_fmt.format(
        "file", "dim",
        "G(x_obs)", "neg_viol", "cap_viol",
        "idem_err", "finite", "F_dim", "PASS"
    )
    sep = "-" * 140

    lines = [header, col_head, sep]
    n_pass = 0
    n_fail = 0

    for path in files:
        rel = str(path.relative_to(root))
        try:
            inst = BACIVIInstance(path)

            g_obs   = inst.residual()
            neg_v   = check_nonnegativity(inst)
            cap_v   = check_capacity(inst, tol=args.tol)
            idem_e  = check_projection_idempotency(inst, tol=args.tol)
            finite  = check_finite(inst)
            f_dim_ok = check_operator_dimension(inst)

            passed = (
                neg_v   < args.tol
                and cap_v   < args.tol
                and idem_e  < args.tol
                and finite
                and f_dim_ok
            )

            status = "OK" if passed else "FAIL"
            if passed:
                n_pass += 1
            else:
                n_fail += 1

            line = col_fmt.format(
                rel[:54], inst.dim,
                f"{g_obs:.4e}", f"{neg_v:.2e}", f"{cap_v:.2e}",
                f"{idem_e:.2e}",
                str(finite), str(f_dim_ok), status
            )
            lines.append(line)
            print(line)

        except Exception as exc:
            n_fail += 1
            err_line = f"  ERROR  {rel}: {exc}"
            lines.append(err_line)
            print(err_line)

    summary = (
        sep + "\n"
        f"Summary: {n_pass}/{len(files)} instances passed all checks.  "
        f"Failed: {n_fail}\n"
    )
    lines.append(summary)
    print(summary)

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation log written to: {log_path.relative_to(root)}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
