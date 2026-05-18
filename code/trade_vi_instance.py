"""Utilities for loading and evaluating BACI-VI-Bench instances."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


class BACIVIInstance:
    """Finite-dimensional VI instance derived from BACI trade-flow data."""

    def __init__(self, npz_path: str | Path):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        self.x_obs = data["x_obs"].astype(float)
        self.flow_value_musd = data["flow_value_musd"].astype(float)
        self.flow_quantity_ton = data["flow_quantity_ton"].astype(float)
        self.exporter_codes = data["exporter_codes"]
        self.importer_codes = data["importer_codes"]
        self.exporter_names = data["exporter_names"]
        self.importer_names = data["importer_names"]
        self.sector_names = data["sector_names"]
        self.supply_cap = data["supply_cap"].astype(float)
        self.a_cost = data["a_cost"].astype(float)
        self.b_congestion = float(data["b_congestion"][0])
        self.tau_transport = float(data["tau_transport"][0])
        self.p_price = data["p_price"].astype(float)
        self.d_demand = data["d_demand"].astype(float)
        self.normalization_scale_musd = float(data["normalization_scale_musd"][0])
        self.metadata: Dict[str, Any] = json.loads(str(data["metadata_json"][0]))
        self.K, self.m, self.n, self.L = self.x_obs.shape
        self.dim = int(self.K * self.m * self.n * self.L)

    def F(self, x_flat: np.ndarray) -> np.ndarray:
        """Calibrated trade-network VI operator F(x)."""
        x = np.asarray(x_flat, dtype=float).reshape(self.K, self.m, self.n, self.L)
        out = np.zeros_like(x)
        for k in range(self.K):
            for i in range(self.m):
                export_total = x[k, i, :, :].sum()
                production_cost = self.a_cost[k, i] * (1.0 + self.b_congestion * export_total)
                transport_cost = self.tau_transport * self.a_cost[k, i]
                for j in range(self.n):
                    import_total = x[k, :, j, :].sum()
                    price = self.p_price[k, j] / (1.0 + self.d_demand[k, j] * import_total)
                    out[k, i, j, :] = production_cost + transport_cost - price
        return out.reshape(-1)

    def project(self, x_flat: np.ndarray) -> np.ndarray:
        """Projection onto nonnegative exporter-wise capacity constraints."""
        x = np.maximum(np.asarray(x_flat, dtype=float), 0.0).reshape(self.K, self.m, self.n, self.L)
        for i in range(self.m):
            total = x[:, i, :, :].sum()
            cap = max(float(self.supply_cap[i]), 1e-12)
            if total > cap:
                x[:, i, :, :] *= cap / total
        return x.reshape(-1)

    def residual(self, x_flat: np.ndarray | None = None) -> float:
        """Natural projection residual ||x - P_C(x - F(x))||."""
        if x_flat is None:
            x_flat = self.x_obs.reshape(-1)
        x_flat = np.asarray(x_flat, dtype=float).reshape(-1)
        return float(np.linalg.norm(x_flat - self.project(x_flat - self.F(x_flat))))

    def equilibrium_proximity(self, x_flat: np.ndarray | None = None) -> float:
        """MARL-compatible proximity score rho_eq = 1/(1+residual)."""
        return float(1.0 / (1.0 + self.residual(x_flat)))


def load_instance(path: str | Path) -> BACIVIInstance:
    return BACIVIInstance(path)
