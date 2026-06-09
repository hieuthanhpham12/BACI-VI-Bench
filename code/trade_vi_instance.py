"""
Utilities for loading and evaluating BACI-VI-Bench instances.

Dataset version: v0.3
Revision notes (v0.3): Added explicit mathematical documentation for the VI
operator, projection, and residual; added operator-property notes (Lipschitz
continuity, pseudomonotonicity); clarified data-calibrated vs. fixed-default
parameters following peer-review of the companion Data in Brief manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


class BACIVIInstance:
    """
    Finite-dimensional variational inequality (VI) instance derived from
    CEPII-BACI trade-flow data.

    VI problem
    ----------
    Find x* in C such that <F(x*), x - x*> >= 0  for all x in C,

    where:
      - x  is the normalized trade-flow vector of shape (K, m, n, L),
        flattened to R^d with d = K * m * n * L;
      - C  = { x >= 0 : sum_{k,j,l} x[k,i,j,l] <= supply_cap[i], i=1,...,m }
        is the nonnegative exporter-wise capacity set;
      - F  is the calibrated Nagurney-style trade-network operator (see F()).

    Operator properties
    -------------------
    Under the calibrated parameter ranges used in BACI-VI-Bench:
      - F is Lipschitz continuous on C. A conservative Lipschitz bound is
        L_F <= max_{k,i,j} { b * a_cost[k,i] * K * n
                             + p_price[k,j] * d_demand[k,j] * K * m }.
      - F is pseudomonotone on C under standard Nagurney network-equilibrium
        theory (Nagurney 1999; Facchinei & Pang 2003). Strict monotonicity is
        not guaranteed for all calibrated instances.
      - Solution existence on C is guaranteed by the Hartman-Stampacchia
        theorem applied to continuous pseudomonotone operators on compact
        convex sets.

    Note: A reference equilibrium solution x_ref is not stored in v0.3
    instances. It can be approximated by running an extragradient or
    subgradient-type method from x_obs to small residual tolerance.

    Parameter calibration
    ---------------------
    Data-calibrated from BACI (per instance, per sector k, per country i/j):
      a_cost[k, i]   -- exporter/sector production-cost coefficient, derived
                        from normalised unit values (value / quantity ratio);
      p_price[k, j]  -- importer demand-price scale, calibrated so that costs
                        approximately cover observed import prices at x_obs;
      d_demand[k, j] -- demand-price sensitivity, set so that the implied
                        price elasticity at observed import totals equals 0.4.

    Fixed default scalars (same for all instances, may be overridden):
      b_congestion  = 0.25  -- marginal congestion / production-cost slope;
      tau_transport = 0.05  -- transport-cost loading factor.
    """

    def __init__(self, npz_path: str | Path):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)

        # Observed normalized trade-flow vector (shape: K x m x n x L)
        self.x_obs = data["x_obs"].astype(float)

        # Raw flow tensors
        self.flow_value_musd   = data["flow_value_musd"].astype(float)
        self.flow_quantity_ton = data["flow_quantity_ton"].astype(float)

        # Country / sector labels
        self.exporter_codes = data["exporter_codes"]
        self.importer_codes = data["importer_codes"]
        self.exporter_names = data["exporter_names"]
        self.importer_names = data["importer_names"]
        self.sector_names   = data["sector_names"]

        # Feasible set: exporter capacity caps (shape: m)
        self.supply_cap = data["supply_cap"].astype(float)

        # --- VI operator parameters ---
        # Data-calibrated from BACI unit values:
        self.a_cost    = data["a_cost"].astype(float)    # shape: (K, m)
        self.p_price   = data["p_price"].astype(float)   # shape: (K, n)
        self.d_demand  = data["d_demand"].astype(float)  # shape: (K, n)
        # Fixed default scalars (not calibrated from data):
        self.b_congestion  = float(data["b_congestion"][0])   # default 0.25
        self.tau_transport = float(data["tau_transport"][0])  # default 0.05

        self.normalization_scale_musd = float(data["normalization_scale_musd"][0])
        self.metadata: Dict[str, Any] = json.loads(str(data["metadata_json"][0]))
        self.K, self.m, self.n, self.L = self.x_obs.shape
        self.dim = int(self.K * self.m * self.n * self.L)

    def F(self, x_flat: np.ndarray) -> np.ndarray:
        """
        Calibrated trade-network VI operator F(x).

        For each sector k, exporter i, importer j, route l:

            F^k_{i,j,l}(x) = a_cost[k,i] * (1 + b * sum_j' x[k,i,j',l])
                            + tau * a_cost[k,i]
                            - p_price[k,j] / (1 + d_demand[k,j] * sum_i' x[k,i',j,l])

        The first two terms represent marginal production and transport cost
        for exporter i in sector k, increasing in total outgoing flows.
        The third term is the demand price received by importer j, decreasing
        in total incoming flows.

        b   = b_congestion  (fixed default scalar, not calibrated from data)
        tau = tau_transport (fixed default scalar, not calibrated from data)

        Parameters
        ----------
        x_flat : np.ndarray, shape (dim,)

        Returns
        -------
        np.ndarray, shape (dim,) -- operator output F(x)
        """
        x = np.asarray(x_flat, dtype=float).reshape(self.K, self.m, self.n, self.L)
        out = np.zeros_like(x)
        for k in range(self.K):
            for i in range(self.m):
                export_total   = x[k, i, :, :].sum()
                production_cost = self.a_cost[k, i] * (1.0 + self.b_congestion * export_total)
                transport_cost  = self.tau_transport * self.a_cost[k, i]
                for j in range(self.n):
                    import_total = x[k, :, j, :].sum()
                    price = self.p_price[k, j] / (1.0 + self.d_demand[k, j] * import_total)
                    out[k, i, j, :] = production_cost + transport_cost - price
        return out.reshape(-1)

    def project(self, x_flat: np.ndarray) -> np.ndarray:
        """
        Projection P_C(x) onto the feasible set C.

        C = { x >= 0 : sum_{k,j,l} x[k,i,j,l] <= supply_cap[i], i=1,...,m }

        Two-step algorithm:
          1. Truncation:  x <- max(x, 0)          (nonnegativity)
          2. Rescaling:   for each exporter i,
                          if sum > supply_cap[i]:  x[:,i,:,:] *= cap[i] / sum

        Parameters
        ----------
        x_flat : np.ndarray, shape (dim,)

        Returns
        -------
        np.ndarray, shape (dim,) -- projected point in C
        """
        x = np.maximum(np.asarray(x_flat, dtype=float), 0.0).reshape(self.K, self.m, self.n, self.L)
        for i in range(self.m):
            total = x[:, i, :, :].sum()
            cap   = max(float(self.supply_cap[i]), 1e-12)
            if total > cap:
                x[:, i, :, :] *= cap / total
        return x.reshape(-1)

    def residual(self, x_flat: np.ndarray | None = None) -> float:
        """
        Natural projection residual G(x) = || x - P_C(x - F(x)) ||.

        G(x) = 0 if and only if x is a solution of the VI
        (Facchinei & Pang 2003, Proposition 1.5.8).
        Used as the primary benchmark convergence criterion.

        Parameters
        ----------
        x_flat : np.ndarray or None
            Evaluation point. Defaults to x_obs (observed BACI flow vector).

        Returns
        -------
        float -- residual G(x)
        """
        if x_flat is None:
            x_flat = self.x_obs.reshape(-1)
        x_flat = np.asarray(x_flat, dtype=float).reshape(-1)
        return float(np.linalg.norm(x_flat - self.project(x_flat - self.F(x_flat))))

    def equilibrium_proximity(self, x_flat: np.ndarray | None = None) -> float:
        """
        MARL-compatible equilibrium proximity score rho_eq(x) = 1 / (1 + G(x)).

        rho_eq in (0, 1]. Close to 1 => near-equilibrium; close to 0 => large
        residual. Suitable as a reward signal or terminal criterion in
        MARL trade-network environments.
        """
        return float(1.0 / (1.0 + self.residual(x_flat)))


def load_instance(path: str | Path) -> BACIVIInstance:
    """Load a BACI-VI-Bench NPZ instance from disk."""
    return BACIVIInstance(path)
