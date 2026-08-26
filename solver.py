from typing import Any

import gurobipy as gp
import numpy as np
import torch
from torch import Tensor


class GurobiTransportSolver:
    """Exact balanced transport with Gurobi and LP subgradients."""

    def __init__(
        self,
        *,
        atol: float = 1e-7,
        rtol: float = 1e-6,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if atol < 0.0 or rtol < 0.0:
            raise ValueError("Transport tolerances must be non-negative")
        self.atol = atol
        self.rtol = rtol
        self._env = gp.Env(params={"OutputFlag": 0, **(parameters or {})})

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        self._validate(cost, supply, demand)
        n, m = cost.shape
        cost_np = cost.detach().double().cpu().numpy()
        supply_np = supply.detach().double().cpu().numpy()
        demand_np = demand.detach().double().cpu().numpy()

        with gp.Model(env=self._env) as model:
            plan_variable = model.addMVar((n, m), lb=0.0)
            model.setObjective((cost_np * plan_variable).sum(), gp.GRB.MINIMIZE)
            row_constraints = model.addConstr(plan_variable.sum(axis=1) == supply_np)
            column_constraints = (
                model.addConstr(plan_variable.sum(axis=0)[:-1] == demand_np[:-1])
                if m > 1
                else None
            )
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Gurobi failed to solve balanced transport: {model.Status}")

            plan_np = np.asarray(plan_variable.X)
            row_dual_np = np.asarray(row_constraints.Pi)
            column_dual_np = np.zeros(m)
            if column_constraints is not None:
                column_dual_np[:-1] = np.asarray(column_constraints.Pi)
            objective = model.ObjVal

        plan = torch.as_tensor(plan_np, device=cost.device, dtype=cost.dtype)
        row_dual = torch.as_tensor(row_dual_np, device=cost.device, dtype=cost.dtype)
        column_dual = torch.as_tensor(column_dual_np, device=cost.device, dtype=cost.dtype)
        value = torch.as_tensor(objective, device=cost.device, dtype=cost.dtype)
        return (
            value
            + torch.sum((cost - cost.detach()) * plan)
            + torch.sum((supply - supply.detach()) * row_dual)
            + torch.sum((demand - demand.detach()) * column_dual)
        )

    def close(self) -> None:
        self._env.close()

    def _validate(self, cost: Tensor, supply: Tensor, demand: Tensor) -> None:
        if cost.ndim != 2:
            raise ValueError(f"Expected a cost matrix, found shape {cost.shape}")
        n, m = cost.shape
        if supply.shape != (n,) or demand.shape != (m,):
            raise ValueError(
                f"Expected marginal shapes {(n,)} and {(m,)}, "
                f"found {supply.shape} and {demand.shape}"
            )
        if cost.device != supply.device or cost.device != demand.device:
            raise ValueError("Transport costs and marginals must be on the same device")
        if cost.dtype != supply.dtype or cost.dtype != demand.dtype:
            raise ValueError("Transport costs and marginals must have the same dtype")
        if not cost.is_floating_point():
            raise ValueError("Transport inputs must be floating-point tensors")
        self._validate_nonnegative(cost, "transport costs")
        self._validate_probability(supply, "transport supply")
        self._validate_probability(demand, "transport demand")

    @staticmethod
    def _validate_nonnegative(value: Tensor, name: str) -> None:
        if not torch.all(torch.isfinite(value)).item():
            raise ValueError(f"{name.capitalize()} must be finite")
        if torch.any(value < 0.0).item():
            raise ValueError(f"{name.capitalize()} must be non-negative")

    def _validate_probability(self, value: Tensor, name: str) -> None:
        self._validate_nonnegative(value, name)
        one = torch.ones((), device=value.device, dtype=value.dtype)
        if not torch.isclose(value.sum(), one, atol=self.atol, rtol=self.rtol).item():
            raise ValueError(f"{name.capitalize()} must sum to one")
