import math
from functools import lru_cache
from typing import Any, Tuple

import gurobipy as gp
import numpy as np
import torch
from torch import Tensor


class _TransportValue(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        cost: Tensor,
        supply: Tensor,
        demand: Tensor,
        value: Tensor,
        plan: Tensor,
        row_dual: Tensor,
        column_dual: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(plan, row_dual, column_dual)
        return value

    @staticmethod
    def backward(
        ctx: Any, output_gradient: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None, None]:
        plan, row_dual, column_dual = ctx.saved_tensors
        return (
            output_gradient[..., None, None] * plan,
            output_gradient[..., None] * row_dual,
            output_gradient[..., None] * column_dual,
            None,
            None,
            None,
            None,
        )


@lru_cache(maxsize=None)
def _transport_constraints(n: int, m: int) -> np.ndarray:
    num_constraints = n + m - 1
    num_edges = n * m
    constraints = np.zeros((num_constraints, num_edges))
    for i in range(n):
        for j in range(m):
            edge = i * m + j
            constraints[i, edge] = 1.0
            if j < m - 1:
                constraints[n + j, edge] = 1.0
    return constraints

class TorchTransportSolver:
    """Exact batched transport using device-native revised simplex."""

    def __init__(
        self,
        *,
        feasibility_atol: float = 1e-7,
        optimality_atol: float = 1e-7,
        max_simplex_iterations: int = 256,
        validate: bool = False,
    ) -> None:
        self.feasibility_atol = feasibility_atol
        self.optimality_atol = optimality_atol
        self.max_simplex_iterations = max_simplex_iterations
        self.validate = validate
        self._device_constraints: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        if self.validate:
            self._validate(cost, supply, demand)

        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        num_edges = n * m
        batch_size = int(math.prod(batch_shape)) if batch_shape else 1

        with torch.no_grad():
            flat_cost = cost.reshape(batch_size, num_edges)
            flat_supply = supply.reshape(batch_size, n)
            flat_demand = demand.reshape(batch_size, m)
            plan, dual, value = self._solve_simplex(
                flat_cost, flat_supply, flat_demand, n, m
            )

        plan = plan.view(*batch_shape, n, m)
        row_dual = dual[:, :n].view(*batch_shape, n)
        column_dual = cost.new_zeros(batch_size, m)
        column_dual[:, :-1] = dual[:, n:]
        column_dual = column_dual.view(*batch_shape, m)
        value = value.view(batch_shape)
        return _TransportValue.apply(cost, supply, demand, value, plan, row_dual, column_dual)

    def _solve_simplex(
        self,
        cost: Tensor,
        supply: Tensor,
        demand: Tensor,
        n: int,
        m: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = cost.shape[0]
        num_constraints = n + m - 1
        num_edges = n * m
        device = cost.device
        dtype = cost.dtype

        key = (n, m, device, dtype)
        constraints = self._device_constraints.get(key)
        if constraints is None:
            constraints = torch.as_tensor(
                _transport_constraints(n, m), device=device, dtype=dtype
            )  # (R, E)
            self._device_constraints[key] = constraints
        edge_constraints = constraints.T  # (E, R) – contiguous for gather

        # ---------- Northwest-corner initial basic feasible solution ----------
        plan = cost.new_zeros(batch_size, num_edges)
        basis = torch.empty(batch_size, num_constraints, device=device, dtype=torch.long)

        rem_s = supply.clone()
        rem_d = demand.clone()
        rows = torch.zeros(batch_size, device=device, dtype=torch.long)
        cols = torch.zeros(batch_size, device=device, dtype=torch.long)
        batches = torch.arange(batch_size, device=device)

        # Unrolled / tight loop – num_constraints is tiny (≤ ~30 for practical n,m)
        for pos in range(num_constraints):
            edges = rows * m + cols
            basis[:, pos] = edges
            s_val = rem_s[batches, rows]
            d_val = rem_d[batches, cols]
            flow = torch.minimum(s_val, d_val)
            plan[batches, edges] = flow
            rem_s[batches, rows] = (s_val - flow).clamp_min_(0.0)
            rem_d[batches, cols] = (d_val - flow).clamp_min_(0.0)
            advance_row = s_val <= d_val
            rows += advance_row
            cols += ~advance_row

        # ---------- Revised simplex iterations ----------
        dual = None
        for _ in range(self.max_simplex_iterations):
            # B = edge_constraints[basis]  →  (B, R, R)
            B = edge_constraints[basis]  # advanced indexing is fused

            basis_cost = cost.gather(1, basis)  # (B, R)
            dual = torch.linalg.solve(B, basis_cost)  # (B, R)

            # reduced cost = c - dual @ A
            reduced = cost - dual @ constraints  # (B, E)
            reduced.scatter_(1, basis, torch.inf)  # basic variables never enter

            entering = reduced.argmin(dim=-1)  # (B,)
            rc_enter = reduced[batches, entering]
            active = rc_enter < -self.optimality_atol

            # Early exit without Python branch on the hot path when possible
            if not active.any():
                break

            # Direction: solve B^T d = -A_e
            Ae = edge_constraints[entering]  # (B, R)
            direction = torch.linalg.solve(
                B.transpose(-1, -2), -Ae.unsqueeze(-1)
            ).squeeze(-1)  # (B, R)

            basis_flow = plan.gather(1, basis)  # (B, R)
            # Ratio test
            ratios = torch.where(
                direction < -self.feasibility_atol,
                basis_flow / -direction,
                torch.full_like(basis_flow, torch.inf),
            )
            leaving_pos = ratios.argmin(dim=-1)  # (B,)
            theta = ratios[batches, leaving_pos]
            theta = torch.where(active, theta, torch.zeros_like(theta))

            # Update plan in-place
            plan.scatter_(1, basis, basis_flow + theta.unsqueeze(1) * direction)
            plan[batches, entering] += theta

            # Pivot only active problems
            act_idx = batches[active]
            leave_edge = basis[act_idx, leaving_pos[active]]
            plan[act_idx, leave_edge] = 0.0
            basis[act_idx, leaving_pos[active]] = entering[active]
        else:
            raise RuntimeError(
                f"Transport simplex did not converge in {self.max_simplex_iterations} iterations"
            )

        value = (cost * plan).sum(dim=-1)
        return plan, dual, value

    @staticmethod
    def _validate(cost: Tensor, supply: Tensor, demand: Tensor) -> None:
        if cost.ndim < 2:
            raise ValueError(f"Expected batched cost matrices, found shape {cost.shape}")
        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        if supply.shape != (*batch_shape, n) or demand.shape != (*batch_shape, m):
            raise ValueError("Transport marginals do not match the cost batch")
        if cost.device != supply.device or cost.device != demand.device:
            raise ValueError("Transport inputs must be on the same device")
        if cost.dtype != supply.dtype or cost.dtype != demand.dtype:
            raise ValueError("Transport inputs must have the same dtype")
        if not torch.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("Transport costs must be finite and non-negative")
        for marg in (supply, demand):
            if (
                not torch.isfinite(marg).all()
                or (marg < 0).any()
                or not torch.allclose(marg.sum(-1), torch.ones_like(marg[..., 0]))
            ):
                raise ValueError("Transport marginals must be probability vectors")

class TorchTransportSolver2:
    """Exact batched transport using device-native revised simplex."""

    def __init__(
        self,
        *,
        feasibility_atol: float = 1e-7,
        optimality_atol: float = 1e-7,
        max_simplex_iterations: int = 256,
        validate: bool = False,
    ) -> None:
        self.feasibility_atol = feasibility_atol
        self.optimality_atol = optimality_atol
        self.max_simplex_iterations = max_simplex_iterations
        self.validate = validate
        self._device_constraints: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        if self.validate:
            self._validate(cost, supply, demand)

        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        num_edges = n * m
        batch_size = int(math.prod(batch_shape)) if batch_shape else 1

        with torch.no_grad():
            flat_cost = cost.reshape(batch_size, num_edges)
            flat_supply = supply.reshape(batch_size, n)
            flat_demand = demand.reshape(batch_size, m)
            plan, dual, value = self._solve_simplex(
                flat_cost, flat_supply, flat_demand, n, m
            )

        plan = plan.view(*batch_shape, n, m)
        row_dual = dual[:, :n].view(*batch_shape, n)
        column_dual = cost.new_zeros(batch_size, m)
        column_dual[:, :-1] = dual[:, n:]
        column_dual = column_dual.view(*batch_shape, m)
        value = value.view(batch_shape)
        return _TransportValue.apply(cost, supply, demand, value, plan, row_dual, column_dual)

    def _solve_simplex(
        self,
        cost: Tensor,
        supply: Tensor,
        demand: Tensor,
        n: int,
        m: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = cost.shape[0]
        num_constraints = n + m - 1
        num_edges = n * m
        device = cost.device
        dtype = cost.dtype

        key = (n, m, device, dtype)
        constraints = self._device_constraints.get(key)
        if constraints is None:
            constraints = torch.as_tensor(
                _transport_constraints(n, m), device=device, dtype=dtype
            )  # (R, E)
            self._device_constraints[key] = constraints
        edge_constraints = constraints.T  # (E, R) – contiguous for gather

        # ---------- Northwest-corner initial basic feasible solution ----------
        plan = cost.new_zeros(batch_size, num_edges)
        basis = torch.empty(batch_size, num_constraints, device=device, dtype=torch.long)

        rem_s = supply.clone()
        rem_d = demand.clone()
        rows = torch.zeros(batch_size, device=device, dtype=torch.long)
        cols = torch.zeros(batch_size, device=device, dtype=torch.long)
        batches = torch.arange(batch_size, device=device)

        # Unrolled / tight loop – num_constraints is tiny (≤ ~30 for practical n,m)
        for pos in range(num_constraints):
            edges = rows * m + cols
            basis[:, pos] = edges
            s_val = rem_s[batches, rows]
            d_val = rem_d[batches, cols]
            flow = torch.minimum(s_val, d_val)
            plan[batches, edges] = flow
            rem_s[batches, rows] = (s_val - flow).clamp_min_(0.0)
            rem_d[batches, cols] = (d_val - flow).clamp_min_(0.0)
            advance_row = s_val <= d_val
            rows += advance_row
            cols += ~advance_row

        # ---------- Revised simplex iterations ----------
        dual = None
        for _ in range(self.max_simplex_iterations):
            # B = edge_constraints[basis]  →  (B, R, R)
            B = edge_constraints[basis]  # advanced indexing is fused

            basis_cost = cost.gather(1, basis)  # (B, R)
            LU, pivots, _ = torch.linalg.lu_factor_ex(B, check_errors=False)
            dual = torch.linalg.lu_solve(
                LU, pivots, basis_cost.unsqueeze(-1)
            ).squeeze(-1)

            # reduced cost = c - dual @ A
            reduced = cost - dual @ constraints  # (B, E)
            reduced.scatter_(1, basis, torch.inf)  # basic variables never enter

            entering = reduced.argmin(dim=-1)  # (B,)
            rc_enter = reduced[batches, entering]
            active = rc_enter < -self.optimality_atol

            # Early exit without Python branch on the hot path when possible
            if not active.any():
                break

            # Direction: solve B^T d = -A_e
            Ae = edge_constraints[entering]  # (B, R)
            direction = torch.linalg.lu_solve(
                LU, pivots, -Ae.unsqueeze(-1), adjoint=True
            ).squeeze(-1)  # (B, R)

            basis_flow = plan.gather(1, basis)  # (B, R)
            # Ratio test
            ratios = torch.where(
                direction < -self.feasibility_atol,
                basis_flow / -direction,
                torch.full_like(basis_flow, torch.inf),
            )
            leaving_pos = ratios.argmin(dim=-1)  # (B,)
            theta = ratios[batches, leaving_pos]
            theta = torch.where(active, theta, torch.zeros_like(theta))

            # Update plan in-place
            plan.scatter_(1, basis, basis_flow + theta.unsqueeze(1) * direction)
            plan[batches, entering] += theta

            # Pivot only active problems
            leave_edge = basis[batches, leaving_pos]
            leave_flow = plan[batches, leave_edge]
            plan[batches, leave_edge] = leave_flow.masked_fill(active, 0.0)
            basis[batches, leaving_pos] = torch.where(active, entering, leave_edge)
        else:
            raise RuntimeError(
                f"Transport simplex did not converge in {self.max_simplex_iterations} iterations"
            )

        value = (cost * plan).sum(dim=-1)
        return plan, dual, value

    @staticmethod
    def _validate(cost: Tensor, supply: Tensor, demand: Tensor) -> None:
        if cost.ndim < 2:
            raise ValueError(f"Expected batched cost matrices, found shape {cost.shape}")
        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        if supply.shape != (*batch_shape, n) or demand.shape != (*batch_shape, m):
            raise ValueError("Transport marginals do not match the cost batch")
        if cost.device != supply.device or cost.device != demand.device:
            raise ValueError("Transport inputs must be on the same device")
        if cost.dtype != supply.dtype or cost.dtype != demand.dtype:
            raise ValueError("Transport inputs must have the same dtype")
        if not torch.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("Transport costs must be finite and non-negative")
        for marg in (supply, demand):
            if (
                not torch.isfinite(marg).all()
                or (marg < 0).any()
                or not torch.allclose(marg.sum(-1), torch.ones_like(marg[..., 0]))
            ):
                raise ValueError("Transport marginals must be probability vectors")

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
        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        batch_size = int(np.prod(batch_shape)) if batch_shape else 1
        cost_np = cost.detach().double().cpu().numpy().reshape(batch_size, n, m)
        supply_np = supply.detach().double().cpu().numpy().reshape(batch_size, n)
        demand_np = demand.detach().double().cpu().numpy().reshape(batch_size, m)

        with gp.Model(env=self._env) as model:
            plan_variable = model.addMVar((batch_size, n, m), lb=0.0)
            model.setObjective((cost_np * plan_variable).sum(), gp.GRB.MINIMIZE)
            row_constraints = model.addConstr(plan_variable.sum(axis=2) == supply_np)
            column_constraints = (
                model.addConstr(plan_variable.sum(axis=1)[:, :-1] == demand_np[:, :-1])
                if m > 1
                else None
            )
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Gurobi failed to solve balanced transport: {model.Status}")

            plan_np = np.asarray(plan_variable.X)
            row_dual_np = np.asarray(row_constraints.Pi)
            column_dual_np = np.zeros((batch_size, m))
            if column_constraints is not None:
                column_dual_np[:, :-1] = np.asarray(column_constraints.Pi)
            objectives_np = np.sum(cost_np * plan_np, axis=(-2, -1))

        plan = torch.as_tensor(
            plan_np.reshape(*batch_shape, n, m), device=cost.device, dtype=cost.dtype
        )
        row_dual = torch.as_tensor(
            row_dual_np.reshape(*batch_shape, n), device=cost.device, dtype=cost.dtype
        )
        column_dual = torch.as_tensor(
            column_dual_np.reshape(*batch_shape, m), device=cost.device, dtype=cost.dtype
        )
        value = torch.as_tensor(
            objectives_np.reshape(batch_shape), device=cost.device, dtype=cost.dtype
        )
        return _TransportValue.apply(cost, supply, demand, value, plan, row_dual, column_dual)

    def close(self) -> None:
        self._env.close()

    def _validate(self, cost: Tensor, supply: Tensor, demand: Tensor) -> None:
        if cost.ndim < 2:
            raise ValueError(f"Expected batched cost matrices, found shape {cost.shape}")
        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        if supply.shape != (*batch_shape, n) or demand.shape != (*batch_shape, m):
            raise ValueError(
                f"Expected marginal shapes {(*batch_shape, n)} and {(*batch_shape, m)}, "
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
        sums = value.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=self.atol, rtol=self.rtol):
            raise ValueError(f"{name.capitalize()} must sum to one")
