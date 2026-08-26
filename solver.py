import math
from functools import lru_cache
from itertools import combinations
from typing import Any

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


@lru_cache(maxsize=None)
def _transport_basis_maps(n: int, m: int) -> np.ndarray:
    constraints = _transport_constraints(n, m)
    num_constraints, num_edges = constraints.shape
    maps: list[np.ndarray] = []
    for columns in combinations(range(num_edges), num_constraints):
        basis = constraints[:, columns]
        if abs(np.linalg.det(basis)) < 0.5:
            continue
        basis_map = np.zeros((num_edges, num_constraints), dtype=np.int8)
        basis_map[list(columns)] = np.rint(np.linalg.inv(basis)).astype(np.int8)
        maps.append(basis_map)
    return np.stack(maps)


class TorchTransportSolver:
    """Exact batched transport using device-native LP algorithms."""

    def __init__(
        self,
        *,
        feasibility_atol: float = 1e-7,
        optimality_atol: float = 1e-7,
        max_basis_subsets: int = 100_000,
        max_simplex_iterations: int = 256,
        validate: bool = False,
    ) -> None:
        self.feasibility_atol = feasibility_atol
        self.optimality_atol = optimality_atol
        self.max_basis_subsets = max_basis_subsets
        self.max_simplex_iterations = max_simplex_iterations
        self.validate = validate
        self._device_maps: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}
        self._device_constraints: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        if self.validate:
            self._validate(cost, supply, demand)
        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        num_constraints = n + m - 1
        num_edges = n * m
        num_subsets = math.comb(num_edges, num_constraints)
        batch_size = math.prod(batch_shape) if batch_shape else 1

        with torch.no_grad():
            flat_cost = cost.detach().reshape(batch_size, num_edges)
            flat_supply = supply.detach().reshape(batch_size, n)
            flat_demand = demand.detach().reshape(batch_size, m)
            if num_subsets <= self.max_basis_subsets:
                plan, dual, value = self._solve_enumerative(
                    flat_cost, flat_supply, flat_demand, n, m
                )
            else:
                plan, dual, value = self._solve_simplex(flat_cost, flat_supply, flat_demand, n, m)

        plan = plan.reshape(*batch_shape, n, m)
        row_dual = dual[:, :n].reshape(*batch_shape, n)
        column_dual = cost.new_zeros((batch_size, m))
        column_dual[:, :-1] = dual[:, n:]
        column_dual = column_dual.reshape(*batch_shape, m)
        value = value.reshape(batch_shape)
        return _TransportValue.apply(cost, supply, demand, value, plan, row_dual, column_dual)

    def _solve_enumerative(
        self,
        cost: Tensor,
        supply: Tensor,
        demand: Tensor,
        n: int,
        m: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = (n, m, cost.device, cost.dtype)
        if key not in self._device_maps:
            self._device_maps[key] = torch.as_tensor(
                _transport_basis_maps(n, m), device=cost.device, dtype=cost.dtype
            )
        basis_maps = self._device_maps[key]
        rhs = torch.cat((supply, demand[:, :-1]), dim=-1)
        plans = torch.einsum("ver,br->bve", basis_maps, rhs)
        feasible = torch.all(plans >= -self.feasibility_atol, dim=-1)
        objectives = torch.einsum("be,bve->bv", cost, plans)
        objectives.masked_fill_(~feasible, torch.inf)
        indices = torch.argmin(objectives, dim=-1)
        selected_maps = basis_maps[indices]
        plan = torch.einsum("ber,br->be", selected_maps, rhs)
        dual = torch.einsum("ber,be->br", selected_maps, cost)
        value = torch.sum(cost * plan, dim=-1)
        return plan, dual, value

    def _solve_simplex(
        self,
        cost: Tensor,
        supply: Tensor,
        demand: Tensor,
        n: int,
        m: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = cost.shape[0]
        num_constraints = n + m - 1
        num_edges = n * m
        key = (n, m, cost.device, cost.dtype)
        if key not in self._device_constraints:
            self._device_constraints[key] = torch.as_tensor(
                _transport_constraints(n, m), device=cost.device, dtype=cost.dtype
            )
        constraints = self._device_constraints[key]
        edge_constraints = constraints.T

        plan = cost.new_zeros((batch_size, num_edges))
        basis = torch.empty((batch_size, num_constraints), device=cost.device, dtype=torch.long)
        remaining_supply = supply.clone()
        remaining_demand = demand.clone()
        rows = torch.zeros(batch_size, device=cost.device, dtype=torch.long)
        columns = torch.zeros(batch_size, device=cost.device, dtype=torch.long)
        batches = torch.arange(batch_size, device=cost.device)
        for position in range(num_constraints):
            edges = rows * m + columns
            basis[:, position] = edges
            supply_values = remaining_supply[batches, rows]
            demand_values = remaining_demand[batches, columns]
            flow = torch.minimum(supply_values, demand_values)
            plan[batches, edges] = flow
            remaining_supply[batches, rows] = torch.clamp_min(supply_values - flow, 0.0)
            remaining_demand[batches, columns] = torch.clamp_min(demand_values - flow, 0.0)
            advance_row = supply_values <= demand_values
            rows = rows + advance_row
            columns = columns + ~advance_row

        dual: Tensor | None = None
        for _ in range(self.max_simplex_iterations):
            basis_transpose = edge_constraints[basis]
            basis_cost = torch.gather(cost, 1, basis)
            dual = torch.linalg.solve(basis_transpose, basis_cost)
            reduced_cost = cost - dual @ constraints
            reduced_cost.scatter_(1, basis, torch.inf)
            entering = torch.argmin(reduced_cost, dim=-1)
            active = reduced_cost[batches, entering] < -self.optimality_atol
            if not torch.any(active).item():
                break

            entering_constraints = edge_constraints[entering]
            direction = torch.linalg.solve(
                basis_transpose.transpose(-2, -1),
                -entering_constraints.unsqueeze(-1),
            ).squeeze(-1)
            basis_flow = torch.gather(plan, 1, basis)
            ratios = torch.where(
                direction < -self.feasibility_atol,
                basis_flow / -direction,
                torch.inf,
            )
            leaving_position = torch.argmin(ratios, dim=-1)
            theta = ratios[batches, leaving_position]
            theta = torch.where(active, theta, torch.zeros_like(theta))

            plan.scatter_(1, basis, basis_flow + theta[:, None] * direction)
            plan[batches, entering] += theta
            active_batches = batches[active]
            leaving_edges = basis[active_batches, leaving_position[active]]
            plan[active_batches, leaving_edges] = 0.0
            basis[active_batches, leaving_position[active]] = entering[active]
        else:
            raise RuntimeError(
                f"Transport simplex did not converge in {self.max_simplex_iterations} iterations"
            )

        assert dual is not None
        value = torch.sum(cost * plan, dim=-1)
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
        if not torch.all(torch.isfinite(cost)).item() or torch.any(cost < 0.0).item():
            raise ValueError("Transport costs must be finite and non-negative")
        for marginal in (supply, demand):
            if (
                not torch.all(torch.isfinite(marginal)).item()
                or torch.any(marginal < 0.0).item()
                or not torch.allclose(
                    marginal.sum(dim=-1),
                    torch.ones_like(marginal[..., 0]),
                )
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
