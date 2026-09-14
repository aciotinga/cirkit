# Exact and differentiable Circuit-Wasserstein query

import math
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import (
    TorchCategoricalLayer,
    TorchGaussianLayer,
    TorchHadamardLayer,
    TorchInputLayer,
    TorchKroneckerLayer,
    TorchLayer,
    TorchSumLayer,
)
from cirkit.backend.torch.parameters.nodes import TorchSoftmaxParameter, TorchTensorParameter
from cirkit.utils.scope import Scope


class TransportSolver(Protocol):
    # Generic balanced optimal transport solver

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        """Solve a batch of transport problems from live query tensors.

        For batch shape ``B``, inputs have shapes ``B + (n, m)``,
        ``B + (n,)``, and ``B + (m,)``. The result has shape ``B``.
        The solver owns validation, device handling, and LP subgradients.
        """


class HighsTransportSolver:
    # Solve balanced OT with HiGHS and return LP subgradient

    def __init__(self, *, atol: float = 1e-7, rtol: float = 1e-6) -> None:
        assert atol >= 0.0 and rtol >= 0.0, "Transport tolerances must be non-negative"
        self.atol = atol
        self.rtol = rtol

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        assert cost.ndim >= 2, f"Expected batched cost matrices, found shape {cost.shape}"
        batch_shape = cost.shape[:-2]
        n, m = cost.shape[-2:]
        assert supply.shape == (
            *batch_shape,
            n,
        ), f"Expected supply shape {(*batch_shape, n)}, found {supply.shape}"
        assert demand.shape == (
            *batch_shape,
            m,
        ), f"Expected demand shape {(*batch_shape, m)}, found {demand.shape}"
        assert (
            cost.device == supply.device == demand.device
        ), "Transport costs and marginals must be on the same device"
        assert (
            cost.dtype == supply.dtype == demand.dtype
        ), "Transport costs and marginals must have the same dtype"
        assert cost.is_floating_point(), "Transport inputs must be floating-point tensors"
        _validate_finite_nonnegative(cost, "transport costs")
        _validate_probability_vectors(supply, "transport supply", self.atol, self.rtol)
        _validate_probability_vectors(demand, "transport demand", self.atol, self.rtol)

        cost_np = cost.detach().double().cpu().numpy()
        supply_np = supply.detach().double().cpu().numpy()
        demand_np = demand.detach().double().cpu().numpy()
        if not np.all(
            np.isclose(
                supply_np.sum(axis=-1),
                demand_np.sum(axis=-1),
                atol=self.atol,
                rtol=self.rtol,
            )
        ):
            raise ValueError("Balanced transport requires equal total mass")

        batch_size = int(np.prod(batch_shape)) if batch_shape else 1
        costs_np = cost_np.reshape(batch_size, n, m)
        supplies_np = supply_np.reshape(batch_size, n)
        demands_np = demand_np.reshape(batch_size, m)
        plans_np = np.empty((batch_size, n, m))
        row_duals_np = np.empty((batch_size, n))
        column_duals_np = np.zeros((batch_size, m))
        values_np = np.empty(batch_size)
        equality_matrix = _transport_equality_matrix(n, m)
        for index, (cost_i, supply_i, demand_i) in enumerate(
            zip(costs_np, supplies_np, demands_np)
        ):
            result = linprog(
                cost_i.reshape(-1),
                A_eq=equality_matrix,
                b_eq=np.concatenate((supply_i, demand_i[:-1])),
                bounds=(0.0, None),
                method="highs-ds",
            )
            if not result.success:
                raise RuntimeError(f"HiGHS failed to solve balanced transport: {result.message}")
            duals = np.asarray(result.eqlin.marginals)
            plans_np[index] = result.x.reshape(n, m)
            row_duals_np[index] = duals[:n]
            column_duals_np[index, :-1] = duals[n:]
            values_np[index] = result.fun

        plan = torch.as_tensor(
            plans_np.reshape(*batch_shape, n, m), device=cost.device, dtype=cost.dtype
        )
        row_dual = torch.as_tensor(
            row_duals_np.reshape(*batch_shape, n), device=cost.device, dtype=cost.dtype
        )
        column_dual = torch.as_tensor(
            column_duals_np.reshape(*batch_shape, m), device=cost.device, dtype=cost.dtype
        )
        value = torch.as_tensor(
            values_np.reshape(batch_shape), device=cost.device, dtype=cost.dtype
        )
        return (
            value
            + torch.sum((cost - cost.detach()) * plan, dim=(-2, -1))
            + torch.sum((supply - supply.detach()) * row_dual, dim=-1)
            + torch.sum((demand - demand.detach()) * column_dual, dim=-1)
        )


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
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = cost.shape[0]
        num_constraints = n + m - 1
        num_edges = n * m
        device = cost.device
        dtype = cost.dtype
        numerical_optimality_atol = (
            num_constraints * torch.finfo(dtype).eps * cost.abs().amax(dim=-1)
        )

        key = (n, m, device, dtype)
        constraints = self._device_constraints.get(key)
        if constraints is None:
            constraints = torch.as_tensor(
                _transport_constraints(n, m), device=device, dtype=dtype
            )
            self._device_constraints[key] = constraints
        edge_constraints = constraints.T

        plan = cost.new_zeros(batch_size, num_edges)
        basis = torch.empty(batch_size, num_constraints, device=device, dtype=torch.long)

        rem_s = supply.clone()
        rem_d = demand.clone()
        rows = torch.zeros(batch_size, device=device, dtype=torch.long)
        cols = torch.zeros(batch_size, device=device, dtype=torch.long)
        batches = torch.arange(batch_size, device=device)

        for pos in range(num_constraints):
            edges = rows * m + cols
            basis[:, pos] = edges
            s_val = rem_s[batches, rows]
            d_val = rem_d[batches, cols]
            flow = torch.minimum(s_val, d_val)
            plan[batches, edges] = flow
            rem_s[batches, rows] = (s_val - flow).clamp_min_(0.0)
            rem_d[batches, cols] = (d_val - flow).clamp_min_(0.0)
            row_empty = rem_s[batches, rows] == 0
            column_empty = rem_d[batches, cols] == 0
            # On a tie, advance one axis so the next zero-flow edge keeps the basis full rank.
            advance_row = row_empty & (~column_empty | (rows < n - 1))
            rows += advance_row
            cols += column_empty & ~advance_row
            rows.clamp_(max=n - 1)
            cols.clamp_(max=m - 1)

        dual = None
        for iteration in range(self.max_simplex_iterations):
            B = edge_constraints[basis]
            basis_cost = cost.gather(1, basis)
            LU, pivots, _ = torch.linalg.lu_factor_ex(B, check_errors=False)
            dual = torch.linalg.lu_solve(LU, pivots, basis_cost.unsqueeze(-1)).squeeze(-1)

            reduced = cost - dual @ constraints
            reduced.scatter_(1, basis, torch.inf)

            # Use the fast Dantzig rule normally, then Bland's rule to break rare cycles.
            if iteration < num_edges:
                entering = reduced.argmin(dim=-1)
                active = reduced[batches, entering] < -(
                    self.optimality_atol + numerical_optimality_atol
                )
            else:
                improving = reduced < -(
                    self.optimality_atol + numerical_optimality_atol[:, None]
                )
                entering = improving.to(torch.int8).argmax(dim=-1)
                active = improving[batches, entering]
            if not active.any():
                break

            Ae = edge_constraints[entering]
            direction = torch.linalg.lu_solve(
                LU, pivots, -Ae.unsqueeze(-1), adjoint=True
            ).squeeze(-1)

            basis_flow = plan.gather(1, basis)
            ratios = torch.where(
                direction < -self.feasibility_atol,
                basis_flow / -direction,
                torch.full_like(basis_flow, torch.inf),
            )
            if iteration < num_edges:
                leaving_pos = ratios.argmin(dim=-1)
            else:
                minimum_ratio = ratios.amin(dim=-1, keepdim=True)
                leaving_edges = basis.masked_fill(ratios != minimum_ratio, num_edges)
                leaving_pos = leaving_edges.argmin(dim=-1)
            theta = ratios[batches, leaving_pos]
            theta = torch.where(active, theta, torch.zeros_like(theta))

            plan.scatter_(1, basis, basis_flow + theta.unsqueeze(1) * direction)
            plan[batches, entering] += theta

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


@lru_cache(maxsize=None)
def _transport_equality_matrix(n: int, m: int) -> sparse.csr_matrix:
    # Build row constraints and all but one redundant column constraint, cached because it's expensive

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for i in range(n):
        for j in range(m):
            rows.append(i)
            columns.append(i * m + j)
            values.append(1.0)
    for j in range(m - 1):
        for i in range(n):
            rows.append(n + j)
            columns.append(i * m + j)
            values.append(1.0)
    return sparse.csr_matrix((values, (rows, columns)), shape=(n + m - 1, n * m))


TorchProductLayer = TorchHadamardLayer | TorchKroneckerLayer
LayerPair = tuple[TorchLayer, TorchLayer]


def _same_fold_index(indices1: list[Any], indices2: list[Any]) -> bool:
    if len(indices1) != len(indices2):
        return False
    for index1, index2 in zip(indices1, indices2):
        if isinstance(index1, Tensor) and isinstance(index2, Tensor):
            if not torch.equal(index1.cpu(), index2.cpu()):
                return False
        elif type(index1) is not type(index2) or index1 != index2:
            return False
    return True


def _product_unit_indices(layer: TorchProductLayer, child_index: int) -> list[int]:
    if isinstance(layer, TorchHadamardLayer):
        return list(range(layer.num_output_units))
    stride = layer.num_input_units ** (layer.arity - child_index - 1)
    return [
        (unit // stride) % layer.num_input_units for unit in range(layer.num_output_units)
    ]


class _FoldedCircuitWassersteinEngine:
    """Vectorized CW evaluation for identically folded circuit structures."""

    def __init__(
        self,
        circuit1: TorchCircuit,
        circuit2: TorchCircuit,
        *,
        metric_p: float,
        scale_factor: float,
        transport_solver: TransportSolver,
        probability_atol: float,
        probability_rtol: float,
    ) -> None:
        assert metric_p > 0.0, "metric_p must be positive"
        assert scale_factor > 0.0, "scale_factor must be positive"
        assert probability_atol >= 0.0 and probability_rtol >= 0.0
        self.circuits = (circuit1, circuit2)
        self.metric_p = metric_p
        self.scale_factor = scale_factor
        self.transport_solver = transport_solver
        self.probability_atol = probability_atol
        self.probability_rtol = probability_rtol
        self._reference_tensor: Tensor | None = None
        self._output_layer, self._output_fold = self._validate_circuits()

    def __call__(self) -> Tensor:
        self._reference_tensor = None
        values: list[Tensor] = []
        entries = list(self.circuits[0].address_book)
        for layer1, layer2, entry in zip(
            self.circuits[0].layers, self.circuits[1].layers, entries
        ):
            if isinstance(layer1, TorchInputLayer):
                assert isinstance(layer2, TorchInputLayer)
                value = self._input_matrix(layer1, layer2)
            else:
                children = self._gather_inputs(values, entry)
                if isinstance(layer1, TorchSumLayer):
                    assert isinstance(layer2, TorchSumLayer)
                    value = self._sum_matrix(layer1, layer2, children)
                else:
                    assert isinstance(layer1, (TorchHadamardLayer, TorchKroneckerLayer))
                    assert isinstance(layer2, (TorchHadamardLayer, TorchKroneckerLayer))
                    value = self._product_matrix(layer1, layer2, children)
            values.append(value)
        return values[self._output_layer][self._output_fold, 0, 0]

    @staticmethod
    def _gather_inputs(values: list[Tensor], entry: Any) -> Tensor:
        (input_ids,) = entry.in_module_ids
        inputs = (
            values[input_ids[0]]
            if len(input_ids) == 1
            else torch.cat([values[index] for index in input_ids])
        )
        (fold_index,) = entry.in_fold_idx
        return inputs[fold_index]

    def _input_matrix(self, layer1: TorchInputLayer, layer2: TorchInputLayer) -> Tensor:
        if isinstance(layer1, TorchCategoricalLayer) and isinstance(
            layer2, TorchCategoricalLayer
        ):
            probs1 = self._categorical_probabilities(layer1)
            probs2 = self._categorical_probabilities(layer2)
            _validate_probability_rows(
                probs1, "categorical probabilities", self.probability_atol, self.probability_rtol
            )
            _validate_probability_rows(
                probs2, "categorical probabilities", self.probability_atol, self.probability_rtol
            )
            num_categories = max(probs1.shape[-1], probs2.shape[-1])
            probs1 = torch.nn.functional.pad(probs1, (0, num_categories - probs1.shape[-1]))
            probs2 = torch.nn.functional.pad(probs2, (0, num_categories - probs2.shape[-1]))
            if self.metric_p == 1.0:
                cdf1 = torch.cumsum(probs1, dim=-1)
                cdf2 = torch.cumsum(probs2, dim=-1)
                return (
                    torch.abs(cdf1[:, :, None, :-1] - cdf2[:, None, :, :-1]).sum(dim=-1)
                    / self.scale_factor
                )
            support = torch.arange(num_categories, device=probs1.device, dtype=probs1.dtype)
            cost = torch.abs(support[:, None] - support[None, :]).pow(self.metric_p)
            num_folds, num_units1 = probs1.shape[:2]
            num_units2 = probs2.shape[1]
            return self.transport_solver(
                cost.expand(num_folds, num_units1, num_units2, -1, -1) / self.scale_factor,
                probs1[:, :, None].expand(-1, -1, num_units2, -1),
                probs2[:, None].expand(-1, num_units1, -1, -1),
            )

        assert isinstance(layer1, TorchGaussianLayer)
        assert isinstance(layer2, TorchGaussianLayer)
        if layer1.log_partition is not None or layer2.log_partition is not None:
            raise ValueError("Circuit-Wasserstein requires normalized Gaussian leaves")
        mean1 = self._parameter(layer1, "mean")
        mean2 = self._parameter(layer2, "mean")
        stddev1 = self._parameter(layer1, "stddev")
        stddev2 = self._parameter(layer2, "stddev")
        for value, name in (
            (mean1, "Gaussian mean"),
            (mean2, "Gaussian mean"),
            (stddev1, "Gaussian stddev"),
            (stddev2, "Gaussian stddev"),
        ):
            if not torch.all(torch.isfinite(value)).item():
                raise ValueError(f"{name} must be finite")
        if torch.any(stddev1 <= 0.0).item() or torch.any(stddev2 <= 0.0).item():
            raise ValueError("Gaussian standard deviations must be positive")
        return (
            torch.square(mean1[:, :, None] - mean2[:, None, :])
            + torch.square(stddev1[:, :, None] - stddev2[:, None, :])
        ) / self.scale_factor

    def _sum_matrix(
        self, layer1: TorchSumLayer, layer2: TorchSumLayer, children: Tensor
    ) -> Tensor:
        if children.shape[1] != 1:
            raise RuntimeError("Aligned folded Circuit-Wasserstein requires unary sum layers")
        cost = children[:, 0]
        weights1 = self._parameter(layer1, "weight")
        weights2 = self._parameter(layer2, "weight")
        num_units1, num_units2 = weights1.shape[1], weights2.shape[1]
        return self.transport_solver(
            cost[:, None, None].expand(-1, num_units1, num_units2, -1, -1),
            weights1[:, :, None].expand(-1, -1, num_units2, -1),
            weights2[:, None].expand(-1, num_units1, -1, -1),
        )

    @staticmethod
    def _product_matrix(
        layer1: TorchProductLayer, layer2: TorchProductLayer, children: Tensor
    ) -> Tensor:
        if isinstance(layer1, TorchHadamardLayer) and isinstance(layer2, TorchHadamardLayer):
            return children.sum(dim=1)

        result: Tensor | None = None
        for child_index in range(layer1.arity):
            units1 = _product_unit_indices(layer1, child_index)
            units2 = _product_unit_indices(layer2, child_index)
            child = children[:, child_index][:, units1][:, :, units2]
            result = child if result is None else result + child
        assert result is not None
        return result

    def _categorical_probabilities(self, layer: TorchCategoricalLayer) -> Tensor:
        if layer.logits is not None:
            return torch.softmax(self._parameter(layer, "logits"), dim=-1)
        return self._parameter(layer, "probs")

    def _parameter(self, layer: TorchLayer, name: str) -> Tensor:
        value = layer.params[name]()
        if value.shape[0] != layer.num_folds:
            raise ValueError("Circuit parameter folds do not match their layer")
        if not value.is_floating_point():
            raise ValueError("Circuit parameters must be floating-point tensors")
        if self._reference_tensor is None:
            self._reference_tensor = value
        elif (
            value.device != self._reference_tensor.device
            or value.dtype != self._reference_tensor.dtype
        ):
            raise ValueError("Both circuits must use the same parameter device and dtype")
        return value

    def _validate_circuits(self) -> tuple[int, int]:
        circuit1, circuit2 = self.circuits
        for index, circuit in enumerate(self.circuits, 1):
            if not circuit.is_folded:
                raise ValueError(f"Circuit {index} is not folded")
            if (
                not circuit.properties.smooth
                or not circuit.properties.decomposable
                or not circuit.properties.structured_decomposable
            ):
                raise ValueError(
                    f"Circuit {index} must be smooth and structured-decomposable, "
                    f"found {circuit.properties}"
                )
            if len(circuit.outputs) != 1 or circuit.outputs[0].num_output_units != 1:
                raise ValueError(f"Circuit {index} must have exactly one scalar output")
        if circuit1.scope != circuit2.scope:
            raise ValueError("Circuits must have identical scopes")
        if len(circuit1.layers) != len(circuit2.layers):
            raise ValueError("Folded circuits must have aligned layer structures")

        entries1 = list(circuit1.address_book)
        entries2 = list(circuit2.address_book)
        for layer1, layer2, entry1, entry2 in zip(
            circuit1.layers, circuit2.layers, entries1, entries2
        ):
            if layer1.num_folds != layer2.num_folds:
                raise ValueError("Folded circuits must have aligned folds")
            if entry1.in_module_ids != entry2.in_module_ids or not _same_fold_index(
                entry1.in_fold_idx, entry2.in_fold_idx
            ):
                raise ValueError("Folded circuits must have aligned connectivity")
            if isinstance(layer1, TorchInputLayer) and isinstance(layer2, TorchInputLayer):
                if type(layer1) is not type(layer2):
                    raise ValueError("Paired folded input layers have different operations")
                if not torch.equal(layer1.scope_idx.cpu(), layer2.scope_idx.cpu()):
                    raise ValueError("Paired folded input layers have different scopes")
                if isinstance(layer1, TorchGaussianLayer) and self.metric_p != 2.0:
                    raise ValueError("Gaussian Circuit-Wasserstein requires metric_p=2")
            elif isinstance(layer1, TorchSumLayer) and isinstance(layer2, TorchSumLayer):
                if layer1.arity != 1 or layer2.arity != 1:
                    raise NotImplementedError(
                        "Folded Circuit-Wasserstein currently requires unary sum layers; "
                        "compile with fold=False for general sum structures"
                    )
            elif isinstance(layer1, (TorchHadamardLayer, TorchKroneckerLayer)) and isinstance(
                layer2, (TorchHadamardLayer, TorchKroneckerLayer)
            ):
                if layer1.arity != layer2.arity:
                    raise ValueError("Paired folded products have different arities")
            else:
                raise ValueError("Paired folded layers have different operations")

        output1, output2 = entries1[-1], entries2[-1]
        if output1.in_module_ids != output2.in_module_ids or not _same_fold_index(
            output1.in_fold_idx, output2.in_fold_idx
        ):
            raise ValueError("Folded circuits must have aligned outputs")
        (output_ids,) = output1.in_module_ids
        (output_index,) = output1.in_fold_idx
        if len(output_ids) != 1 or not isinstance(output_index, Tensor) or output_index.numel() != 1:
            raise ValueError("Folded Circuit-Wasserstein requires one aligned output fold")
        return output_ids[0], int(output_index.item())


class _CircuitWassersteinEngine:  # pylint: disable=too-many-instance-attributes
    # Bottom-up CW evaluation over pairs of unfolded Torch circuit layers

    def __init__(
        self,
        circuit1: TorchCircuit,
        circuit2: TorchCircuit,
        *,
        metric_p: float,
        scale_factor: float,
        transport_solver: TransportSolver,
        probability_atol: float,
        probability_rtol: float,
    ) -> None:
        assert metric_p > 0.0, "metric_p must be positive"
        assert scale_factor > 0.0, "scale_factor must be positive"
        assert (
            probability_atol >= 0.0 and probability_rtol >= 0.0
        ), "Probability tolerances must be non-negative"
        self.circuits = (circuit1, circuit2)
        self.metric_p = metric_p
        self.scale_factor = scale_factor
        self.transport_solver = transport_solver
        self.probability_atol = probability_atol
        self.probability_rtol = probability_rtol
        self._scopes = (self._validate_circuit(circuit1, 1), self._validate_circuit(circuit2, 2))

        # Compatible PCs necessary
        assert circuit1.scope == circuit2.scope, "Circuits must have identical scopes"
        self._validate_pair(circuit1.outputs[0], 0, circuit2.outputs[0], 0, set())
        self._levels, self._dependencies = self._build_layer_pair_schedule(
            (circuit1.outputs[0], circuit2.outputs[0])
        )
        self._product_specs = self._build_product_specs()
        self._batched_softmax_groups = self._build_batched_softmax_groups()
        self._parameter_cache: tuple[
            dict[tuple[TorchLayer, str], Tensor],
            dict[tuple[TorchLayer, str], Tensor],
        ]
        self._reference_tensor: Tensor | None

    def __call__(self) -> Tensor:
        self._parameter_cache = ({}, {})
        self._reference_tensor = None
        self._evaluate_batched_softmaxes()
        values: dict[LayerPair, Tensor] = {}
        for level in self._levels:
            input_pairs: list[tuple[TorchInputLayer, TorchInputLayer]] = []
            sum_pairs: list[tuple[TorchSumLayer, TorchSumLayer]] = []
            for layer1, layer2 in level:
                if isinstance(layer1, TorchInputLayer) and isinstance(layer2, TorchInputLayer):
                    input_pairs.append((layer1, layer2))
                elif isinstance(layer1, TorchSumLayer) and isinstance(layer2, TorchSumLayer):
                    sum_pairs.append((layer1, layer2))
                elif isinstance(layer1, (TorchHadamardLayer, TorchKroneckerLayer)) and isinstance(
                    layer2, (TorchHadamardLayer, TorchKroneckerLayer)
                ):
                    values[(layer1, layer2)] = self._product_matrix(layer1, layer2, values)
                else:
                    raise RuntimeError("Internal Circuit-Wasserstein operation mismatch")
            self._evaluate_input_pairs(input_pairs, values)
            self._evaluate_sum_pairs(sum_pairs, values)
        return values[(self.circuits[0].outputs[0], self.circuits[1].outputs[0])][0, 0]

    @staticmethod
    def _validate_circuit(circuit: TorchCircuit, index: int) -> dict[TorchLayer, Scope]:
        if not circuit.properties.smooth or not circuit.properties.decomposable:
            raise ValueError(
                f"Circuit {index} must be smooth and decomposable, found {circuit.properties}"
            )
        if not circuit.properties.structured_decomposable:
            raise ValueError(
                f"Circuit {index} must be structured-decomposable, found {circuit.properties}"
            )
        if len(circuit.outputs) != 1 or circuit.outputs[0].num_output_units != 1:
            raise ValueError(f"Circuit {index} must have exactly one scalar output")

        supported = (
            TorchCategoricalLayer,
            TorchGaussianLayer,
            TorchSumLayer,
            TorchHadamardLayer,
            TorchKroneckerLayer,
        )
        scopes: dict[TorchLayer, Scope] = {}
        for layer in circuit.topological_ordering():
            if layer.num_folds != 1:
                raise ValueError("Circuit-Wasserstein requires unfolded Torch circuits")
            if not isinstance(layer, supported):
                raise NotImplementedError(
                    f"Unsupported Torch layer {type(layer).__name__}; "
                    "only categorical, Gaussian, sum, Hadamard, and Kronecker layers are supported"
                )
            inputs = circuit.layer_inputs(layer)
            if isinstance(layer, TorchInputLayer):
                scopes[layer] = Scope(layer.scope_idx[0].detach().cpu().tolist())
            else:
                scopes[layer] = Scope.union(*(scopes[child] for child in inputs))
        return scopes

    def _validate_pair(
        self,
        layer1: TorchLayer,
        unit1: int,
        layer2: TorchLayer,
        unit2: int,
        seen: set[tuple[TorchLayer, int, TorchLayer, int]],
    ) -> None:
        # Check if the circuits are compatible
        key = (layer1, unit1, layer2, unit2)
        if key in seen:
            return
        seen.add(key)
        scope1 = self._scopes[0][layer1]
        scope2 = self._scopes[1][layer2]
        if scope1 != scope2:
            raise ValueError(f"Paired units have different scopes: {scope1} and {scope2}")

        if isinstance(layer1, TorchInputLayer) and isinstance(layer2, TorchInputLayer):
            if type(layer1) is not type(layer2):
                raise ValueError(
                    f"Paired input units have different operations: "
                    f"{type(layer1).__name__} and {type(layer2).__name__}"
                )
            if isinstance(layer1, TorchGaussianLayer) and self.metric_p != 2.0:
                raise ValueError("Gaussian Circuit-Wasserstein requires metric_p=2")
            return
        if isinstance(layer1, TorchSumLayer) and isinstance(layer2, TorchSumLayer):
            for child1, child_unit1 in self._sum_children(0, layer1):
                for child2, child_unit2 in self._sum_children(1, layer2):
                    self._validate_pair(child1, child_unit1, child2, child_unit2, seen)
            return
        if isinstance(layer1, (TorchHadamardLayer, TorchKroneckerLayer)) and isinstance(
            layer2, (TorchHadamardLayer, TorchKroneckerLayer)
        ):
            children1 = self._product_children(0, layer1, unit1)
            children2 = self._product_children(1, layer2, unit2)
            for child1, child_unit1, child2, child_unit2 in self._match_product_children(
                children1, children2
            ):
                self._validate_pair(child1, child_unit1, child2, child_unit2, seen)
            return
        raise ValueError(
            f"Paired units have different operations: "
            f"{type(layer1).__name__} and {type(layer2).__name__}"
        )

    def _build_layer_pair_schedule(
        self, output_pair: LayerPair
    ) -> tuple[list[list[LayerPair]], dict[LayerPair, list[LayerPair]]]:
        dependencies: dict[LayerPair, list[LayerPair]] = {}
        depths: dict[LayerPair, int] = {}

        def visit(pair: LayerPair) -> int:
            if pair in depths:
                return depths[pair]
            pair_dependencies = self._layer_pair_dependencies(*pair)
            dependencies[pair] = pair_dependencies
            depth = (
                1 + max(visit(dependency) for dependency in pair_dependencies)
                if pair_dependencies
                else 0
            )
            depths[pair] = depth
            return depth

        max_depth = visit(output_pair)
        levels: list[list[LayerPair]] = [[] for _ in range(max_depth + 1)]
        for pair, depth in depths.items():
            levels[depth].append(pair)
        return levels, dependencies

    def _layer_pair_dependencies(self, layer1: TorchLayer, layer2: TorchLayer) -> list[LayerPair]:
        if isinstance(layer1, TorchInputLayer) and isinstance(layer2, TorchInputLayer):
            return []
        if isinstance(layer1, TorchSumLayer) and isinstance(layer2, TorchSumLayer):
            return [
                (child1, child2)
                for child1 in self.circuits[0].layer_inputs(layer1)
                for child2 in self.circuits[1].layer_inputs(layer2)
            ]
        if isinstance(layer1, (TorchHadamardLayer, TorchKroneckerLayer)) and isinstance(
            layer2, (TorchHadamardLayer, TorchKroneckerLayer)
        ):
            matched = self._match_product_children(
                self._product_children(0, layer1, 0),
                self._product_children(1, layer2, 0),
            )
            return [(child1, child2) for child1, _, child2, _ in matched]
        raise RuntimeError("Internal Circuit-Wasserstein operation mismatch")

    def _build_product_specs(
        self,
    ) -> dict[
        LayerPair,
        list[tuple[LayerPair, tuple[int, ...], tuple[int, ...]]],
    ]:
        specs: dict[
            LayerPair,
            list[tuple[LayerPair, tuple[int, ...], tuple[int, ...]]],
        ] = {}
        for pair in self._dependencies:
            layer1, layer2 = pair
            if not isinstance(layer1, (TorchHadamardLayer, TorchKroneckerLayer)):
                continue
            assert isinstance(layer2, (TorchHadamardLayer, TorchKroneckerLayer))
            pair_specs = []
            for child1, child2 in self._dependencies[pair]:
                units1 = tuple(
                    dict(self._product_children(0, layer1, unit))[child1]
                    for unit in range(layer1.num_output_units)
                )
                units2 = tuple(
                    dict(self._product_children(1, layer2, unit))[child2]
                    for unit in range(layer2.num_output_units)
                )
                pair_specs.append(((child1, child2), units1, units2))
            specs[pair] = pair_specs
        return specs

    def _input_matrix(self, layer1: TorchInputLayer, layer2: TorchInputLayer) -> Tensor:
        if isinstance(layer1, TorchCategoricalLayer) and isinstance(layer2, TorchCategoricalLayer):
            return self._categorical_matrix(layer1, layer2)
        if isinstance(layer1, TorchGaussianLayer) and isinstance(layer2, TorchGaussianLayer):
            return self._gaussian_matrix(layer1, layer2)
        raise RuntimeError("Internal Circuit-Wasserstein input mismatch")

    def _evaluate_input_pairs(
        self,
        pairs: Sequence[tuple[TorchInputLayer, TorchInputLayer]],
        values: dict[LayerPair, Tensor],
    ) -> None:
        categorical_groups: dict[
            tuple[int, int, int, int],
            list[tuple[TorchCategoricalLayer, TorchCategoricalLayer]],
        ] = defaultdict(list)
        for layer1, layer2 in pairs:
            if (
                self.metric_p == 1.0
                and isinstance(layer1, TorchCategoricalLayer)
                and isinstance(layer2, TorchCategoricalLayer)
            ):
                categorical_groups[
                    (
                        layer1.num_output_units,
                        layer2.num_output_units,
                        layer1.num_categories,
                        layer2.num_categories,
                    )
                ].append((layer1, layer2))
            else:
                values[(layer1, layer2)] = self._input_matrix(layer1, layer2)

        for group in categorical_groups.values():
            probs1 = torch.stack(
                [self._categorical_probabilities(0, layer1) for layer1, _ in group]
            )
            probs2 = torch.stack(
                [self._categorical_probabilities(1, layer2) for _, layer2 in group]
            )
            _validate_probability_rows(
                probs1,
                "categorical probabilities",
                self.probability_atol,
                self.probability_rtol,
            )
            _validate_probability_rows(
                probs2,
                "categorical probabilities",
                self.probability_atol,
                self.probability_rtol,
            )
            num_categories = max(probs1.shape[-1], probs2.shape[-1])
            probs1 = torch.nn.functional.pad(probs1, (0, num_categories - probs1.shape[-1]))
            probs2 = torch.nn.functional.pad(probs2, (0, num_categories - probs2.shape[-1]))
            cdf1 = torch.cumsum(probs1, dim=-1)
            cdf2 = torch.cumsum(probs2, dim=-1)
            distances = (
                torch.abs(cdf1[:, :, None, :-1] - cdf2[:, None, :, :-1]).sum(dim=-1)
                / self.scale_factor
            )
            values.update(zip(group, distances.unbind()))

    def _categorical_matrix(
        self, layer1: TorchCategoricalLayer, layer2: TorchCategoricalLayer
    ) -> Tensor:
        probs1 = self._categorical_probabilities(0, layer1)
        probs2 = self._categorical_probabilities(1, layer2)
        if self.metric_p == 1.0:
            _validate_probability_rows(
                probs1,
                "categorical probabilities",
                self.probability_atol,
                self.probability_rtol,
            )
            _validate_probability_rows(
                probs2,
                "categorical probabilities",
                self.probability_atol,
                self.probability_rtol,
            )
            num_categories = max(layer1.num_categories, layer2.num_categories)
            probs1 = torch.nn.functional.pad(probs1, (0, num_categories - layer1.num_categories))
            probs2 = torch.nn.functional.pad(probs2, (0, num_categories - layer2.num_categories))
            cdf1 = torch.cumsum(probs1, dim=-1)
            cdf2 = torch.cumsum(probs2, dim=-1)
            return (
                torch.abs(cdf1[:, None, :-1] - cdf2[None, :, :-1]).sum(dim=-1) / self.scale_factor
            )

        support1 = torch.arange(layer1.num_categories, device=probs1.device, dtype=probs1.dtype)
        support2 = torch.arange(layer2.num_categories, device=probs2.device, dtype=probs2.dtype)
        cost = (
            torch.abs(support1.unsqueeze(1) - support2.unsqueeze(0)).pow(self.metric_p)
            / self.scale_factor
        )
        num_units1, num_units2 = probs1.shape[0], probs2.shape[0]
        return self.transport_solver(
            cost.expand(num_units1, num_units2, -1, -1),
            probs1[:, None, :].expand(-1, num_units2, -1),
            probs2[None, :, :].expand(num_units1, -1, -1),
        )

    def _gaussian_matrix(self, layer1: TorchGaussianLayer, layer2: TorchGaussianLayer) -> Tensor:
        if layer1.log_partition is not None or layer2.log_partition is not None:
            raise ValueError("Circuit-Wasserstein requires normalized Gaussian leaves")
        mean1 = self._parameter(0, layer1, "mean")
        mean2 = self._parameter(1, layer2, "mean")
        stddev1 = self._parameter(0, layer1, "stddev")
        stddev2 = self._parameter(1, layer2, "stddev")
        for value, name in (
            (mean1, "Gaussian mean"),
            (mean2, "Gaussian mean"),
            (stddev1, "Gaussian stddev"),
            (stddev2, "Gaussian stddev"),
        ):
            if not torch.all(torch.isfinite(value)).item():
                raise ValueError(f"{name} must be finite")
        if torch.any(stddev1 <= 0.0).item() or torch.any(stddev2 <= 0.0).item():
            raise ValueError("Gaussian standard deviations must be positive")
        return (
            torch.square(mean1[:, None] - mean2[None, :])
            + torch.square(stddev1[:, None] - stddev2[None, :])
        ) / self.scale_factor

    def _product_matrix(
        self,
        layer1: TorchProductLayer,
        layer2: TorchProductLayer,
        values: dict[LayerPair, Tensor],
    ) -> Tensor:
        result: Tensor | None = None
        identity_units = isinstance(layer1, TorchHadamardLayer) and isinstance(
            layer2, TorchHadamardLayer
        )
        for dependency, units1, units2 in self._product_specs[(layer1, layer2)]:
            child_values = values[dependency]
            if not identity_units:
                child_values = child_values[list(units1)][:, list(units2)]
            result = child_values if result is None else result + child_values
        assert result is not None
        return result

    def _evaluate_sum_pairs(
        self,
        pairs: Sequence[tuple[TorchSumLayer, TorchSumLayer]],
        values: dict[LayerPair, Tensor],
    ) -> None:
        groups: dict[tuple[int, int, int, int], list[tuple[TorchSumLayer, TorchSumLayer]]]
        groups = defaultdict(list)
        for layer1, layer2 in pairs:
            num_inputs1 = sum(
                child.num_output_units for child in self.circuits[0].layer_inputs(layer1)
            )
            num_inputs2 = sum(
                child.num_output_units for child in self.circuits[1].layer_inputs(layer2)
            )
            groups[
                (
                    layer1.num_output_units,
                    layer2.num_output_units,
                    num_inputs1,
                    num_inputs2,
                )
            ].append((layer1, layer2))

        for (num_units1, num_units2, _, _), group in groups.items():
            costs: list[Tensor] = []
            supplies: list[Tensor] = []
            demands: list[Tensor] = []
            for layer1, layer2 in group:
                inputs1 = self.circuits[0].layer_inputs(layer1)
                inputs2 = self.circuits[1].layer_inputs(layer2)
                if len(inputs1) == len(inputs2) == 1:
                    cost = values[(inputs1[0], inputs2[0])]
                else:
                    cost = torch.cat(
                        [
                            torch.cat(
                                [values[(child1, child2)] for child2 in inputs2], dim=1
                            )
                            for child1 in inputs1
                        ],
                        dim=0,
                    )
                costs.append(cost)
                supplies.append(self._sum_weights(0, layer1))
                demands.append(self._sum_weights(1, layer2))

            results = self.transport_solver(
                torch.stack(costs)[:, None, None].expand(
                    -1, num_units1, num_units2, -1, -1
                ),
                torch.stack(supplies)[:, :, None].expand(-1, -1, num_units2, -1),
                torch.stack(demands)[:, None].expand(-1, num_units1, -1, -1),
            )
            values.update(zip(group, results.unbind()))

    def _build_batched_softmax_groups(
        self,
    ) -> list[tuple[int, int, list[tuple[TorchLayer, str, TorchTensorParameter]]]]:
        groups: dict[
            tuple[int, tuple[int, ...], int],
            list[tuple[TorchLayer, str, TorchTensorParameter]],
        ] = defaultdict(list)
        for side, circuit in enumerate(self.circuits):
            for layer in circuit.layers:
                for name, parameter in layer.params.items():
                    nodes = list(parameter.nodes)
                    if (
                        len(nodes) == 2
                        and isinstance(nodes[0], TorchTensorParameter)
                        and isinstance(nodes[1], TorchSoftmaxParameter)
                    ):
                        groups[(side, parameter.shape, nodes[1].dim)].append(
                            (layer, name, nodes[0])
                        )
        return [
            (side, dim, group)
            for (side, _, dim), group in groups.items()
            if len(group) > 1
        ]

    def _evaluate_batched_softmaxes(self) -> None:
        for side, dim, group in self._batched_softmax_groups:
            raw = torch.cat(
                [next(node.parameters(recurse=False)) for _, _, node in group]
            )
            values = torch.softmax(raw, dim=dim + 1)
            self._register_parameter_tensor(values)
            self._parameter_cache[side].update(
                zip(((layer, name) for layer, name, _ in group), values.unbind())
            )

    def _parameter(self, side: int, layer: TorchLayer, name: str) -> Tensor:
        key = (layer, name)
        cache = self._parameter_cache[side]
        if key not in cache:
            value = layer.params[name]()
            if value.shape[0] != 1:
                raise ValueError("Circuit-Wasserstein requires unfolded parameters")
            value = value[0]
            self._register_parameter_tensor(value)
            cache[key] = value
        return cache[key]

    def _register_parameter_tensor(self, value: Tensor) -> None:
        if not value.is_floating_point():
            raise ValueError("Circuit parameters must be floating-point tensors")
        if self._reference_tensor is None:
            self._reference_tensor = value
        elif (
            value.device != self._reference_tensor.device
            or value.dtype != self._reference_tensor.dtype
        ):
            raise ValueError("Both circuits must use the same parameter device and dtype")

    def _categorical_probabilities(self, side: int, layer: TorchCategoricalLayer) -> Tensor:
        if layer.logits is not None:
            return torch.softmax(self._parameter(side, layer, "logits"), dim=-1)
        return self._parameter(side, layer, "probs")

    def _sum_weights(self, side: int, layer: TorchSumLayer) -> Tensor:
        return self._parameter(side, layer, "weight")

    def _sum_children(self, side: int, layer: TorchSumLayer) -> list[tuple[TorchLayer, int]]:
        return [
            (child, unit)
            for child in self.circuits[side].layer_inputs(layer)
            for unit in range(child.num_output_units)
        ]

    def _product_children(
        self, side: int, layer: TorchProductLayer, unit: int
    ) -> list[tuple[TorchLayer, int]]:
        inputs = list(self.circuits[side].layer_inputs(layer))
        if isinstance(layer, TorchHadamardLayer):
            units = [unit] * layer.arity
        else:
            units = [0] * layer.arity
            remainder = unit
            for index in reversed(range(layer.arity)):
                units[index] = remainder % layer.num_input_units
                remainder //= layer.num_input_units
        return list(zip(inputs, units))

    def _match_product_children(
        self,
        children1: Sequence[tuple[TorchLayer, int]],
        children2: Sequence[tuple[TorchLayer, int]],
    ) -> list[tuple[TorchLayer, int, TorchLayer, int]]:
        by_scope2: dict[Scope, tuple[TorchLayer, int]] = {}
        for child2, unit2 in children2:
            scope = self._scopes[1][child2]
            if scope in by_scope2:
                raise ValueError(f"Product has duplicate child scope {scope}")
            by_scope2[scope] = (child2, unit2)
        matched: list[tuple[TorchLayer, int, TorchLayer, int]] = []
        seen: set[Scope] = set()
        for child1, unit1 in children1:
            scope = self._scopes[0][child1]
            if scope in seen:
                raise ValueError(f"Product has duplicate child scope {scope}")
            seen.add(scope)
            if scope not in by_scope2:
                raise ValueError(f"Products do not share child scope {scope}")
            child2, unit2 = by_scope2[scope]
            matched.append((child1, unit1, child2, unit2))
        if len(matched) != len(children2):
            raise ValueError("Products have different scope factorizations")
        return matched


def _validate_finite_nonnegative(value: Tensor, name: str) -> None:
    if not torch.all(torch.isfinite(value)).item():
        raise ValueError(f"{name.capitalize()} must be finite")
    if torch.any(value < 0.0).item():
        raise ValueError(f"{name.capitalize()} must be non-negative")


def _validate_probability_vectors(value: Tensor, name: str, atol: float, rtol: float) -> None:
    _validate_finite_nonnegative(value, name)
    sums = value.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=rtol):
        raise ValueError(f"{name.capitalize()} must sum to one")


def _validate_probability_rows(value: Tensor, name: str, atol: float, rtol: float) -> None:
    _validate_finite_nonnegative(value, name)
    sums = value.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=rtol):
        raise ValueError(f"{name.capitalize()} must sum to one")


def _build_engine(
    circuit1: TorchCircuit,
    circuit2: TorchCircuit,
    *,
    metric_p: float,
    scale_factor: float,
    transport_solver: TransportSolver | None,
    probability_atol: float,
    probability_rtol: float,
) -> _CircuitWassersteinEngine | _FoldedCircuitWassersteinEngine:
    if transport_solver is None:
        transport_solver = TorchTransportSolver()
    if circuit1.is_folded != circuit2.is_folded:
        raise ValueError("Circuit-Wasserstein requires both circuits to use the same fold mode")
    engine_type = (
        _FoldedCircuitWassersteinEngine if circuit1.is_folded else _CircuitWassersteinEngine
    )
    return engine_type(
        circuit1,
        circuit2,
        metric_p=metric_p,
        scale_factor=scale_factor,
        transport_solver=transport_solver,
        probability_atol=probability_atol,
        probability_rtol=probability_rtol,
    )


def circuit_wasserstein(
    circuit1: TorchCircuit,
    circuit2: TorchCircuit,
    *,
    metric_p: float = 1.0,
    scale_factor: float = 1.0,
    transport_solver: TransportSolver | None = None,
    probability_atol: float = 1e-6,
    probability_rtol: float = 1e-5,
) -> Tensor:
    """Compute exact ``CW_p`` between two Torch circuits.

    Categorical ``W_1`` leaves are evaluated natively on their parameter device.
    Remaining transport LPs are solved on-device by ``TorchTransportSolver`` and
    autograd receives the selected first-order LP subgradient. Folded circuits
    must have aligned connectivity and unary sum layers.
    """

    engine = _build_engine(
        circuit1,
        circuit2,
        metric_p=metric_p,
        scale_factor=scale_factor,
        transport_solver=transport_solver,
        probability_atol=probability_atol,
        probability_rtol=probability_rtol,
    )
    return engine()
