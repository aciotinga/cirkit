# Exact and differentiable Circuit-Wasserstein query

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

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
from cirkit.utils.scope import Scope


class TransportSolver(Protocol):
    # Generic optimal transport solver (has to be balanced)

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        """Return the scalar transport value with gradients of the transport plan"""


class HighsTransportSolver:
    # Solve balanced OT with HiGHS and return LP subgradient

    def __init__(self, *, atol: float = 1e-7, rtol: float = 1e-6) -> None:
        assert atol >= 0.0 and rtol >= 0.0, "Transport tolerances must be non-negative"
        self.atol = atol
        self.rtol = rtol

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        # Cost matrix is a 2d tensor denoting the cost of transporting from i to j as cost[i,j]
        assert cost.ndim == 2, "Expected a cost matrix, found shape {cost.shape}"
        n, m = cost.shape
        assert supply.shape == (n,) and demand.shape == (m,), "Expected marginal shapes {(n,)} and {(m,)}, found {supply.shape} and {demand.shape}"
        assert cost.device == supply.device == demand.device, "Transport costs and marginals must be on the same device"
        assert cost.dtype == supply.dtype == demand.dtype, "Transport costs and marginals must have the same dtype"
        assert cost.is_floating_point(), "Transport inputs must be floating-point tensors"
        _validate_finite_nonnegative(cost, "transport costs")
        _validate_probability_vector(supply, "transport supply", self.atol, self.rtol)
        _validate_probability_vector(demand, "transport demand", self.atol, self.rtol)

        cost_np = cost.detach().double().cpu().numpy()
        supply_np = supply.detach().double().cpu().numpy()
        demand_np = demand.detach().double().cpu().numpy()
        if not np.isclose(supply_np.sum(), demand_np.sum(), atol=self.atol, rtol=self.rtol):
            raise ValueError("Balanced transport requires equal total mass")

        result = linprog(
            cost_np.reshape(-1),
            A_eq=_transport_equality_matrix(n, m),
            b_eq=np.concatenate((supply_np, demand_np[:-1])),
            bounds=(0.0, None),
            method="highs-ds",
        )
        if not result.success:
            raise RuntimeError(f"HiGHS failed to solve balanced transport: {result.message}")

        plan = torch.as_tensor(result.x.reshape(n, m), device=cost.device, dtype=cost.dtype)
        duals = np.asarray(result.eqlin.marginals)
        row_dual = torch.as_tensor(duals[:n], device=cost.device, dtype=cost.dtype)
        column_dual = torch.as_tensor(
            np.concatenate((duals[n:], np.zeros(1))),
            device=cost.device,
            dtype=cost.dtype,
        )
        value = torch.as_tensor(result.fun, device=cost.device, dtype=cost.dtype)
        return (
            value
            + torch.sum((cost - cost.detach()) * plan)
            + torch.sum((supply - supply.detach()) * row_dual)
            + torch.sum((demand - demand.detach()) * column_dual)
        )


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


class _CircuitWassersteinEngine:  # pylint: disable=too-many-instance-attributes
    # CW recursion over scalar units of two unfolded Torch circuits

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
        assert probability_atol >= 0.0 and probability_rtol >= 0.0, "Probability tolerances must be non-negative"
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
        self._parameter_cache: tuple[
            dict[tuple[TorchLayer, str], Tensor],
            dict[tuple[TorchLayer, str], Tensor],
        ]
        self._categorical_w1_cache: dict[
            tuple[TorchCategoricalLayer, TorchCategoricalLayer], Tensor
        ]
        self._reference_tensor: Tensor | None

    def __call__(self) -> Tensor:
        self._parameter_cache = ({}, {})
        self._categorical_w1_cache = {}
        self._reference_tensor = None
        memo: dict[tuple[TorchLayer, int, TorchLayer, int], Tensor] = {}
        return self._couple(
            self.circuits[0].outputs[0],
            0,
            self.circuits[1].outputs[0],
            0,
            memo,
        )

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

    def _couple(
        self,
        layer1: TorchLayer,
        unit1: int,
        layer2: TorchLayer,
        unit2: int,
        memo: dict[tuple[TorchLayer, int, TorchLayer, int], Tensor],
    ) -> Tensor:
        key = (layer1, unit1, layer2, unit2)
        if key in memo:
            return memo[key]
        
        # Make subcalls to the appropriate CW functions
        if isinstance(layer1, TorchInputLayer) and isinstance(layer2, TorchInputLayer):
            value = self._input(layer1, unit1, layer2, unit2)
        elif isinstance(layer1, TorchSumLayer) and isinstance(layer2, TorchSumLayer):
            value = self._sum(layer1, unit1, layer2, unit2, memo)
        elif isinstance(layer1, (TorchHadamardLayer, TorchKroneckerLayer)) and isinstance(
            layer2, (TorchHadamardLayer, TorchKroneckerLayer)
        ):
            value = self._product(layer1, unit1, layer2, unit2, memo)
        else:
            # Problem :O
            raise RuntimeError("Internal Circuit-Wasserstein operation mismatch")
        memo[key] = value
        return value

    def _input(
        self,
        layer1: TorchInputLayer,
        unit1: int,
        layer2: TorchInputLayer,
        unit2: int,
    ) -> Tensor:
        # Handle CW distances for different layer types
        if isinstance(layer1, TorchCategoricalLayer) and isinstance(layer2, TorchCategoricalLayer):
            return self._categorical(layer1, unit1, layer2, unit2)
        if isinstance(layer1, TorchGaussianLayer) and isinstance(layer2, TorchGaussianLayer):
            return self._gaussian(layer1, unit1, layer2, unit2)
        raise RuntimeError("Internal Circuit-Wasserstein input mismatch")

    def _categorical(
        self,
        layer1: TorchCategoricalLayer,
        unit1: int,
        layer2: TorchCategoricalLayer,
        unit2: int,
    ) -> Tensor:
        if self.metric_p == 1.0:
            key = (layer1, layer2)
            if key not in self._categorical_w1_cache:
                probs1 = self._categorical_probabilities(0, layer1)
                probs2 = self._categorical_probabilities(1, layer2)
                num_categories = max(layer1.num_categories, layer2.num_categories)
                probs1 = torch.nn.functional.pad(
                    probs1, (0, num_categories - layer1.num_categories)
                )
                probs2 = torch.nn.functional.pad(
                    probs2, (0, num_categories - layer2.num_categories)
                )
                cdf1 = torch.cumsum(probs1, dim=-1)
                cdf2 = torch.cumsum(probs2, dim=-1)
                self._categorical_w1_cache[key] = (
                    torch.abs(cdf1[:, None, :-1] - cdf2[None, :, :-1]).sum(dim=-1)
                    / self.scale_factor
                )
            return self._categorical_w1_cache[key][unit1, unit2]

        # Fall back to generic discrete OT for other ground-cost exponents
        probs1 = self._categorical_probabilities(0, layer1)[unit1]
        probs2 = self._categorical_probabilities(1, layer2)[unit2]
        support1 = torch.arange(layer1.num_categories, device=probs1.device, dtype=probs1.dtype)
        support2 = torch.arange(layer2.num_categories, device=probs2.device, dtype=probs2.dtype)
        cost = (
            torch.abs(support1.unsqueeze(1) - support2.unsqueeze(0)).pow(self.metric_p)
            / self.scale_factor
        )
        return self.transport_solver(cost, probs1, probs2)

    def _gaussian(
        self,
        layer1: TorchGaussianLayer,
        unit1: int,
        layer2: TorchGaussianLayer,
        unit2: int,
    ) -> Tensor:
        if layer1.log_partition is not None or layer2.log_partition is not None:
            raise ValueError("Circuit-Wasserstein requires normalized Gaussian leaves")
        # Use the closed-form formula for Gaussian Wasserstein distance (this is when p=2)
        mean1 = self._parameter(0, layer1, "mean")[unit1]
        mean2 = self._parameter(1, layer2, "mean")[unit2]
        stddev1 = self._parameter(0, layer1, "stddev")[unit1]
        stddev2 = self._parameter(1, layer2, "stddev")[unit2]
        for value, name in (
            (mean1, "Gaussian mean"),
            (mean2, "Gaussian mean"),
            (stddev1, "Gaussian stddev"),
            (stddev2, "Gaussian stddev"),
        ):
            if not torch.isfinite(value).item():
                raise ValueError(f"{name} must be finite")
        if stddev1.item() <= 0.0 or stddev2.item() <= 0.0:
            raise ValueError("Gaussian standard deviations must be positive")
        return (torch.square(mean1 - mean2) + torch.square(stddev1 - stddev2)) / self.scale_factor

    def _sum(
        self,
        layer1: TorchSumLayer,
        unit1: int,
        layer2: TorchSumLayer,
        unit2: int,
        memo: dict[tuple[TorchLayer, int, TorchLayer, int], Tensor],
    ) -> Tensor:
        # Child distance pairs are the cost matrix, marginals are the sum node weights
        children1 = self._sum_children(0, layer1)
        children2 = self._sum_children(1, layer2)
        costs = torch.stack(
            [
                torch.stack(
                    [
                        self._couple(child1, child_unit1, child2, child_unit2, memo)
                        for child2, child_unit2 in children2
                    ]
                )
                for child1, child_unit1 in children1
            ]
        )
        weights1 = self._sum_weights(0, layer1)[unit1]
        weights2 = self._sum_weights(1, layer2)[unit2]
        return self.transport_solver(costs, weights1, weights2)

    def _product(
        self,
        layer1: TorchProductLayer,
        unit1: int,
        layer2: TorchProductLayer,
        unit2: int,
        memo: dict[tuple[TorchLayer, int, TorchLayer, int], Tensor],
    ) -> Tensor:
        # Match children by scope and add distances (L_p^p is separable metric)
        matched = self._match_product_children(
            self._product_children(0, layer1, unit1),
            self._product_children(1, layer2, unit2),
        )
        return torch.stack(
            [
                self._couple(child1, child_unit1, child2, child_unit2, memo)
                for child1, child_unit1, child2, child_unit2 in matched
            ]
        ).sum()

    def _parameter(self, side: int, layer: TorchLayer, name: str) -> Tensor:
        key = (layer, name)
        cache = self._parameter_cache[side]
        if key not in cache:
            value = layer.params[name]()
            if value.shape[0] != 1:
                raise ValueError("Circuit-Wasserstein requires unfolded parameters")
            value = value[0]
            if not value.is_floating_point():
                raise ValueError("Circuit parameters must be floating-point tensors")
            if self._reference_tensor is None:
                self._reference_tensor = value
            elif (
                value.device != self._reference_tensor.device
                or value.dtype != self._reference_tensor.dtype
            ):
                raise ValueError("Both circuits must use the same parameter device and dtype")
            cache[key] = value
        return cache[key]

    def _categorical_probabilities(self, side: int, layer: TorchCategoricalLayer) -> Tensor:
        if layer.logits is not None:
            probabilities = torch.softmax(self._parameter(side, layer, "logits"), dim=-1)
        else:
            probabilities = self._parameter(side, layer, "probs")
        _validate_probability_rows(
            probabilities,
            "categorical probabilities",
            self.probability_atol,
            self.probability_rtol,
        )
        return probabilities

    def _sum_weights(self, side: int, layer: TorchSumLayer) -> Tensor:
        weights = self._parameter(side, layer, "weight")
        _validate_probability_rows(
            weights,
            "sum weights",
            self.probability_atol,
            self.probability_rtol,
        )
        return weights

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


def _validate_probability_vector(value: Tensor, name: str, atol: float, rtol: float) -> None:
    _validate_finite_nonnegative(value, name)
    one = torch.ones((), device=value.device, dtype=value.dtype)
    if not torch.isclose(value.sum(), one, atol=atol, rtol=rtol).item():
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
) -> _CircuitWassersteinEngine:
    if transport_solver is None:
        transport_solver = HighsTransportSolver(atol=probability_atol, rtol=probability_rtol)
    return _CircuitWassersteinEngine(
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
    """Compute exact ``CW_p`` between two unfolded Torch circuits.

    Categorical ``W_1`` leaves are evaluated natively on their parameter device.
    HiGHS solves the remaining transport LPs on CPU and autograd receives the
    selected first-order LP subgradient.
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
