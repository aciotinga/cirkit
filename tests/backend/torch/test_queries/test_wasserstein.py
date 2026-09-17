from collections.abc import Iterator
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import Tensor

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.queries import CircuitWassersteinQuery
from cirkit.backend.torch.wasserstein import HighsTransportSolver, circuit_wasserstein
from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional as SF
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import (
    CategoricalLayer,
    GaussianLayer,
    HadamardLayer,
    KroneckerLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import ConstantParameter, Parameter, SoftmaxParameter, TensorParameter
from cirkit.templates import data_modalities, utils
from cirkit.utils.scope import Scope


@pytest.fixture(autouse=True)
def use_float64() -> Iterator[None]:
    old_dtype = torch.get_default_dtype()
    old_grad_enabled = torch.is_grad_enabled()
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(True)
    try:
        yield
    finally:
        torch.set_grad_enabled(old_grad_enabled)
        torch.set_default_dtype(old_dtype)


def parameter(
    value: list[float] | list[list[float]] | np.ndarray,
    *,
    softmax: bool = False,
    learnable: bool = True,
) -> tuple[Parameter, TensorParameter]:
    array = np.asarray(value, dtype=np.float64)
    raw = TensorParameter(
        *array.shape,
        initializer=ConstantTensorInitializer(array),
        learnable=learnable,
    )
    graph = Parameter.from_input(raw)
    if softmax:
        graph = Parameter.from_unary(SoftmaxParameter(array.shape, axis=-1), graph)
    return graph, raw


def categorical_circuit(
    probabilities: list[float],
    *,
    variable: int = 0,
    learnable: bool = True,
) -> tuple[Circuit, TensorParameter]:
    probs, raw = parameter(
        [np.log(np.asarray(probabilities, dtype=np.float64))],
        softmax=True,
        learnable=learnable,
    )
    layer = CategoricalLayer(
        Scope([variable]),
        1,
        num_categories=len(probabilities),
        probs=probs,
    )
    return Circuit([layer], {}, [layer]), raw


def gaussian_circuit(
    mean_value: float,
    stddev_value: float,
    *,
    variable: int = 0,
    learnable: bool = True,
) -> tuple[Circuit, TensorParameter, TensorParameter]:
    mean, raw_mean = parameter([mean_value], learnable=learnable)
    stddev, raw_stddev = parameter([stddev_value], learnable=learnable)
    layer = GaussianLayer(
        Scope([variable]),
        1,
        mean=mean,
        stddev=stddev,
    )
    return Circuit([layer], {}, [layer]), raw_mean, raw_stddev


def product_circuit(
    probabilities1: list[float],
    probabilities2: list[float],
    *,
    product: str = "hadamard",
) -> Circuit:
    probs1, _ = parameter(
        [np.log(probabilities1)],
        softmax=True,
    )
    probs2, _ = parameter(
        [np.log(probabilities2)],
        softmax=True,
    )
    input1 = CategoricalLayer(Scope([0]), 1, num_categories=2, probs=probs1)
    input2 = CategoricalLayer(Scope([1]), 1, num_categories=2, probs=probs2)
    if product == "hadamard":
        product_layer = HadamardLayer(1, arity=2)
    elif product == "kronecker":
        product_layer = KroneckerLayer(1, arity=2)
    else:
        raise ValueError(product)
    return Circuit(
        [input1, input2, product_layer],
        {product_layer: [input1, input2]},
        [product_layer],
    )


def mixture_circuit(
    leaf_probabilities: list[list[float]],
    mixture_weights: list[float],
    *,
    product: str | None = None,
) -> tuple[Circuit, dict[str, TensorParameter]]:
    num_units = len(leaf_probabilities)
    probs, raw_probs = parameter(
        np.log(np.asarray(leaf_probabilities)),
        softmax=True,
    )
    input1 = CategoricalLayer(Scope([0]), num_units, num_categories=2, probs=probs)
    layers = [input1]
    in_layers = {}
    root_input = input1
    raw_parameters = {"probs1": raw_probs}

    if product is not None:
        probs2, raw_probs2 = parameter(
            np.log(np.asarray(list(reversed(leaf_probabilities)))),
            softmax=True,
        )
        input2 = CategoricalLayer(Scope([1]), num_units, num_categories=2, probs=probs2)
        if product == "hadamard":
            product_layer = HadamardLayer(num_units, arity=2)
        elif product == "kronecker":
            product_layer = KroneckerLayer(num_units, arity=2)
        else:
            raise ValueError(product)
        layers.extend([input2, product_layer])
        in_layers[product_layer] = [input1, input2]
        root_input = product_layer
        raw_parameters["probs2"] = raw_probs2

    weights, raw_weights = parameter(
        [np.log(mixture_weights)],
        softmax=True,
    )
    root = SumLayer(root_input.num_output_units, 1, weight=weights)
    layers.append(root)
    in_layers[root] = [root_input]
    raw_parameters["weights"] = raw_weights
    return Circuit(layers, in_layers, [root]), raw_parameters


def raw_compiled_parameter(ctx: PipelineContext, raw: TensorParameter) -> tuple[Tensor, int]:
    compiler = getattr(ctx, "_compiler")
    assert isinstance(compiler, TorchCompiler)
    node, fold_idx = compiler.state.retrieve_compiled_parameter(raw)
    return next(node.parameters()), fold_idx


def central_difference(
    query: CircuitWassersteinQuery, parameter_tensor: Tensor, index: tuple[int, ...]
):
    epsilon = 1e-6
    with torch.no_grad():
        parameter_tensor[index] += epsilon
        plus = query().item()
        parameter_tensor[index] -= 2.0 * epsilon
        minus = query().item()
        parameter_tensor[index] += epsilon
    return (plus - minus) / (2.0 * epsilon)


def test_categorical_matches_reference_value():
    circuit1, _ = categorical_circuit([0.75, 0.25])
    circuit2, _ = categorical_circuit([0.25, 0.75])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    value = circuit_wasserstein(ctx.compile(circuit1), ctx.compile(circuit2))

    # The optimal leaf coupling moves mass 0.5 by distance 1.
    assert torch.allclose(value, torch.tensor(0.5))


def test_categorical_w1_handles_different_support_sizes_and_scale_without_solver():
    circuit1, _ = categorical_circuit([0.25, 0.75])
    circuit2, _ = categorical_circuit([0.25, 0.25, 0.5])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    def unexpected_solver(cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        raise AssertionError("Categorical W1 should not call the generic transport solver")

    value = circuit_wasserstein(
        ctx.compile(circuit1),
        ctx.compile(circuit2),
        scale_factor=2.0,
        transport_solver=unexpected_solver,
    )

    assert torch.allclose(value, torch.tensor(0.25))


def test_categorical_w1_batches_all_unit_pairs():
    circuit1, _ = mixture_circuit([[0.9, 0.1], [0.2, 0.8]], [0.65, 0.35])
    circuit2, _ = mixture_circuit([[0.8, 0.2], [0.1, 0.9]], [0.25, 0.75])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    solver = HighsTransportSolver()
    solved_shapes: list[tuple[int, int]] = []

    def recording_solver(cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        solved_shapes.append(tuple(cost.shape))
        return solver(cost, supply, demand)

    value = circuit_wasserstein(
        ctx.compile(circuit1),
        ctx.compile(circuit2),
        transport_solver=recording_solver,
    )

    assert torch.allclose(value, torch.tensor(0.38), atol=1e-12, rtol=1e-12)
    assert solved_shapes == [(1, 1, 1, 2, 2)]


def test_categorical_non_w1_uses_generic_transport_solver():
    circuit1, _ = categorical_circuit([0.75, 0.25])
    circuit2, _ = categorical_circuit([0.25, 0.75])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    solver = HighsTransportSolver()
    solved_shapes: list[tuple[int, int]] = []

    def recording_solver(cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        solved_shapes.append(tuple(cost.shape))
        return solver(cost, supply, demand)

    value = circuit_wasserstein(
        ctx.compile(circuit1),
        ctx.compile(circuit2),
        metric_p=2.0,
        transport_solver=recording_solver,
    )

    assert torch.allclose(value, torch.tensor(0.5))
    assert solved_shapes == [(1, 1, 2, 2)]


def test_gaussian_w2_squared_and_scale():
    circuit1, _, _ = gaussian_circuit(0.0, 1.0)
    circuit2, _, _ = gaussian_circuit(2.0, 3.0)
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    value = circuit_wasserstein(
        ctx.compile(circuit1), ctx.compile(circuit2), metric_p=2.0, scale_factor=2.0
    )

    assert torch.allclose(value, torch.tensor(4.0))


@pytest.mark.parametrize("product", ["hadamard", "kronecker"])
def test_product_recursion_is_additive(product: str):
    circuit1 = product_circuit([0.75, 0.25], [0.25, 0.75], product=product)
    circuit2 = product_circuit([0.25, 0.75], [0.5, 0.5], product=product)
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    value = circuit_wasserstein(ctx.compile(circuit1), ctx.compile(circuit2))

    assert torch.allclose(value, torch.tensor(0.75))


def test_sum_recursion_matches_transport_reference():
    circuit1, _ = mixture_circuit([[0.999, 0.001], [0.001, 0.999]], [0.5, 0.5])
    circuit2, _ = mixture_circuit([[0.999, 0.001], [0.001, 0.999]], [0.25, 0.75])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    value = circuit_wasserstein(ctx.compile(circuit1), ctx.compile(circuit2))

    assert torch.allclose(value, torch.tensor(0.2495), atol=1e-12, rtol=1e-12)


def test_custom_solver_owns_sum_weight_validation():
    leaf_probs1, _ = parameter([[0.9, 0.1], [0.2, 0.8]])
    leaf1 = CategoricalLayer(Scope([0]), 2, num_categories=2, probs=leaf_probs1)
    weights1, _ = parameter([[0.8, 0.3]])
    root1 = SumLayer(2, 1, weight=weights1)
    circuit1 = Circuit([leaf1, root1], {root1: [leaf1]}, [root1])

    leaf_probs2, _ = parameter([[0.8, 0.2], [0.1, 0.9]])
    leaf2 = CategoricalLayer(Scope([0]), 2, num_categories=2, probs=leaf_probs2)
    weights2, _ = parameter([[0.25, 0.75]])
    root2 = SumLayer(2, 1, weight=weights2)
    circuit2 = Circuit([leaf2, root2], {root2: [leaf2]}, [root2])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    calls = 0

    def permissive_solver(cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        assert torch.allclose(supply.sum(), torch.tensor(1.1))
        return torch.sum(
            cost * supply[..., :, None] * demand[..., None, :],
            dim=(-2, -1),
        )

    value = circuit_wasserstein(
        ctx.compile(circuit1),
        ctx.compile(circuit2),
        transport_solver=permissive_solver,
    )

    assert torch.isfinite(value)
    assert calls == 1


def test_identity_and_symmetry():
    circuit1, _ = categorical_circuit([0.8, 0.2])
    circuit2, _ = categorical_circuit([0.3, 0.7])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    torch_circuit1 = ctx.compile(circuit1)
    torch_circuit2 = ctx.compile(circuit2)

    value12 = circuit_wasserstein(torch_circuit1, torch_circuit2, metric_p=2.0)
    value21 = circuit_wasserstein(torch_circuit2, torch_circuit1, metric_p=2.0)
    identity = circuit_wasserstein(torch_circuit1, torch_circuit1, metric_p=2.0)

    assert torch.allclose(value12, value21)
    assert torch.allclose(identity, torch.zeros(()))


def test_categorical_gradients_match_finite_differences_for_both_circuits():
    circuit1, raw1 = categorical_circuit([0.8, 0.2])
    circuit2, raw2 = categorical_circuit([0.3, 0.7])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    query = CircuitWassersteinQuery(ctx.compile(circuit1), ctx.compile(circuit2))
    tensor1, fold1 = raw_compiled_parameter(ctx, raw1)
    tensor2, fold2 = raw_compiled_parameter(ctx, raw2)

    value = query()
    value.backward()
    grad1 = tensor1.grad[fold1, 0, 0].item()
    grad2 = tensor2.grad[fold2, 0, 0].item()
    fd1 = central_difference(query, tensor1, (fold1, 0, 0))
    fd2 = central_difference(query, tensor2, (fold2, 0, 0))

    assert grad1 == pytest.approx(fd1, abs=1e-7, rel=1e-6)
    assert grad2 == pytest.approx(fd2, abs=1e-7, rel=1e-6)


def test_gaussian_gradients_match_finite_differences_for_both_circuits():
    circuit1, mean1, stddev1 = gaussian_circuit(0.0, 1.0)
    circuit2, mean2, stddev2 = gaussian_circuit(2.0, 3.0)
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    query = CircuitWassersteinQuery(ctx.compile(circuit1), ctx.compile(circuit2), metric_p=2.0)
    parameters = [raw_compiled_parameter(ctx, raw) for raw in (mean1, stddev1, mean2, stddev2)]

    query().backward()

    for tensor, fold_idx in parameters:
        gradient = tensor.grad[fold_idx, 0].item()
        finite_difference = central_difference(query, tensor, (fold_idx, 0))
        assert gradient == pytest.approx(finite_difference, abs=1e-7, rel=1e-6)


def test_sum_weight_gradients_match_finite_differences():
    circuit1, parameters1 = mixture_circuit([[0.9, 0.1], [0.2, 0.8]], [0.65, 0.35])
    circuit2, parameters2 = mixture_circuit([[0.8, 0.2], [0.1, 0.9]], [0.25, 0.75])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    query = CircuitWassersteinQuery(ctx.compile(circuit1), ctx.compile(circuit2))
    raw_tensors = [
        raw_compiled_parameter(ctx, parameters1["weights"]),
        raw_compiled_parameter(ctx, parameters2["weights"]),
    ]

    query().backward()

    for tensor, fold_idx in raw_tensors:
        gradient = tensor.grad[fold_idx, 0, 0].item()
        finite_difference = central_difference(query, tensor, (fold_idx, 0, 0))
        assert gradient == pytest.approx(finite_difference, abs=2e-7, rel=2e-6)


def test_sum_and_leaf_logit_gradients_match_reference():
    circuit1, _ = mixture_circuit(
        [[0.9, 0.1], [0.2, 0.8]],
        [0.65, 0.35],
    )
    circuit2, parameters2 = mixture_circuit(
        [[0.8, 0.2], [0.1, 0.9]],
        [0.25, 0.75],
    )
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    query = CircuitWassersteinQuery(ctx.compile(circuit1), ctx.compile(circuit2))
    probs_tensor, probs_fold = raw_compiled_parameter(ctx, parameters2["probs1"])
    weight_tensor, weight_fold = raw_compiled_parameter(ctx, parameters2["weights"])

    value = query()
    value.backward()

    # The probability-space leaf gradients are
    # [[-0.25, 0.0], [-0.75, 0.0]], and the sum gradients are [0.1, 0.8].
    # Projecting each through the matching softmax Jacobian yields these logits gradients.
    expected_probs_gradient = torch.tensor([[-0.04, 0.04], [-0.0675, 0.0675]])
    expected_weight_gradient = torch.tensor([[-0.13125, 0.13125]])
    assert torch.allclose(value, torch.tensor(0.38), atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        probs_tensor.grad[probs_fold],
        expected_probs_gradient,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        weight_tensor.grad[weight_fold],
        expected_weight_gradient,
        atol=1e-12,
        rtol=1e-12,
    )


def test_nll_and_cw_share_model_parameters_and_optimizer_step_reduces_loss():
    reference, _ = categorical_circuit([0.8, 0.2], learnable=False)
    model_circuit, raw_model = categorical_circuit([0.4, 0.6])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    query = CircuitWassersteinQuery(ctx.compile(reference), ctx.compile(model_circuit))
    model = query.circuit2
    raw_tensor, _ = raw_compiled_parameter(ctx, raw_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    data = torch.zeros((32, 1), dtype=torch.long)

    nll = -model(data).mean()
    cw = query()
    nll_gradient = torch.autograd.grad(nll, raw_tensor, retain_graph=True)[0]
    cw_gradient = torch.autograd.grad(cw, raw_tensor, retain_graph=True)[0]
    assert nll_gradient.norm().item() > 0.0
    assert cw_gradient.norm().item() > 0.0

    loss = nll + 0.5 * cw
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    new_loss = -model(data).mean() + 0.5 * query()
    assert new_loss.item() < loss.item()


@pytest.mark.parametrize(
    "semiring",
    ["sum-product", "lse-sum"],
)
def test_query_is_independent_of_semiring(semiring: str):
    circuit1, _ = mixture_circuit([[0.9, 0.1], [0.2, 0.8]], [0.65, 0.35], product="hadamard")
    circuit2, parameters2 = mixture_circuit(
        [[0.8, 0.2], [0.1, 0.9]], [0.25, 0.75], product="hadamard"
    )
    ctx = PipelineContext(
        backend="torch",
        semiring=semiring,
        fold=False,
        optimize=False,
    )
    query = CircuitWassersteinQuery(ctx.compile(circuit1), ctx.compile(circuit2))
    weight_tensor, weight_fold = raw_compiled_parameter(ctx, parameters2["weights"])

    value = query()
    value.backward()

    assert torch.allclose(value, torch.tensor(0.68), atol=1e-10, rtol=1e-10)
    assert weight_tensor.grad[weight_fold].norm().item() > 0.0


def test_query_reuses_live_parameters():
    circuit1, _ = categorical_circuit([0.8, 0.2])
    circuit2, raw2 = categorical_circuit([0.3, 0.7])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    query = CircuitWassersteinQuery(ctx.compile(circuit1), ctx.compile(circuit2))
    tensor2, fold2 = raw_compiled_parameter(ctx, raw2)
    before = query()

    with torch.no_grad():
        tensor2[fold2, 0] += torch.tensor([0.3, -0.3])
    after = query()

    assert not torch.allclose(before, after)


def test_sum_collapse_can_change_canonical_cw():
    leaf_probs, _ = parameter([[1.0, 0.0], [0.0, 1.0]], learnable=False)
    leaf1 = CategoricalLayer(Scope([0]), 2, num_categories=2, probs=leaf_probs)
    inner_weights1, _ = parameter(
        [[0.5, 0.5], [0.5, 0.5]],
        learnable=False,
    )
    inner1 = SumLayer(2, 2, weight=inner_weights1)
    root_weights1, _ = parameter([[1.0, 0.0]], learnable=False)
    root1 = SumLayer(2, 1, weight=root_weights1)
    nested1 = Circuit(
        [leaf1, inner1, root1],
        {inner1: [leaf1], root1: [inner1]},
        [root1],
    )

    leaf_probs2, _ = parameter([[1.0, 0.0], [0.0, 1.0]], learnable=False)
    leaf2 = CategoricalLayer(Scope([0]), 2, num_categories=2, probs=leaf_probs2)
    inner_weights2, _ = parameter(
        [[1.0, 0.0], [0.0, 1.0]],
        learnable=False,
    )
    inner2 = SumLayer(2, 2, weight=inner_weights2)
    root_weights2, _ = parameter([[0.5, 0.5]], learnable=False)
    root2 = SumLayer(2, 1, weight=root_weights2)
    nested2 = Circuit(
        [leaf2, inner2, root2],
        {inner2: [leaf2], root2: [inner2]},
        [root2],
    )

    flat1, _ = categorical_circuit([0.5, 0.5], learnable=False)
    flat2, _ = categorical_circuit([0.5, 0.5], learnable=False)
    nested_ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    flat_ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    nested_value = circuit_wasserstein(nested_ctx.compile(nested1), nested_ctx.compile(nested2))
    collapsed_value = circuit_wasserstein(flat_ctx.compile(flat1), flat_ctx.compile(flat2))

    # The pairs represent the same flat distribution. Collapsing adjacent sums
    # nevertheless removes the hierarchical transport cost.
    assert torch.allclose(nested_value, torch.tensor(0.5))
    assert torch.allclose(collapsed_value, torch.tensor(0.0))


def test_rejects_unnormalized_probabilities():
    probs1, _ = parameter([[0.8, 0.3]])
    probs2, _ = parameter([[0.3, 0.7]])
    layer1 = CategoricalLayer(Scope([0]), 1, num_categories=2, probs=probs1)
    layer2 = CategoricalLayer(Scope([0]), 1, num_categories=2, probs=probs2)
    circuit1 = Circuit([layer1], {}, [layer1])
    circuit2 = Circuit([layer2], {}, [layer2])
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    with pytest.raises(ValueError, match="sum to one"):
        circuit_wasserstein(ctx.compile(circuit1), ctx.compile(circuit2))


def test_rejects_mixed_leaf_families_and_non_scalar_outputs():
    categorical, _ = categorical_circuit([0.5, 0.5])
    gaussian, _, _ = gaussian_circuit(0.0, 1.0)
    ctx = PipelineContext(backend="torch", fold=False, optimize=False)

    with pytest.raises(ValueError, match="different operations"):
        circuit_wasserstein(ctx.compile(categorical), ctx.compile(gaussian))

    probs, _ = parameter([[0.5, 0.5], [0.4, 0.6]])
    layer = CategoricalLayer(Scope([0]), 2, num_categories=2, probs=probs)
    vector_circuit = Circuit([layer], {}, [layer])
    with pytest.raises(ValueError, match="scalar output"):
        torch_vector = ctx.compile(vector_circuit)
        circuit_wasserstein(torch_vector, torch_vector)


@pytest.mark.parametrize("product", ["hadamard", "kronecker"])
def test_folded_matches_unfolded_and_rejects_fused_torch_layers(product: str):
    weights1 = [0.65, 0.35] if product == "hadamard" else [0.4, 0.3, 0.2, 0.1]
    weights2 = [0.25, 0.75] if product == "hadamard" else [0.1, 0.2, 0.3, 0.4]
    circuit1, parameters1 = mixture_circuit(
        [[0.9, 0.1], [0.2, 0.8]], weights1, product=product
    )
    circuit2, parameters2 = mixture_circuit(
        [[0.8, 0.2], [0.1, 0.9]], weights2, product=product
    )
    raw_parameters = [*parameters1.values(), *parameters2.values()]

    unfolded_ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    unfolded1 = unfolded_ctx.compile(circuit1)
    unfolded2 = unfolded_ctx.compile(circuit2)
    expected = circuit_wasserstein(unfolded1, unfolded2)
    expected.backward()
    expected_gradients = [
        tensor.grad[fold_idx].clone()
        for raw in raw_parameters
        for tensor, fold_idx in [raw_compiled_parameter(unfolded_ctx, raw)]
    ]

    folded_ctx = PipelineContext(backend="torch", fold=True, optimize=False)
    folded1 = folded_ctx.compile(circuit1)
    folded2 = folded_ctx.compile(circuit2)
    actual = circuit_wasserstein(folded1, folded2)

    assert torch.allclose(actual, expected)
    actual.backward()
    for raw, expected_gradient in zip(raw_parameters, expected_gradients):
        tensor, fold_idx = raw_compiled_parameter(folded_ctx, raw)
        assert torch.allclose(tensor.grad[fold_idx], expected_gradient)

    with pytest.raises(ValueError, match="same fold mode"):
        circuit_wasserstein(folded1, unfolded2)

    fused_ctx = PipelineContext(backend="torch", fold=False, optimize=True)
    with pytest.raises(NotImplementedError, match="Unsupported Torch layer"):
        circuit_wasserstein(fused_ctx.compile(circuit1), fused_ctx.compile(circuit2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("fold", [False, True])
def test_query_preserves_cuda_device(fold: bool):
    circuit1, _ = categorical_circuit([0.8, 0.2])
    circuit2, _ = categorical_circuit([0.3, 0.7])
    ctx = PipelineContext(backend="torch", fold=fold, optimize=False)
    query = CircuitWassersteinQuery(ctx.compile(circuit1), ctx.compile(circuit2))
    query.circuit1.to("cuda")
    query.circuit2.to("cuda")

    value = query()

    assert value.device.type == "cuda"
    value.backward()


def test_tabular_product_marginal_cw_and_r_theta_gradients() -> None:
    torch.manual_seed(0)
    num_features = 2
    label_var = num_features
    p = data_modalities.tabular_data(
        region_graph="random-binary-tree",
        num_features=num_features + 1,
        input_layers={"name": "categorical", "args": {"num_categories": 2}},
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        sum_weight_param=utils.Parameterization(activation="softmax", initialization="normal"),
    )
    r = deepcopy(p)
    for layer in r.input_layers:
        if layer.scope != Scope([label_var]):
            continue
        assert isinstance(layer, CategoricalLayer)
        layer.probs = None
        layer.logits = Parameter.from_input(
            ConstantParameter(layer.num_output_units, layer.num_categories, value=0.0)
        )

    q = SF.normalize(SF.multiply(p, r))
    p_x = SF.normalize(SF.integrate(p, scope=Scope([label_var])))
    q_x = SF.normalize(SF.integrate(q, scope=Scope([label_var])))

    ctx = PipelineContext(backend="torch", fold=False, optimize=False)
    p_torch = ctx.compile(p)
    r_torch = ctx.compile(r)
    for parameter in p_torch.parameters():
        parameter.requires_grad_(False)

    query = CircuitWassersteinQuery(ctx.compile(p_x), ctx.compile(q_x))
    value = query()
    assert torch.isfinite(value)
    value.backward()

    r_gradients = [
        parameter.grad
        for parameter in r_torch.parameters()
        if parameter.requires_grad
    ]
    assert r_gradients
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0.0
        for gradient in r_gradients
    )
    assert all(
        parameter.grad is None or parameter.grad.abs().sum() == 0.0
        for parameter in p_torch.parameters()
    )
