import pytest
import torch

from cirkit.backend.torch.wasserstein import HighsTransportSolver
from solver import TorchTransportSolver, TorchTransportSolver2


@pytest.mark.parametrize("solver_type", [TorchTransportSolver, TorchTransportSolver2])
@pytest.mark.parametrize("num_units", [4, 8])
def test_torch_transport_solver_matches_highs_value_and_gradients(
    num_units: int, solver_type: type[TorchTransportSolver] | type[TorchTransportSolver2]
):
    with torch.enable_grad():
        torch.manual_seed(0)
        cost = torch.rand((3, num_units, num_units), dtype=torch.float64, requires_grad=True)
        supply = torch.softmax(
            torch.randn((3, num_units), dtype=torch.float64), dim=-1
        ).requires_grad_()
        demand = torch.softmax(
            torch.randn((3, num_units), dtype=torch.float64), dim=-1
        ).requires_grad_()

        expected = HighsTransportSolver()(cost, supply, demand)
        expected_gradients = torch.autograd.grad(
            expected.sum(), (cost, supply, demand), retain_graph=True
        )
        actual = solver_type(validate=True)(cost, supply, demand)
        actual_gradients = torch.autograd.grad(actual.sum(), (cost, supply, demand))

        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            assert torch.allclose(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)
