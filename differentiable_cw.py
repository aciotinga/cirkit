import argparse
import statistics
import time
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.queries import CircuitWassersteinQuery
from cirkit.backend.torch.wasserstein import (
    HighsTransportSolver,
    TorchTransportSolver,
    TransportSolver,
)
from cirkit.pipeline import PipelineContext
from cirkit.templates import data_modalities, utils

WARMUP_RUNS = 1
PROFILE_RUNS = 5


class ProfilingTransportSolver:
    def __init__(self, solver: TransportSolver) -> None:
        self.solver = solver
        self.times: dict[tuple[int, int], list[float]] = defaultdict(list)
        self.problems: dict[tuple[int, int], int] = defaultdict(int)

    def __call__(self, cost: Tensor, supply: Tensor, demand: Tensor) -> Tensor:
        start = time.perf_counter()
        value = self.solver(cost, supply, demand)
        shape = tuple(cost.shape[-2:])
        self.times[shape].append(time.perf_counter() - start)
        self.problems[shape] += cost[..., 0, 0].numel()
        return value

    def close(self) -> None:
        close = getattr(self.solver, "close", None)
        if close is not None:
            close()


def make_transport_solver(name: str | None) -> TransportSolver:
    if name == "gurobi":
        from solver import GurobiTransportSolver

        return GurobiTransportSolver()
    if name == "highs":
        return HighsTransportSolver(atol=1e-6, rtol=1e-5)
    return TorchTransportSolver()


def random_circuit(side: int, units: int) -> Any:
    return data_modalities.image_data(
        (1, side, side),
        region_graph="quad-tree-2",
        input_layer="categorical",
        num_input_units=units,
        sum_product_layer="cp",
        num_sum_units=units,
        sum_weight_param=utils.Parameterization(
            activation="softmax",
            initialization="normal",
        ),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_profiler_summary(prof: torch.profiler.profile, top_k: int = 20) -> None:
    print("\n--- torch.profiler (sorted by self CUDA time) ---")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=top_k,
        )
    )
    print("\n--- torch.profiler (sorted by total CUDA time) ---")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=top_k,
        )
    )


def profile(
    device: torch.device,
    side: int,
    units: int,
    profile_runs: int,
    solver_name: str | None,
    *,
    use_profiler: bool = False,
) -> None:
    pipeline: PipelineContext[TorchCircuit] = PipelineContext(
        backend="torch", fold=True, optimize=False
    )

    synchronize(device)
    start = time.perf_counter()
    torch.manual_seed(0)
    circuit1 = pipeline.compile(random_circuit(side, units)).to(device)
    torch.manual_seed(1)
    circuit2 = pipeline.compile(random_circuit(side, units)).to(device)
    synchronize(device)
    compile_seconds = time.perf_counter() - start

    transport_solver = ProfilingTransportSolver(make_transport_solver(solver_name))
    start = time.perf_counter()
    query = CircuitWassersteinQuery(circuit1, circuit2, transport_solver=transport_solver)
    query_seconds = time.perf_counter() - start

    for _ in range(WARMUP_RUNS):
        circuit1.zero_grad(set_to_none=True)
        circuit2.zero_grad(set_to_none=True)
        query().backward()
    synchronize(device)
    transport_solver.times.clear()
    transport_solver.problems.clear()

    forward_times: list[float] = []
    backward_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(profile_runs):
        circuit1.zero_grad(set_to_none=True)
        circuit2.zero_grad(set_to_none=True)

        synchronize(device)
        start = time.perf_counter()
        cw = query()
        synchronize(device)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        cw.backward()
        synchronize(device)
        backward_times.append(time.perf_counter() - start)

    total_times = [forward + backward for forward, backward in zip(forward_times, backward_times)]
    num_parameters = sum(parameter.numel() for parameter in circuit1.parameters())
    solver_seconds = sum(map(sum, transport_solver.times.values()))
    solver_batches = sum(map(len, transport_solver.times.values()))
    solver_problems = sum(transport_solver.problems.values())

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device:           {device_name}")
    print(f"Solver:           {solver_name or 'default'}")
    print(f"Image:            {side}x{side}")
    print(f"Units/layer:      {units}")
    print(f"Layers:           {len(circuit1.layers):,}")
    print(f"Parameters:       {num_parameters:,}")
    print(f"Compile:          {compile_seconds:.4f} s")
    print(f"Query setup:      {query_seconds:.4f} s")
    forward_median = statistics.median(forward_times)
    backward_median = statistics.median(backward_times)
    print(f"Forward median:   {forward_median:.4f} s")
    print(f"Backward median:  {backward_median:.4f} s")
    print(f"Backward/Forward: {backward_median / forward_median:.2f}x")
    print(f"Total median:     {statistics.median(total_times):.4f} s")
    print(f"Total minimum:    {min(total_times):.4f} s")
    print(f"Solver batches/run: {solver_batches / profile_runs:,.0f}")
    print(f"Solver LPs/run:     {solver_problems / profile_runs:,.0f}")
    solver_time_per_run = solver_seconds / profile_runs
    print(f"Solver time/run:  {solver_time_per_run:.4f} s")
    print(
        f"Solver share:     {100 * solver_time_per_run / forward_median:.1f}% of forward, "
        f"{100 * solver_time_per_run / (forward_median + backward_median):.1f}% of total"
    )
    for shape, times in sorted(transport_solver.times.items()):
        print(
            f"  {shape[0]}x{shape[1]} LPs: "
            f"{transport_solver.problems[shape] / profile_runs:,.0f} in "
            f"{len(times) / profile_runs:,.0f} batches/run, "
            f"{sum(times) / profile_runs:.4f} s/run"
        )
    if device.type == "cuda":
        print(f"Peak GPU memory:  {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MiB")
    print(f"CW_p:             {cw.item():.6f}")

    if use_profiler:
        circuit1.zero_grad(set_to_none=True)
        circuit2.zero_grad(set_to_none=True)
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            cw = query()
            cw.backward()
        synchronize(device)
        print_profiler_summary(prof)

    transport_solver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--units", type=int, default=1)
    parser.add_argument("--runs", type=int, default=PROFILE_RUNS)
    parser.add_argument("--solver", choices=("gurobi", "highs", "torch"), default="torch")
    parser.add_argument("--profiler", action="store_true")
    args = parser.parse_args()
    profile(
        torch.device(args.device),
        args.side,
        args.units,
        args.runs,
        args.solver,
        use_profiler=args.profiler,
    )
