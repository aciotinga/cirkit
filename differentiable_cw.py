import statistics
import time
from typing import Any

import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.queries import CircuitWassersteinQuery
from cirkit.pipeline import PipelineContext
from cirkit.templates import data_modalities, utils

WARMUP_RUNS = 1
PROFILE_RUNS = 5


def random_circuit() -> Any:
    return data_modalities.image_data(
        (1, 4, 4),
        region_graph="quad-tree-2",
        input_layer="categorical",
        num_input_units=1,
        sum_product_layer="cp",
        num_sum_units=1,
        sum_weight_param=utils.Parameterization(
            activation="softmax",
            initialization="normal",
        ),
    )


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile() -> None:
    device = torch.device("cuda")
    pipeline: PipelineContext[TorchCircuit] = PipelineContext(
        backend="torch", fold=False, optimize=False
    )

    synchronize()
    start = time.perf_counter()
    torch.manual_seed(0)
    circuit1 = pipeline.compile(random_circuit()).to(device)
    torch.manual_seed(1)
    circuit2 = pipeline.compile(random_circuit()).to(device)
    synchronize()
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    query = CircuitWassersteinQuery(circuit1, circuit2)
    query_seconds = time.perf_counter() - start

    for _ in range(WARMUP_RUNS):
        circuit1.zero_grad(set_to_none=True)
        circuit2.zero_grad(set_to_none=True)
        query().backward()
    synchronize()

    forward_times: list[float] = []
    backward_times: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(PROFILE_RUNS):
        circuit1.zero_grad(set_to_none=True)
        circuit2.zero_grad(set_to_none=True)

        synchronize()
        start = time.perf_counter()
        cw = query()
        synchronize()
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        cw.backward()
        synchronize()
        backward_times.append(time.perf_counter() - start)

    total_times = [forward + backward for forward, backward in zip(forward_times, backward_times)]
    num_parameters = sum(parameter.numel() for parameter in circuit1.parameters())

    print(f"Device:           {torch.cuda.get_device_name(device)}")
    print(f"Layers:           {len(circuit1.layers):,}")
    print(f"Parameters:       {num_parameters:,}")
    print(f"Compile:          {compile_seconds:.4f} s")
    print(f"Query setup:      {query_seconds:.4f} s")
    print(f"Forward median:   {statistics.median(forward_times):.4f} s")
    print(f"Backward median:  {statistics.median(backward_times):.4f} s")
    print(f"Total median:     {statistics.median(total_times):.4f} s")
    print(f"Total minimum:    {min(total_times):.4f} s")
    print(f"Peak GPU memory:  {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")
    print(f"CW_p:             {cw.item():.6f}")


if __name__ == "__main__":
    profile()
