import subprocess
import pytest
import socket

import torch

num_gpu = torch.cuda.device_count()


def get_random_available_port_list(num_port):
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    ports = []

    while len(ports) < num_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            new_port = s.getsockname()[1]

        if new_port not in ports:
            ports.append(new_port)

    return ports


def get_random_available_port(num_port):
    return get_random_available_port_list(num_port)[0]


def gpus_to_use(num_nodes, num_gpu, rank):
    return ",".join(map(str, range(rank * num_gpu, (rank + 1) * num_gpu)))


def _test_torchrun(num_gpus, config, extra_args=[]):
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "src/zeroband/train.py",
        f"@configs/{config}",
        *extra_args,
    ]

    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process {result} failed {result}")


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_train(num_gpus):
    _test_torchrun(num_gpus=num_gpus, config="debug/normal.toml")
