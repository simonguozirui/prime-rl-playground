"""
torch distribted test

this test are different from the torchrun integration tests

They manually do the job of torchrun to start the distributed process making it easy to write unit tests
"""

import torch.distributed as dist
import torch
import pytest
from torch.distributed import destroy_process_group, init_process_group


import os
from unittest import mock
import socket
from contextlib import contextmanager
import multiprocessing
import gc


@pytest.fixture(autouse=True)
def memory_cleanup():
    # credits to : https://github.com/pytorch/pytorch/issues/82218#issuecomment-1675254117
    try:
        gc.collect()
        torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def get_random_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def random_available_port():
    return get_random_available_port()


@contextmanager
def dist_environment(random_available_port, local_rank=0, world_size=1):
    with mock.patch.dict(
        os.environ,
        {
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "RANK": str(local_rank),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(random_available_port),
        },
    ):
        try:
            init_process_group()
            torch.cuda.set_device(local_rank)
            yield
        finally:
            destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
def test_all_reduce(world_size, random_available_port):
    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, local_rank=rank, world_size=world_size):
            print(f"os.environ['LOCAL_RANK'] {os.environ['WORLD_SIZE']}")
            data = (rank + 1) * torch.ones(10, 10).to("cuda")
            print(data.mean())
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            print(data.mean())
            assert data.mean() == sum([i + 1 for i in range(world_size)])

    processes = [multiprocessing.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")
