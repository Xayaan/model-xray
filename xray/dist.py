"""Multi-GPU runs on one machine (2, 4, 8 or more GPUs).

Launched with torchrun (`xray ... --gpus N` does it for you), each process keeps a copy of the
model on its own GPU and reads its own share of the data. The dictionaries are divided between the
processes. After every forward pass the processes swap activations with one all_to_all, so each
process trains only its own dictionaries, on the tokens from every GPU. Throughput and the memory
available for dictionaries both grow with the number of GPUs.

Evaluation and feature analysis split the same way: each process handles its own dictionaries and
the main process merges the results.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world_info() -> tuple[int, int, int]:
    """(rank, world size, local rank) from the torchrun environment; (0, 1, 0) for a plain run."""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))


def setup() -> tuple[int, int, torch.device | None]:
    """Join the process group if launched with torchrun. Returns (rank, world, device) where
    device is this process's GPU, or None for a single-process run."""
    rank, world, local_rank = world_info()
    if world == 1:
        return 0, 1, None
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if use_cuda else "gloo")
    return rank, world, torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")


def teardown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def any_true(flag: bool, device) -> bool:
    """True on every process if `flag` is true on any of them (keeps the processes in step)."""
    if not dist.is_initialized():
        return flag
    t = torch.tensor([int(flag)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def owned(items: list, rank: int, world: int) -> list:
    """The share of `items` this process is responsible for (round robin, so layers spread evenly)."""
    return items[rank::world]


def exchange(rows: dict[str, tuple[torch.Tensor, torch.Tensor]], names: list[str], paired: bool,
             rank: int, world: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """rows holds this process's (input, target) activations for every site, in `names` order.
    Returns, for each site this process owns, the rows from every process stacked together.

    Every process holds the same number of rows per site, so message sizes are known everywhere.
    Messages are padded to one common size because some backends (gloo) need equal sizes, which
    matters when the sites don't divide evenly across GPUs."""
    if world == 1:
        return rows
    mine = owned(names, rank, world)
    sample = next(iter(rows.values()))[0]

    def flat(site_names: list[str]) -> torch.Tensor:
        parts = []
        for name in site_names:
            x, y = rows[name]
            parts.append(x.reshape(-1))
            if paired:
                parts.append(y.reshape(-1))
        return torch.cat(parts) if parts else sample.new_empty(0)

    send = [flat(owned(names, dst, world)) for dst in range(world)]
    sizes = [t.numel() for t in send]
    size = max(sizes)
    send = [torch.cat([t, t.new_zeros(size - t.numel())]) if t.numel() < size else t for t in send]
    recv = [sample.new_empty(size) for _ in range(world)]
    dist.all_to_all(recv, send)
    recv = [buffer[: sizes[rank]] for buffer in recv]

    pieces: dict[str, tuple[list, list]] = {name: ([], []) for name in mine}
    for buffer in recv:
        offset = 0
        for name in mine:
            x, y = rows[name]
            pieces[name][0].append(buffer[offset : offset + x.numel()].view_as(x))
            offset += x.numel()
            if paired:
                pieces[name][1].append(buffer[offset : offset + y.numel()].view_as(y))
                offset += y.numel()
    out = {}
    for name in mine:
        x = torch.cat(pieces[name][0])
        out[name] = (x, torch.cat(pieces[name][1]) if paired else x)
    return out
