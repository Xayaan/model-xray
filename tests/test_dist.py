"""Multi-GPU code paths, run as two CPU processes over gloo (the same collectives NCCL runs on GPUs)."""

import json
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from tests.test_pipeline import TEXT


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _join(rank: int, world: int, port: int) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))


def _exchange_worker(rank, world, port, out_dir):
    from xray import dist

    _join(rank, world, port)
    dist.setup()
    names = ["A", "B", "C"]
    rows = {n: (torch.full((3, 2), 100.0 * rank + i), torch.full((3, 4), -(100.0 * rank + i))) for i, n in enumerate(names)}
    got = dist.exchange(rows, names, paired=True, rank=rank, world=world)
    summary = {n: {"x": x[:, 0].tolist(), "y_matches": bool(torch.equal(y[:, 0], -x[:, 0])), "shape": list(x.shape)}
               for n, (x, y) in got.items()}
    (out_dir / f"rank{rank}.json").write_text(json.dumps(summary))
    dist.teardown()


def test_exchange_sends_each_site_to_its_owner(tmp_path):
    mp.spawn(_exchange_worker, args=(2, _free_port(), tmp_path), nprocs=2)
    r0 = json.loads((tmp_path / "rank0.json").read_text())
    r1 = json.loads((tmp_path / "rank1.json").read_text())
    assert set(r0) == {"A", "C"} and set(r1) == {"B"}  # round-robin ownership
    assert r0["A"]["x"] == [0.0] * 3 + [100.0] * 3  # rows from rank 0, then rank 1
    assert r1["B"]["x"] == [1.0] * 3 + [101.0] * 3
    assert all(v["y_matches"] and v["shape"] == [6, 2] for v in (*r0.values(), *r1.values()))


def _pipeline_worker(rank, world, port, cfg_dict):
    from xray.config import RunConfig
    from xray.evaluate import evaluate
    from xray.features import analyze
    from xray.train import train
    from xray import dist

    _join(rank, world, port)
    cfg = RunConfig.from_dict(cfg_dict)
    train(cfg)
    evaluate(cfg)
    analyze(cfg)
    dist.teardown()


@pytest.mark.network
def test_two_gpu_style_pipeline(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT)
    cfg = {
        "name": "dist",
        "out_root": str(tmp_path / "runs"),
        "model": {"name": "HuggingFaceTB/SmolLM2-135M", "dtype": "float32", "device": "cpu",
                  "random_init": {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3,
                                  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8}},
        "data": {"sources": [{"file": str(corpus)}], "ctx_len": 32, "batch_size": 4, "shuffle_buffer": 0},
        "dictionary": {"kind": "transcoder", "expansion": 4, "k": 4, "dead_after_tokens": 2000},
        "train": {"sites": ["mlp:all"], "total_tokens": 4000, "train_batch": 64, "warmup_steps": 5, "log_every": 20},
        "eval": {"n_tokens": 300, "batch_size": 2},
        "analysis": {"n_tokens": 500, "top_examples": 3, "window_left": 4, "window_right": 2, "chunk_size": 50},
    }
    mp.spawn(_pipeline_worker, args=(2, _free_port(), cfg), nprocs=2)

    run = tmp_path / "runs" / "dist"
    sites = {"L0.mlp", "L1.mlp", "L2.mlp"}  # 3 sites over 2 processes: one owns two, one owns one
    assert {p.name for p in (run / "dictionaries").iterdir()} == sites
    for name in sites:
        assert json.loads((run / "dictionaries" / name / "cfg.json").read_text())["tokens"] >= 4000
    assert set(json.loads((run / "eval.json").read_text())["sites"]) == sites
    manifest = json.loads((run / "features" / "manifest.json").read_text())
    assert [s["site"] for s in manifest["sites"]] == ["L0.mlp", "L1.mlp", "L2.mlp"]
    assert not list(run.glob("eval.rank*.json")) and not list((run / "features").glob("manifest.rank*.json"))
    assert (run / "train_log.rank0.jsonl").exists() and (run / "train_log.rank1.jsonl").exists()
