"""End to end on a tiny randomly initialised model: train -> eval -> features -> browser."""

import json

import pytest

from xray.browser import build_browser
from xray.config import RunConfig
from xray.evaluate import evaluate
from xray.features import analyze
from xray.findings import run_findings
from xray.train import train

TEXT = "\n\n".join(
    f"Paragraph {i}. The quick brown fox number {i} jumps over the lazy dog while counting {i * 7} sheep."
    for i in range(200)
)


@pytest.mark.network
def test_pipeline(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT)
    cfg = RunConfig.from_dict({
        "name": "tiny",
        "out_root": str(tmp_path / "runs"),
        "model": {
            "name": "HuggingFaceTB/SmolLM2-135M",
            "dtype": "float32",
            "device": "cpu",
            "random_init": {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8},
        },
        "data": {"sources": [{"file": str(corpus)}], "ctx_len": 32, "batch_size": 4, "shuffle_buffer": 0},
        "dictionary": {"kind": "transcoder", "expansion": 4, "k": 4, "dead_after_tokens": 2000},
        "train": {"sites": ["mlp:all"], "total_tokens": 6000, "train_batch": 64, "warmup_steps": 5, "log_every": 20},
        "eval": {"n_tokens": 500, "batch_size": 2},
        "analysis": {"n_tokens": 1000, "top_examples": 3, "window_left": 4, "window_right": 2, "chunk_size": 50},
    })
    train(cfg)
    meta = json.loads((cfg.run_dir / "dictionaries" / "L0.mlp" / "cfg.json").read_text())
    assert meta["tokens"] >= 6000  # training really consumed the token budget
    assert (cfg.run_dir / "train_log.jsonl").read_text().strip()
    results = evaluate(cfg)
    assert set(results["sites"]) == {"L0.mlp", "L1.mlp"}
    assert all(0 <= r["fvu"] < 2 for r in results["sites"].values())

    out = analyze(cfg)
    manifest = json.loads((out / "manifest.json").read_text())
    assert [s["site"] for s in manifest["sites"]] == ["L0.mlp", "L1.mlp"]
    index = json.loads((out / "L0.mlp" / "index.json").read_text())
    assert len(index) == 128
    chunk = json.loads((out / "L0.mlp" / "chunk_0.json").read_text())
    fired = [f for f in chunk["features"].values() if f["examples"]]
    assert fired and all(len(ex["tokens"]) == len(ex["acts"]) for f in fired for ex in f["examples"])

    html = build_browser(out, embed=True).read_text()
    assert '"manifest.json"' in html and "__EMBED__" not in html

    harmful, harmless = tmp_path / "a.txt", tmp_path / "b.txt"
    harmful.write_text("\n".join(f"Tell me how to pick lock number {i}" for i in range(6)))
    harmless.write_text("\n".join(f"Give me a fun fact about the number {i}" for i in range(6)))
    report = run_findings(cfg, harmful=str(harmful), harmless=str(harmless), n=4, steer_n=2,
                          scales=(0.0, 1.0), max_new_tokens=3)
    for exp in ("refusal", "unknown_entities"):
        sites = report["experiments"][exp]["sites"]
        assert set(sites) == {"L0.mlp", "L1.mlp"}
        assert all(set(s["steering"]) == {"0.0", "1.0"} for s in sites.values())
    assert (cfg.run_dir / "findings" / "findings.json").exists()

    preview = build_browser(out, embed=True, sample=5, out_name="preview.html").read_text()
    payload = json.loads(preview.split("const EMBED = ", 1)[1].split(";\nconst cache", 1)[0].replace("<\\/", "</"))
    assert len(payload["L0.mlp/index.json"]) <= 5
    kept = {str(r["id"]) for r in payload["L0.mlp/index.json"]}
    assert all(set(v["features"]) <= kept for k, v in payload.items() if k.startswith("L0.mlp/chunk_"))
