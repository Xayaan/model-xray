"""Optional: label features with an LLM through any OpenAI-compatible chat endpoint
(a vLLM server on a rented GPU, or a hosted API). Labels land in features/<site>/labels.json and
appear in the browser after re-running `xray features`."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

PROMPT = """Below are text excerpts where one feature of a language model fires. The token(s) where it fires \
most strongly are marked like <<this>>. Reply with a short label (at most 8 words) for what the feature \
detects. Reply with the label only.

{examples}"""


def format_examples(feature: dict, max_examples: int = 8) -> str:
    lines = []
    for n, ex in enumerate(feature["examples"][:max_examples], 1):
        top = max(ex["acts"]) or 1.0
        text = "".join(f"<<{t}>>" if a >= 0.5 * top else t for t, a in zip(ex["tokens"], ex["acts"]))
        lines.append(f"{n}. {text.strip()}")
    return "\n".join(lines)


def ask(base_url: str, model: str, prompt: str, api_key: str | None) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0})
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body.encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip().strip('"')


def label_site(features_dir: str | Path, site: str, base_url: str, model: str, limit: int | None = None,
               min_density: float = 1e-6) -> Path:
    site_dir = Path(features_dir) / site
    index = json.loads((site_dir / "index.json").read_text())
    labels_path = site_dir / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    api_key = os.environ.get("XRAY_LLM_API_KEY")
    chunk_size = _chunk_size(site_dir)
    chunks: dict[int, dict] = {}
    todo = [r for r in index if r["density"] >= min_density and str(r["id"]) not in labels]
    for n, row in enumerate(todo[:limit] if limit else todo, 1):
        chunk_index = row["id"] // chunk_size
        if chunk_index not in chunks:
            chunks[chunk_index] = json.loads((site_dir / f"chunk_{chunk_index}.json").read_text())
        feature = chunks[chunk_index]["features"][str(row["id"])]
        if not feature["examples"]:
            continue
        labels[str(row["id"])] = ask(base_url, model, PROMPT.format(examples=format_examples(feature)), api_key)
        if n % 25 == 0:
            labels_path.write_text(json.dumps(labels, indent=1))
            print(f"labelled {n}/{len(todo)}")
    labels_path.write_text(json.dumps(labels, indent=1))
    return labels_path


def _chunk_size(site_dir: Path) -> int:
    manifest = json.loads((site_dir.parent / "manifest.json").read_text())
    return next(s["chunk_size"] for s in manifest["sites"] if s["site"] == site_dir.name)
