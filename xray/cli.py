"""Command line: xray {info,train,eval,features,browse,run,contrast,steer,findings,label}.

train, eval, features and run take --gpus N to use N GPUs on one machine (see dist.py)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import RunConfig


def relaunch_on_gpus(gpus: int | None, argv: list[str]) -> bool:
    """Re-run this command under torchrun with one process per GPU. Returns True if it did."""
    if not gpus or gpus < 2 or "RANK" in os.environ:
        return False
    import socket
    import subprocess

    with socket.socket() as s:  # an explicit local rendezvous works on every OS (--standalone can hang on macOS)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    rest, skip = [], False
    for arg in argv:  # drop --gpus N / --gpus=N
        if skip:
            skip = False
        elif arg == "--gpus":
            skip = True
        elif not arg.startswith("--gpus="):
            rest.append(arg)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={gpus}",
           "--master_addr=127.0.0.1", f"--master_port={port}", "-m", "xray.cli", *rest]
    print("launching:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


def cmd_info(args) -> None:
    from .model import build_meta, describe

    model = build_meta(args.model, trust_remote_code=args.trust_remote_code)
    blocks = describe(model)
    kinds = {}
    for b in blocks:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    print(f"{args.model}: {len(blocks)} blocks {kinds}")
    print("mlp sites:", [b["layer"] for b in blocks if b["mlp"]])
    for b in blocks:
        print(f"  L{b['layer']:<3} {b['kind']:<12} mlp={b['mlp']} mixer={b['mixer']}")


def cmd_train(args) -> None:
    from .train import train

    cfg = RunConfig.load(args.config)
    if args.tokens:
        cfg.train.total_tokens = args.tokens
    train(cfg, sites_override=args.sites)


def cmd_eval(args) -> None:
    from .evaluate import evaluate

    evaluate(RunConfig.load(args.config), n_tokens=args.tokens)


def cmd_features(args) -> None:
    from .features import analyze

    analyze(RunConfig.load(args.config), n_tokens=args.tokens)


def cmd_browse(args) -> None:
    from .browser import build_browser

    cfg = RunConfig.load(args.config)
    if args.sample:
        out = build_browser(cfg.run_dir / "features", embed=True, sample=args.sample, out_name="preview.html")
    else:
        out = build_browser(cfg.run_dir / "features", embed=args.embed)
    print(out)


def cmd_run(args) -> None:
    from . import dist
    from .browser import build_browser
    from .evaluate import evaluate
    from .features import analyze
    from .train import train

    cfg = RunConfig.load(args.config)
    train(cfg)
    evaluate(cfg)
    out = analyze(cfg)
    if dist.is_main():
        print(build_browser(out, embed=args.embed))


def _load_site(cfg: RunConfig, site_name: str):
    from .model import load_model
    from .train import load_dictionaries

    model, tokenizer = load_model(cfg.model)
    loaded = load_dictionaries(cfg.run_dir, model)
    if site_name not in loaded:
        raise SystemExit(f"no dictionary for {site_name}; trained: {', '.join(loaded)}")
    site, d = loaded[site_name]
    return model, tokenizer, site, d


def cmd_contrast(args) -> None:
    from .contrast import contrast, last_token_codes, load_prompts, save_json

    cfg = RunConfig.load(args.config)
    model, tokenizer, site, d = _load_site(cfg, args.site)
    transcoder = cfg.dictionary.kind == "transcoder"
    prompts_a, prompts_b = load_prompts(args.a, args.limit), load_prompts(args.b, args.limit)
    one = {site.name: (site, d)}
    codes_a = last_token_codes(model, tokenizer, one, prompts_a, transcoder, args.system)[site.name]
    codes_b = last_token_codes(model, tokenizer, one, prompts_b, transcoder, args.system)[site.name]
    result = {"site": site.name, "a": args.a, "b": args.b, "n_a": len(prompts_a), "n_b": len(prompts_b),
              **contrast(codes_a, codes_b, args.top)}
    path = cfg.run_dir / "findings" / f"contrast_{site.name}.json"
    save_json(result, path)
    print(f"features firing more on A ({args.a}) than B ({args.b}) at {site.name}:")
    for r in result["a_over_b"][:10]:
        print(f"  #{r['feature']:<7} mean A {r['mean_a']:.3f} vs B {r['mean_b']:.3f}  (fires on {r['freq_a']:.0%} vs {r['freq_b']:.0%})")
    print(f"saved {path}")


def cmd_steer(args) -> None:
    from .contrast import is_refusal, is_unknown, load_prompts, save_json, steer

    cfg = RunConfig.load(args.config)
    model, tokenizer, site, d = _load_site(cfg, args.site)
    prompts = load_prompts(args.prompts, args.limit)
    judge = {"refusal": is_refusal, "unknown": is_unknown}[args.measure]
    unit = 1.0
    if args.relative:
        index = json.loads((cfg.run_dir / "features" / site.name / "index.json").read_text())
        unit = index[args.feature]["max_act"]
    runs = {}
    for scale in args.scales:
        results = steer(model, tokenizer, site, d, args.feature, scale * unit, prompts, judge, args.max_new_tokens, args.system)
        rate = sum(r["hit"] for r in results) / len(results)
        runs[str(scale)] = {"rate": rate, "absolute_scale": scale * unit, "samples": results}
        print(f"scale {scale:>6}: {args.measure} rate {rate:.0%}  e.g. {results[0]['completion'][:90]!r}")
    path = cfg.run_dir / "findings" / f"steer_{site.name}_{args.feature}.json"
    save_json({"site": site.name, "feature": args.feature, "prompts": args.prompts, "measure": args.measure, "runs": runs}, path)
    print(f"saved {path}")


def cmd_findings(args) -> None:
    from .findings import run_findings

    run_findings(RunConfig.load(args.config), n=args.n, steer_n=args.steer_n, scales=tuple(args.scales))


def cmd_label(args) -> None:
    from .autointerp import label_site

    cfg = RunConfig.load(args.config)
    print(label_site(cfg.run_dir / "features", args.site, args.base_url, args.model, args.limit))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="xray", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="show a model's blocks and which sites exist")
    p.add_argument("model")
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(fn=cmd_info)

    for name, fn, help_text in (
        ("train", cmd_train, "train dictionaries"),
        ("eval", cmd_eval, "measure reconstruction and spliced loss"),
        ("features", cmd_features, "collect top examples and logit lens for every feature"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config")
        p.add_argument("--tokens", type=int, help="override the token budget")
        p.add_argument("--gpus", type=int, help="use this many GPUs on this machine (2, 4, 8, ...)")
        if name == "train":
            p.add_argument("--sites", nargs="+", help="override train.sites, e.g. mlp:0-5")
        p.set_defaults(fn=fn)

    for name, fn, help_text in (("browse", cmd_browse, "write the HTML browser"), ("run", cmd_run, "train, eval, features, browse")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config")
        p.add_argument("--embed", action="store_true", help="inline the data so the page opens from disk")
        if name == "browse":
            p.add_argument("--sample", type=int, help="write preview.html with N random live features per site")
        else:
            p.add_argument("--gpus", type=int, help="use this many GPUs on this machine (2, 4, 8, ...)")
        p.set_defaults(fn=fn)

    p = sub.add_parser("contrast", help="features that separate prompt set A from set B")
    p.add_argument("config")
    p.add_argument("--site", required=True)
    p.add_argument("--a", required=True, help="file or hf:<repo>[:<field>][@split], e.g. hf:walledai/AdvBench")
    p.add_argument("--b", required=True, help="e.g. hf:tatsu-lab/alpaca:instruction")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--system", help="system prompt, e.g. /no_think for reasoning models")
    p.set_defaults(fn=cmd_contrast)

    p = sub.add_parser("steer", help="generate with a feature turned up and measure the behaviour")
    p.add_argument("config")
    p.add_argument("--site", required=True)
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    p.add_argument("--relative", action="store_true", help="scales are multiples of the feature's max activation")
    p.add_argument("--measure", choices=["refusal", "unknown"], default="refusal")
    p.add_argument("--prompts", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--system")
    p.set_defaults(fn=cmd_steer)

    p = sub.add_parser("findings", help="refusal and unknown-entity experiments across every trained site")
    p.add_argument("config")
    p.add_argument("--n", type=int, default=200, help="prompts per contrast set")
    p.add_argument("--steer-n", type=int, default=30, help="held-out prompts to steer")
    p.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    p.set_defaults(fn=cmd_findings)

    p = sub.add_parser("label", help="label features with an OpenAI-compatible LLM endpoint")
    p.add_argument("config")
    p.add_argument("--site", required=True)
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_label)

    args = parser.parse_args(argv)
    relaunch_on_gpus(getattr(args, "gpus", None), sys.argv[1:] if argv is None else argv)
    try:
        args.fn(args)
    finally:
        from . import dist

        dist.teardown()


if __name__ == "__main__":
    main()
