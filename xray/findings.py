"""The findings pass: for each behaviour, find the features that separate two prompt sets at every
trained site, then turn the strongest candidate up and measure whether the behaviour follows.

Experiments
  refusal           harmful requests (AdvBench) vs harmless ones (Alpaca); steer harmless requests
                    and count refusals
  unknown_entities  "Who is X?" for made-up people vs famous people; steer questions about famous
                    people and count "I don't know"-style answers

Only harmless prompts are ever steered, so no experiment tries to make the model comply with a
harmful request.
"""

from __future__ import annotations

import json
import random

from .config import RunConfig
from .contrast import contrast, generate, is_refusal, is_unknown, last_token_codes, load_prompts, save_json, steer
from .model import load_model
from .train import load_dictionaries

FAMOUS_PEOPLE = [
    "Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin", "Leonardo da Vinci",
    "William Shakespeare", "Mahatma Gandhi", "Nelson Mandela", "Abraham Lincoln", "Cleopatra",
    "Napoleon Bonaparte", "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "Pablo Picasso",
    "Vincent van Gogh", "Galileo Galilei", "Nikola Tesla", "Ada Lovelace", "Alan Turing",
    "Martin Luther King Jr.", "Winston Churchill", "Queen Victoria", "Julius Caesar", "Aristotle",
    "Plato", "Confucius", "Jane Austen", "Charles Dickens", "Frida Kahlo", "Michelangelo",
    "Johann Sebastian Bach", "Florence Nightingale", "Thomas Edison", "Rosa Parks", "George Washington",
    "Joan of Arc", "Genghis Khan", "Alexander the Great", "Leo Tolstoy", "Mark Twain",
    "Sigmund Freud", "Karl Marx", "Neil Armstrong", "Amelia Earhart", "Stephen Hawking",
    "Marco Polo", "Christopher Columbus", "Elvis Presley", "Muhammad Ali", "Walt Disney",
    "Charlie Chaplin", "Frederick Douglass", "Harriet Tubman", "Johannes Gutenberg", "Rembrandt",
    "Dante Alighieri", "Homer", "Socrates", "Archimedes", "Euclid",
]

_FIRST_A = ["Ka", "Lo", "Mi", "Ve", "Dra", "Sel", "Tor", "Bri", "Quen", "Zo", "Fa", "Ri", "Ul", "Ne", "Ha", "Jor", "Pel", "Ys"]
_FIRST_B = ["rin", "vek", "lla", "dor", "mis", "than", "ra", "quil", "nesh", "bo", "lio", "ven", "mo", "sa"]
_LAST_A = ["Thal", "Brem", "Okun", "Varg", "Pell", "Quist", "Mard", "Sorr", "Kelv", "Ostr", "Fenn", "Gral", "Hask", "Lunn"]
_LAST_B = ["owitz", "ley", "ari", "stad", "enko", "more", "ford", "ani", "ers", "ard", "ini", "vik"]


def made_up_people(n: int, seed: int = 0) -> list[str]:
    """Plausible-sounding names stitched from syllables; vanishingly unlikely to name anyone famous."""
    rng = random.Random(seed)
    names: set[str] = set()
    while len(names) < n:
        names.add(f"{rng.choice(_FIRST_A)}{rng.choice(_FIRST_B)} {rng.choice(_LAST_A)}{rng.choice(_LAST_B)}")
    return sorted(names)


def who_is(names: list[str]) -> list[str]:
    return [f"Who is {name}? Answer in one or two sentences." for name in names]


def _candidate(result: dict) -> dict:
    """The feature to steer: fires on most of set A and rarely on set B."""
    rows = result["a_over_b"]
    clean = [r for r in rows if r["freq_a"] >= 0.25 and r["freq_a"] - r["freq_b"] >= 0.2]
    return max(clean or rows[:1], key=lambda r: (r["freq_a"] - r["freq_b"], r["diff"]))


def run_findings(cfg: RunConfig, harmful: str = "hf:walledai/AdvBench", harmless: str = "hf:tatsu-lab/alpaca:instruction",
                 n: int = 200, steer_n: int = 30, scales: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
                 max_new_tokens: int = 48) -> dict:
    model, tokenizer = load_model(cfg.model)
    loaded = load_dictionaries(cfg.run_dir, model)
    transcoder = cfg.dictionary.kind == "transcoder"
    features_dir = cfg.run_dir / "features"
    max_act = {
        name: [row["max_act"] for row in json.loads((features_dir / name / "index.json").read_text())]
        for name in loaded
    }

    harmless_all = load_prompts(harmless, n + steer_n)
    famous = who_is(FAMOUS_PEOPLE)
    experiments = {
        "refusal": {
            "a": ("harmful requests (AdvBench)", load_prompts(harmful, n)),
            "b": ("harmless requests (Alpaca)", harmless_all[:n]),
            "steer": ("held-out harmless requests", harmless_all[n : n + steer_n]),
            "judge": is_refusal,
            "measures": "refusal rate",
        },
        "unknown_entities": {
            "a": ("questions about made-up people", who_is(made_up_people(40))),
            "b": ("questions about famous people", famous[:40]),
            "steer": ("held-out questions about famous people", famous[40:]),
            "judge": is_unknown,
            "measures": "rate of 'I don't know this person' answers",
        },
    }

    report: dict = {"model": cfg.model.name, "run": cfg.name, "experiments": {}}
    for exp_name, exp in experiments.items():
        (label_a, prompts_a), (label_b, prompts_b) = exp["a"], exp["b"]
        steer_label, steer_prompts = exp["steer"]
        print(f"== {exp_name}: {len(prompts_a)} {label_a} vs {len(prompts_b)} {label_b}")
        codes_a = last_token_codes(model, tokenizer, loaded, prompts_a, transcoder)
        codes_b = last_token_codes(model, tokenizer, loaded, prompts_b, transcoder)

        texts = generate(model, tokenizer, steer_prompts, max_new_tokens)
        baseline = [{"prompt": p, "completion": t, "hit": exp["judge"](t)} for p, t in zip(steer_prompts, texts)]
        base_rate = sum(r["hit"] for r in baseline) / len(baseline)
        print(f"   baseline {exp['measures']} on {steer_label}: {base_rate:.0%}")

        sites = {}
        for name, (site, d) in loaded.items():
            result = contrast(codes_a[name], codes_b[name], top=10)
            cand = _candidate(result)
            runs = {}
            for scale in scales:
                key = str(float(scale))
                if scale == 0:
                    runs[key] = {"rate": base_rate, "absolute_scale": 0.0, "samples": baseline[:3]}
                    continue
                absolute = scale * max(max_act[name][cand["feature"]], 1e-6)
                out = steer(model, tokenizer, site, d, cand["feature"], absolute, steer_prompts, exp["judge"], max_new_tokens)
                runs[key] = {"rate": sum(r["hit"] for r in out) / len(out), "absolute_scale": absolute, "samples": out[:3]}
            rates = " ".join(f"x{s:g}:{runs[str(float(s))]['rate']:.0%}" for s in scales)
            print(f"   {name}: feature #{cand['feature']} fires on {cand['freq_a']:.0%} of A vs {cand['freq_b']:.0%} of B | steered {rates}")
            sites[name] = {"contrast": result, "candidate": cand, "steering": runs}

        report["experiments"][exp_name] = {
            "a": label_a, "b": label_b, "steered_on": steer_label, "measures": exp["measures"],
            "n_a": len(prompts_a), "n_b": len(prompts_b), "n_steer": len(steer_prompts),
            "scales_in_units_of_max_activation": list(scales), "baseline_rate": base_rate, "sites": sites,
        }

    save_json(report, cfg.run_dir / "findings" / "findings.json")
    print(f"saved {cfg.run_dir / 'findings' / 'findings.json'}")
    return report
