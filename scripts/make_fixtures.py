"""Write the parity fixtures for the Rust port: inputs and the Python model's outputs.

usage (from justin_rs/, with the Python repository as its sibling ../justin):
    uv run --project ../justin python scripts/make_fixtures.py
JUSTIN_ROOT points to the Python repository if it lives elsewhere.

Everything runs on the CPU in float32. Output goes to tests/fixtures/ (git-ignored):
  encoder.json + encoder.safetensors   Timestral, all four sizes, with and without FiLM and LoRA
  minilm.json + minilm.safetensors     token ids, token states and sentence embeddings
  features.json                        `prepare` (deviation and referent features) on crafted cases
  models.json                          Justin end to end: the production export with its location
                                       head, and two models with random weights for the other
                                       config flags (models/ holds their exports)
The production files come from `justin export` / `justin export-location` (runs/export/).
"""

import copy
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from justin.config import Config
from justin.export import export_checkpoint, export_location
from justin.features import prepare
from justin.location.dataset import LocationDataset
from justin.model.encoders import SERIES_ENCODERS, TextEncoder
from justin.model.location import LocationHead
from justin.model.lora import inject_lora
from justin.pretrained import Justin
from justin.training.calibration import probabilities as calibrated
from justin.training.run import build_model

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("JUSTIN_ROOT", HERE.parent / "justin")).resolve()
OUT = HERE / "tests" / "fixtures"
PROD = ROOT / "runs" / "export" / "justin-timestral5m-minilm-prod.safetensors"
PROD_LOCATION = ROOT / "runs" / "export" / "justin-timestral5m-minilm-prod-location.safetensors"
SIZES = ("timestral5m", "timestral28m", "timestral123m", "timestral209m")
torch.set_grad_enabled(False)


def floats(t) -> list:
    """A tensor as nested lists of Python floats (exact for float32), NaN as None."""
    values = torch.as_tensor(t).detach().float().cpu().tolist()

    def clean(v):
        if isinstance(v, list):
            return [clean(x) for x in v]
        return None if math.isnan(v) else v

    return clean(values)


def encoder_fixtures() -> None:
    tensors, cases = {}, []
    g = torch.Generator().manual_seed(0)
    for size in SIZES:
        encoder = SERIES_ENCODERS[size].from_pretrained().eval()
        d = encoder.hidden_size
        series = torch.randn(2, 3, 200, generator=g).cumsum(-1)
        series[0, 1] += 1e4  # a large offset (asinh path)
        series[1, 2] = 5.0  # a constant channel
        series[0, 0, 50:60] = float("nan")  # a hole
        series[1, :, 190:] = float("nan")  # a missing tail
        single = torch.randn(1, 1, 64, generator=g)
        film = [
            (torch.randn(2, d, generator=g) * 0.1, torch.randn(2, d, generator=g) * 0.1) for _ in range(3)
        ]
        variants = [("plain", series, None, False), ("film_lora", series, film, True), ("single", single, None, False)]
        for name, x, conditioning, lora in variants:
            model = copy.deepcopy(encoder)
            key = f"{size}.{name}"
            if lora:
                boundary = model.num_blocks - 2
                inject_lora(model.backbone, model.lora_targets(boundary), rank=4, alpha=8)
                for module_name, parameter in model.backbone.named_parameters():
                    if module_name.endswith((".a.weight", ".b.weight")):
                        parameter.copy_(torch.randn(parameter.shape, generator=g) * 0.05)
                        tensors[f"{key}.lora.{module_name}"] = parameter.clone()
            states, valid = model(x, None, film=conditioning)
            tensors[f"{key}.series"] = x.clone()
            tensors[f"{key}.states"] = states
            tensors[f"{key}.valid"] = valid.to(torch.uint8)
            if conditioning:
                for i, (scale, shift) in enumerate(conditioning):
                    tensors[f"{key}.film.{i}.scale"] = scale
                    tensors[f"{key}.film.{i}.shift"] = shift
            cases.append({"key": key, "size": size, "repo": model.repo, "revision": model.revision,
                          "film_layers": len(conditioning or []), "lora": lora, "lora_rank": 4, "lora_alpha": 8})
            print(f"encoder {key}: states {tuple(states.shape)}")
    save_file({k: v.contiguous() for k, v in tensors.items()}, OUT / "encoder.safetensors")
    (OUT / "encoder.json").write_text(json.dumps(cases, indent=1))


def minilm_fixtures() -> None:
    encoder = TextEncoder.from_pretrained().eval()
    texts = [
        "Is there an anomaly in this time window?\nThe series has one point every 4 s.",
        "cartservice cpu",
        "Which service is the root cause? Ünïcödé, numbers 12.5 and a 'quoted_metric'.",
        "x " * 300,  # truncated at 256 tokens
        "",
    ]
    hidden, tokens = encoder._hidden(texts)
    save_file(
        {"hidden": hidden.contiguous(), "embed": encoder(texts).contiguous(),
         "ids": tokens["input_ids"].to(torch.int64).contiguous(), "mask": tokens["attention_mask"].to(torch.int64).contiguous()},
        OUT / "minilm.safetensors",
    )  # fmt: skip
    (OUT / "minilm.json").write_text(json.dumps({"texts": texts, "repo": encoder.repo, "revision": encoder.revision}))
    print(f"minilm: {tuple(hidden.shape)}")


def features_fixtures() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 300)).cumsum(-1)
    spiky = base.copy()
    spiky[1, 200:] += 40  # a sustained shift in the last third
    spiky[2, 120:140] += 25  # a deviation that ends
    names = ["cartservice_cpu", "cartservice_mem", "redis_latency-50", "frontend_cpu"]
    grouped = ["time-series 1: partition:5", "time-series 1: partition:11", "time-series 2: lag", "time-series 2: rate"]
    window = "The time range is 2024-01-01 00:00:00 to 2024-01-01 05:00:00."
    cases = [
        dict(series=spiky, question="Which service is the root cause?",
             options=["cartservice", "redis", "frontend", "No service is faulty"], names=names, step=4.0, baseline=0.3),
        dict(series=spiky, question="Did the metric 'cartservice_mem' change before the metric 'redis_latency-50'?",
             options=None, names=names, step=None, baseline=[10, 90]),
        dict(series=spiky, question=f"When did time-series 1 change? {window}",
             options=["2024-01-01 03:20:00", "2024-01-01 01:00:00", "partition:5, partition:11", "12.5", "-3"],
             names=grouped, step=60.0, baseline=0.25),
        dict(series=spiky[:1], question="Is this abnormal?\nThe series has one point every 4 s.",
             options=None, names=None, step=4.0, baseline=None),
        dict(series=base[:, :2], question="Tiny?", options=["a", "b"], names=None, step=0.5, baseline=0.3),
        dict(series=spiky[:2] * 1e6, question="Large values", options=["x", "y"], names=names[:2], step=1e-5, baseline=0.3),
    ]
    out = []
    for case in cases:
        baseline = tuple(case["baseline"]) if isinstance(case["baseline"], list) else case["baseline"]
        p = prepare(case["series"].astype(np.float32), case["question"], case["options"], case["names"],
                    step=case["step"], baseline=baseline)  # fmt: skip
        out.append({
            "series": floats(torch.as_tensor(case["series"].astype(np.float32))), "question": case["question"],
            "options": case["options"], "names": case["names"], "step": case["step"], "baseline": case["baseline"],
            "expected": {
                "series": floats(p.series), "question": p.question, "channel_features": floats(p.channel_features),
                "referent_question": floats(p.referent_question), "referent_options": floats(p.referent_options),
            },
        })  # fmt: skip
    (OUT / "features.json").write_text(json.dumps(out))
    print(f"features: {len(out)} cases")


def reference(model: Justin, series, question, answers, names, step, baseline) -> dict:
    """The Python Justin's logits and calibrated probabilities for one question (any head)."""
    options = answers.get("options") if isinstance(answers, dict) else None
    head = {"yes_no": "noul", "score": "score", "classes": "choice"}.get(answers, "choice") if not options else "choice"
    tensor = torch.as_tensor(series, dtype=torch.float32)
    system, extra = model.system, {}
    from justin.features import uses_features

    if uses_features(system):
        if system.channel_projection is not None and names is None:
            raise ValueError("needs names")
        prepared = prepare(tensor.numpy(), question, options, names, step=step, baseline=baseline)
        tensor, question = prepared.series, prepared.question
        if system.feature_projection is not None:
            extra["channel_features"] = prepared.channel_features[None]
        if system.referent_question is not None:
            extra["referent_question"] = prepared.referent_question[None]
            extra["referent_options"] = prepared.referent_options[None]
    takes_names = system.channel_projection is not None or not uses_features(system)
    channel_names = [list(names)] if names is not None and takes_names else None
    logits = system(tensor[None], [question], options=options, head=head, channel_names=channel_names, **extra)
    probability = calibrated(logits, head, model.temperatures.get(head, 1.0))
    return {"logits": floats(logits.reshape(-1)), "probabilities": floats(probability.reshape(-1))}


def obs_cases(count: int) -> list[dict]:
    """Real test windows of the observability data, spread over the tasks."""
    rows = [json.loads(line) for line in open(ROOT / "runs" / "all" / "obs" / "qa.jsonl")]
    rows = [r for r in rows if r["split"] == "test"]
    random.Random(0).shuffle(rows)
    picked, seen = [], {}
    for row in rows:
        if seen.get(row["task"], 0) >= max(1, count // 8):
            continue
        seen[row["task"]] = seen.get(row["task"], 0) + 1
        picked.append(row)
        if len(picked) == count:
            break
    return picked


def model_fixtures() -> None:
    models = {"prod": {"export": str(PROD), "location": str(PROD_LOCATION)}}
    cases = []
    prod = Justin.from_pretrained(PROD, device="cpu", location=PROD_LOCATION)
    for row in obs_cases(24):
        series = [s["values"] for s in row["timeseries"]]
        names = [s["name"] for s in row["timeseries"]]
        step = float(row["timeseries"][0]["step"])
        queries = [{"question": row["question"], "answers": {"options": row["options"]}},
                   {"question": row["question"], "answers": "yes_no"},
                   {"question": row["question"], "answers": "score"}]  # fmt: skip
        expected = [reference(prod, series, q["question"], q["answers"], names, step, 0.3) for q in queries]
        cases.append({"model": "prod", "id": row["id"], "series": floats(torch.tensor(series)), "channel_names": names,
                      "step": step, "baseline": 0.3, "queries": queries, "expected": expected})  # fmt: skip
    data = LocationDataset(ROOT / "data" / "locations", split="test", context_length=32768)
    for index in random.Random(1).sample(range(len(data)), 30):
        sample = data[index]
        names = list(sample["channel_names"])
        result = prod.locate(sample["series"], sample["question"], names)
        cases.append({
            "model": "prod", "id": sample["id"], "series": floats(sample["series"]), "channel_names": names,
            "step": None, "baseline": 0.3, "queries": [{"question": sample["question"], "answers": None}],
            "expected": [{"logits": floats(result["logits"]), "location": result["location"],
                          "null_probability": result["null_probability"], "truth": sample["location"]}],
        })  # fmt: skip
    print(f"prod: {len(cases)} cases")

    # Models with random weights for the flags the production model does not use.
    directory = OUT / "models"
    directory.mkdir(exist_ok=True)
    coverage = {
        "wide": Config(series_encoder="timestral28m", option_attention=True, channel_head=True, channel_names=True,
                       channel_features=9, num_classes=4, series_lora_layers=4, hidden_dim=64, fusion_layers=1,
                       score_levels=3),
        "deep": Config(series_encoder="timestral5m", minilm_lora=True, series_depth=6, series_condition=3,
                       series_lora_layers=2, referents=True, channel_features=9, signal_skip=True, hidden_dim=32,
                       fusion_layers=2, lora_rank=4, lora_alpha=4),
    }  # fmt: skip
    rng = np.random.default_rng(2)
    for name, config in coverage.items():
        torch.manual_seed(3)
        system = build_model(config)
        for parameter_name, parameter in system.named_parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * (0.02 if ".a." in parameter_name or ".b." in parameter_name else 0.1))
        checkpoint = directory / f"{name}.pt"
        torch.save({"model": system.state_dict(), "config": asdict(config), "temperatures": {"choice": 1.3, "noul": 0.8, "score": 2.0}}, checkpoint)
        export = export_checkpoint(checkpoint, directory / f"{name}.safetensors")
        checkpoint.unlink()
        models[name] = {"export": str(export), "location": None}
        model = Justin.from_pretrained(export, device="cpu")
        for i in range(3):
            channels = 3 + i
            series = rng.normal(size=(channels, 150 + 37 * i)).cumsum(-1).astype(np.float32)
            series[i % channels, 100:] += 30
            names = [f"svc{c % 2}_metric{c}" for c in range(channels)] if config.channel_names or config.referents else None
            queries = [{"question": "Which service is the root cause?", "answers": {"options": ["svc0", "svc1", "No service is faulty"]}},
                       {"question": "Is metric 'svc0_metric0' abnormal?", "answers": "yes_no"},
                       {"question": "How severe is it?", "answers": "score"}]  # fmt: skip
            if config.num_classes:
                queries.append({"question": "Which class?", "answers": "classes"})
            expected = [reference(model, series, q["question"], q["answers"], names, 2.0, 0.3) for q in queries]
            cases.append({"model": name, "id": f"{name}-{i}", "series": floats(torch.tensor(series)), "channel_names": names,
                          "step": 2.0, "baseline": 0.3, "queries": queries, "expected": expected})  # fmt: skip
        print(f"{name}: exported and 3 cases")

    # A pointer location head with random weights on the production model.
    head = LocationHead(192, 384, 64)
    torch.nn.init.normal_(head.null_key)
    torch.save({"state": head.state_dict(), "signal_dim": 192, "text_dim": 384, "hidden_dim": 64, "head": "pointer",
                "patch_size": 32}, directory / "pointer.pt")  # fmt: skip
    pointer = export_location(directory / "pointer.pt", PROD, directory / "pointer.safetensors")
    (directory / "pointer.pt").unlink()
    models["prod-pointer"] = {"export": str(PROD), "location": str(pointer)}
    with_pointer = Justin.from_pretrained(PROD, device="cpu", location=pointer)
    for index in random.Random(2).sample(range(len(data)), 5):
        sample = data[index]
        names = list(sample["channel_names"])
        result = with_pointer.locate(sample["series"], sample["question"], names)
        cases.append({
            "model": "prod-pointer", "id": sample["id"], "series": floats(sample["series"]), "channel_names": names,
            "step": None, "baseline": 0.3, "queries": [{"question": sample["question"], "answers": None}],
            "expected": [{"logits": floats(result["logits"]), "location": result["location"],
                          "null_probability": result["null_probability"]}],
        })  # fmt: skip
    (OUT / "models.json").write_text(json.dumps({"models": models, "cases": cases}))
    print(f"models: {len(cases)} cases in total")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    encoder_fixtures()
    minilm_fixtures()
    features_fixtures()
    model_fixtures()
