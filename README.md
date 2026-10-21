# justin (Rust)

Justin on [candle](https://github.com/huggingface/candle): a multichannel time series and a list of
queries in, answers out. A port of the Python package `justin` (expected next to this directory as
`../justin`), checked against it by parity tests (the fixture script lives in the Python repository,
`scripts/make_rust_fixtures.py`).

- **Series encoder:** Timestral (5M, 28M, 123M or 209M), with question conditioning (FiLM) and
  LoRA adapters (merged into the weights at load time).
- **Text encoder:** MiniLM (`sentence-transformers/all-MiniLM-L6-v2`), optionally with LoRA.
- **Heads:** cross-attention fusion with the choice, yes/no and score heads, plus everything the
  config can switch on (signal skip, channel names, channel features, referents, option
  attention, channel head, fixed classes).
- **Feature code:** the deviation and referent features the observability models read, computed
  from the raw series exactly as `justin.features.prepare` does.
- **Location heads:** the pointer and the attention head, which point at a (channel, patch) of the
  series or say "not present".

## Use

```rust
use justin::{Input, Justin, LoadOptions, Output, Query, Source};

let model = Justin::from_pretrained(
    Source::hub("owner/justin-timestral-minilm", "justin.safetensors"), // or Source::local(path)
    LoadOptions {
        location: Some(Source::local("justin-location.safetensors")),
        ..Default::default() // CPU, backbones from their pinned Hub revisions
    },
)?;
let outputs = model.forward(&Input {
    series: vec![cpu, memory], // [channels][time], raw values
    channel_names: Some(vec!["cartservice_cpu".into(), "cartservice_mem".into()]),
    step: Some(4.0), // seconds between points
    queries: vec![
        Query::yes_no("Is there an anomaly in this time window?"),
        Query::choice("Which service is the root cause?", &["cartservice", "redis", "No service is faulty"]),
        Query::locate("Locate the onset of the spike in cartservice_cpu in this window."),
    ],
    ..Default::default()
})?;
// One Output per query: Choice { probabilities, answer, .. }, YesNo { probability, answer, .. },
// Score { .. } or Location { location: Option<Location { channel, patch, start, end, .. }>, .. }.
```

A `Query` is either a question with its possible answers (`Answers::Options`, `YesNo`, `Score`, or
the model's fixed `Classes`) or a location question. All queries of an `Input` share its series.
Probabilities apply the calibration temperatures stored in the export.

## Files

- **Model:** written by `uv run python -m justin export <checkpoint.pt> <out.safetensors>`. It
  holds the trained tensors (fusion, heads, FiLM, LoRA), the model config, the calibration
  temperatures and the Hub revisions of the frozen backbones.
- **Location head:** written by `uv run python -m justin export-location <head.pt>
  <model.safetensors> <out.safetensors>`. It records a fingerprint of the model export it was
  trained on, and loading it next to any other model fails.
- **Backbones:** by default from the Hub at the revisions the export records. Files already in
  the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`) are read
  without network access. `Backbones::Local { series, text }` reads them from directories instead:
  `config.json` and `model.safetensors` for both, plus `tokenizer.json` for MiniLM.

## Devices

- **CPU:** the default.
- **Metal:** `--features metal`, then pass `Device::new_metal(0)?` in `LoadOptions::device`.
- **CUDA:** `--features cuda`. It needs the CUDA toolkit to build and has not been tested.

Timestral's float64 scaler and the feature code always run on the CPU in plain Rust, as the
Python code does on MPS.

## Parity with Python

```
cd ../justin
uv run python -m justin export runs/sweep-lora/prod-film-minilmfrozen.pt runs/export/justin-timestral5m-minilm-prod.safetensors
uv run python -m justin export-location runs/sweep-lora/prod-location-attn.pt \
    runs/export/justin-timestral5m-minilm-prod.safetensors runs/export/justin-timestral5m-minilm-prod-location.safetensors
uv run python scripts/make_rust_fixtures.py   # Python outputs on CPU float32 -> ../justin_rs/tests/fixtures/
cd ../justin_rs
cargo test --release -- --nocapture
```

The tests print their worst differences. Each one asserts a bound:

| What | Bound |
|---|---|
| Timestral states (all four sizes, with and without FiLM and LoRA, NaN holes, padding) | 1e-4 × max \|state\| |
| MiniLM token ids | identical |
| MiniLM token states | 1e-5 |
| MiniLM embeddings | 1e-5 |
| Features | 1e-5 |
| Question-head logits | 1e-4 |
| Question-head probabilities | 1e-5 |
| Location logits | 1e-4, relative to the largest logit |
| Decoded answers and locations | identical |

Without fixtures the parity tests skip.

## Licences

The code is Apache-2.0. Timestral's code is Apache-2.0 and its weights are Modified MIT (Mistral
AI). MiniLM is Apache-2.0.
