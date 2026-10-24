//! Parity with the Python source: the fixtures (`scripts/make_rust_fixtures.py` in the Python
//! repository) hold inputs and the Python model's outputs on CPU float32; these tests run the same
//! inputs through the Rust port.
//! Every test prints its worst difference. Missing fixtures skip a test with a message.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use justin::features::{Baseline, prepare};
use justin::nn::merge_lora;
use justin::source::hub_file;
use justin::text::TextEncoder;
use justin::timestral::{Timestral, TimestralConfig, load_backbone_weights};
use justin::weights::Weights;
use justin::{Answers, Input, Justin, LoadOptions, Output, Query, Source};
use serde_json::Value;

fn fixtures() -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    if path.join("models.json").is_file() {
        Some(path)
    } else {
        eprintln!(
            "skipped: no fixtures, run `uv run python scripts/make_rust_fixtures.py` in ../justin"
        );
        None
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(value: &Value) -> Vec<f32> {
    match value {
        Value::Array(items) => items.iter().flat_map(floats).collect(),
        Value::Null => vec![f32::NAN],
        Value::Number(n) => vec![n.as_f64().unwrap() as f32],
        other => panic!("not a number: {other}"),
    }
}

fn matrix(value: &Value) -> Vec<Vec<f32>> {
    value.as_array().unwrap().iter().map(floats).collect()
}

fn strings(value: &Value) -> Option<Vec<String>> {
    value.as_array().map(|items| {
        items
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect()
    })
}

/// Largest absolute difference (NaN must match NaN).
fn max_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "length");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| {
            if e.is_nan() {
                assert!(a.is_nan(), "expected NaN, got {a}");
                0.0
            } else {
                (a - e).abs()
            }
        })
        .fold(0.0, f32::max)
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn timestral_encoder_matches_python_for_all_sizes() {
    let Some(dir) = fixtures() else { return };
    let cases = read_json(&dir.join("encoder.json"));
    let tensors =
        candle_core::safetensors::load(dir.join("encoder.safetensors"), &Device::Cpu).unwrap();
    for case in cases.as_array().unwrap() {
        let key = case["key"].as_str().unwrap();
        let (repo, revision) = (
            case["repo"].as_str().unwrap(),
            case["revision"].as_str().unwrap(),
        );
        let config =
            TimestralConfig::from_file(&hub_file(repo, revision, "config.json", None).unwrap())
                .unwrap();
        let mut weights = load_backbone_weights(
            &hub_file(repo, revision, "model.safetensors", None).unwrap(),
            &Device::Cpu,
        )
        .unwrap();
        if case["lora"].as_bool().unwrap() {
            let prefix = format!("{key}.lora.");
            let adapters: HashMap<String, Tensor> = tensors
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let adapters = Weights::new(adapters, &Device::Cpu);
            let rank = case["lora_rank"].as_u64().unwrap() as usize;
            let alpha = case["lora_alpha"].as_f64().unwrap();
            assert!(merge_lora(&mut weights, &adapters, &prefix, alpha, rank).unwrap() > 0);
        }
        let encoder = Timestral::load(config, &weights, None).unwrap();
        let series = &tensors[&format!("{key}.series")];
        let (b, c, t) = series.dims3().unwrap();
        let values = flat(series);
        let batch: Vec<Vec<Vec<f32>>> = (0..b)
            .map(|i| {
                (0..c)
                    .map(|j| values[(i * c + j) * t..(i * c + j + 1) * t].to_vec())
                    .collect()
            })
            .collect();
        let film: Vec<(Tensor, Tensor)> = (0..case["film_layers"].as_u64().unwrap() as usize)
            .map(|i| {
                (
                    tensors[&format!("{key}.film.{i}.scale")].clone(),
                    tensors[&format!("{key}.film.{i}.shift")].clone(),
                )
            })
            .collect();
        let prepared = encoder.prepare(&batch, None).unwrap();
        let (states, valid) = encoder.forward(&prepared, &film).unwrap();
        let expected = flat(&tensors[&format!("{key}.states")]);
        let diff = max_diff(&flat(&states), &expected);
        let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(
            flat(&valid),
            flat(&tensors[&format!("{key}.valid")]),
            "{key} validity"
        );
        println!("encoder {key}: max |diff| {diff:.2e} (max |state| {scale:.1})");
        assert!(diff <= 1e-4 * scale.max(1.0), "{key}: {diff}");
    }
}

#[test]
fn minilm_matches_python() {
    let Some(dir) = fixtures() else { return };
    let meta = read_json(&dir.join("minilm.json"));
    let tensors =
        candle_core::safetensors::load(dir.join("minilm.safetensors"), &Device::Cpu).unwrap();
    let (repo, revision) = (
        meta["repo"].as_str().unwrap(),
        meta["revision"].as_str().unwrap(),
    );
    let weights = Weights::load(
        &hub_file(repo, revision, "model.safetensors", None).unwrap(),
        &Device::Cpu,
    )
    .unwrap();
    let encoder = TextEncoder::load(
        &hub_file(repo, revision, "config.json", None).unwrap(),
        &hub_file(repo, revision, "tokenizer.json", None).unwrap(),
        weights,
    )
    .unwrap();
    let texts = strings(&meta["texts"]).unwrap();
    let (ids, mask) = encoder.tokenize(&texts).unwrap();
    let to_i64 = |t: &Tensor| -> Vec<i64> {
        t.to_dtype(candle_core::DType::I64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    assert_eq!(to_i64(&ids), to_i64(&tensors["ids"]), "token ids");
    assert_eq!(to_i64(&mask), to_i64(&tensors["mask"]), "attention mask");
    // Padded positions are masked everywhere they are read, so only valid tokens are compared.
    let (hidden, _) = encoder.token_states(&texts).unwrap();
    let valid: Vec<i64> = to_i64(&tensors["mask"]);
    let width = hidden.dim(2).unwrap();
    let (actual, expected) = (flat(&hidden), flat(&tensors["hidden"]));
    let mut hidden_diff = 0f32;
    for (token, &keep) in valid.iter().enumerate() {
        if keep == 1 {
            let range = token * width..(token + 1) * width;
            hidden_diff = hidden_diff.max(max_diff(&actual[range.clone()], &expected[range]));
        }
    }
    let embed_diff = max_diff(
        &flat(&encoder.embed(&texts).unwrap()),
        &flat(&tensors["embed"]),
    );
    println!(
        "minilm: token states (valid tokens) max |diff| {hidden_diff:.2e}, embeddings {embed_diff:.2e}"
    );
    assert!(hidden_diff <= 1e-5 && embed_diff <= 1e-5);
}

fn baseline(value: &Value) -> Baseline {
    match value {
        Value::Null => Baseline::Given,
        Value::Array(range) => Baseline::Range(
            range[0].as_u64().unwrap() as usize,
            range[1].as_u64().unwrap() as usize,
        ),
        other => Baseline::Fraction(other.as_f64().unwrap()),
    }
}

#[test]
fn features_match_python() {
    let Some(dir) = fixtures() else { return };
    let mut worst = 0f32;
    for case in read_json(&dir.join("features.json")).as_array().unwrap() {
        let options = strings(&case["options"]).unwrap_or_default();
        let names = strings(&case["names"]);
        let p = prepare(
            &matrix(&case["series"]),
            case["question"].as_str().unwrap(),
            &options,
            names.as_deref(),
            case["step"].as_f64(),
            baseline(&case["baseline"]),
        )
        .unwrap();
        let expected = &case["expected"];
        assert_eq!(p.question, expected["question"].as_str().unwrap());
        let series: Vec<f32> = p.series.iter().flatten().copied().collect();
        let scale = floats(&expected["series"])
            .iter()
            .fold(1f32, |m, v| m.max(v.abs()));
        assert!(max_diff(&series, &floats(&expected["series"])) <= 1e-6 * scale);
        let features: Vec<f32> = p.channel_features.iter().flatten().copied().collect();
        let options: Vec<f32> = p.referent_options.iter().flatten().copied().collect();
        for (actual, wanted) in [
            (features, &expected["channel_features"]),
            (p.referent_question.clone(), &expected["referent_question"]),
            (options, &expected["referent_options"]),
        ] {
            worst = worst.max(max_diff(&actual, &floats(wanted)));
        }
    }
    println!("features: max |diff| {worst:.2e}");
    assert!(worst <= 1e-5, "{worst}");
}

fn query(value: &Value) -> Query {
    let question = value["question"].as_str().unwrap().to_string();
    match &value["answers"] {
        Value::Null => Query::Location { question },
        Value::String(kind) => Query::Question {
            question,
            answers: match kind.as_str() {
                "yes_no" => Answers::YesNo,
                "score" => Answers::Score,
                "classes" => Answers::Classes,
                other => panic!("unknown answers {other}"),
            },
        },
        Value::Object(map) => Query::Question {
            question,
            answers: Answers::Options(strings(&map["options"]).unwrap()),
        },
        other => panic!("bad answers {other}"),
    }
}

#[test]
fn justin_matches_python_end_to_end() {
    let Some(dir) = fixtures() else { return };
    let fixture = read_json(&dir.join("models.json"));
    let mut models: HashMap<String, Justin> = HashMap::new();
    for (name, files) in fixture["models"].as_object().unwrap() {
        let export = PathBuf::from(files["export"].as_str().unwrap());
        if !export.is_file() {
            eprintln!("skipped model {name}: {} is missing", export.display());
            continue;
        }
        let location = files["location"].as_str().map(Source::local);
        let options = LoadOptions {
            location,
            ..Default::default()
        };
        models.insert(
            name.clone(),
            Justin::from_pretrained(Source::Local(export), options).unwrap(),
        );
    }
    // Question heads: absolute differences. Location heads (logits around 10): relative to the
    // largest logit, plus the null probability.
    let (mut logit_diff, mut probability_diff, mut location_relative, mut null_diff) =
        (0f32, 0f32, 0f32, 0f32);
    let (mut cases, mut queries) = (0, 0);
    for case in fixture["cases"].as_array().unwrap() {
        let Some(model) = models.get(case["model"].as_str().unwrap()) else {
            continue;
        };
        let input = Input {
            series: matrix(&case["series"]),
            channel_names: strings(&case["channel_names"]),
            step: case["step"].as_f64(),
            baseline: baseline(&case["baseline"]),
            queries: case["queries"]
                .as_array()
                .unwrap()
                .iter()
                .map(query)
                .collect(),
        };
        let outputs = model.forward(&input).unwrap();
        let id = case["id"].as_str().unwrap();
        for (output, expected) in outputs.iter().zip(case["expected"].as_array().unwrap()) {
            let wanted_logits = floats(&expected["logits"]);
            let (logits, probabilities): (Vec<f32>, Vec<f32>) = match output {
                Output::Choice {
                    logits,
                    probabilities,
                    ..
                }
                | Output::Score {
                    logits,
                    probabilities,
                    ..
                } => (logits.clone(), probabilities.clone()),
                Output::YesNo {
                    logit, probability, ..
                } => (vec![*logit], vec![*probability]),
                Output::Location {
                    logits,
                    null_probability,
                    location,
                    ..
                } => {
                    let wanted = &expected["location"];
                    match location {
                        None => assert!(wanted.is_null(), "{id}: expected a location"),
                        Some(found) => {
                            assert_eq!(
                                found.channel as u64,
                                wanted["channel"].as_u64().unwrap(),
                                "{id} channel"
                            );
                            assert_eq!(
                                found.patch as u64,
                                wanted["patch"].as_u64().unwrap(),
                                "{id} patch"
                            );
                            assert_eq!(
                                found.start as u64,
                                wanted["start"].as_u64().unwrap(),
                                "{id} start"
                            );
                            assert_eq!(
                                found.end as u64,
                                wanted["end"].as_u64().unwrap(),
                                "{id} end"
                            );
                        }
                    }
                    (logits.clone(), vec![*null_probability])
                }
            };
            let wanted_probabilities = match &expected["probabilities"] {
                Value::Null => vec![expected["null_probability"].as_f64().unwrap() as f32],
                value => floats(value),
            };
            let _finite: Vec<usize> = (0..wanted_logits.len())
                .filter(|&i| wanted_logits[i].is_finite())
                .collect();
            let mut local = 0f32;
            let mut size = 0f32;
            for (i, wanted) in wanted_logits.iter().enumerate() {
                assert_eq!(
                    wanted.is_finite(),
                    logits[i].is_finite(),
                    "{id}: masked logits differ"
                );
                if wanted.is_finite() {
                    local = local.max((logits[i] - wanted).abs());
                    size = size.max(wanted.abs());
                }
            }
            if std::env::var("JUSTIN_PARITY_VERBOSE").is_ok() {
                println!("{id}: |diff| {local:.2e} max |logit| {size:.1}");
            }
            let probabilities_diff = max_diff(&probabilities, &wanted_probabilities);
            if matches!(output, Output::Location { .. }) {
                location_relative = location_relative.max(local / size.max(1.0));
                null_diff = null_diff.max(probabilities_diff);
            } else {
                logit_diff = logit_diff.max(local);
                probability_diff = probability_diff.max(probabilities_diff);
            }
            let best = |v: &[f32]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
            assert_eq!(
                best(&logits),
                best(&wanted_logits),
                "{id}: a different answer"
            );
            queries += 1;
        }
        cases += 1;
    }
    println!(
        "justin: {cases} cases, {queries} queries; question heads: logits max |diff| {logit_diff:.2e}, \
         probabilities {probability_diff:.2e}; location: logits max relative diff {location_relative:.2e}, \
         null probability {null_diff:.2e}"
    );
    assert!(cases > 0);
    assert!(logit_diff <= 1e-4, "logits {logit_diff}");
    assert!(probability_diff <= 1e-5, "probabilities {probability_diff}");
    assert!(
        location_relative <= 1e-4,
        "location logits {location_relative}"
    );
    assert!(null_diff <= 1e-4, "null probability {null_diff}");
}
