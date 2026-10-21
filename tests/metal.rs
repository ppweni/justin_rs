//! The production cases on Metal against the CPU (`cargo test --release --features metal`).
#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};

use candle_core::Device;
use justin::{Input, Justin, LoadOptions, Output, Query, Source};

#[test]
fn metal_matches_the_cpu() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models.json");
    if !path.is_file() {
        eprintln!("skipped: no fixtures");
        return;
    }
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let files = &fixture["models"]["prod"];
    let load = |device: Device| {
        Justin::from_pretrained(
            Source::Local(PathBuf::from(files["export"].as_str().unwrap())),
            LoadOptions {
                device,
                location: Some(Source::Local(PathBuf::from(
                    files["location"].as_str().unwrap(),
                ))),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let (cpu, metal) = (load(Device::Cpu), load(Device::new_metal(0).unwrap()));
    let mut worst = 0f32;
    let mut count = 0;
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["model"] == "prod")
        .step_by(3)
    {
        let series: Vec<Vec<f32>> = case["series"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().map_or(f32::NAN, |x| x as f32))
                    .collect()
            })
            .collect();
        let names = case["channel_names"]
            .as_array()
            .map(|n| n.iter().map(|s| s.as_str().unwrap().to_string()).collect());
        let queries = case["queries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                let question = q["question"].as_str().unwrap().to_string();
                match &q["answers"] {
                    serde_json::Value::Null => Query::Location { question },
                    serde_json::Value::String(s) if s == "yes_no" => Query::yes_no(question),
                    serde_json::Value::String(_) => Query::Question {
                        question,
                        answers: justin::Answers::Score,
                    },
                    other => Query::Question {
                        question,
                        answers: justin::Answers::Options(
                            other["options"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|s| s.as_str().unwrap().to_string())
                                .collect(),
                        ),
                    },
                }
            })
            .collect();
        let input = Input {
            series,
            channel_names: names,
            step: case["step"].as_f64(),
            queries,
            ..Default::default()
        };
        for (a, b) in cpu
            .forward(&input)
            .unwrap()
            .iter()
            .zip(metal.forward(&input).unwrap().iter())
        {
            let logits = |o: &Output| -> Vec<f32> {
                match o {
                    Output::Choice { logits, .. }
                    | Output::Score { logits, .. }
                    | Output::Location { logits, .. } => logits.clone(),
                    Output::YesNo { logit, .. } => vec![*logit],
                }
            };
            for (x, y) in logits(a).iter().zip(logits(b)) {
                if x.is_finite() {
                    worst = worst.max((x - y).abs() / x.abs().max(1.0));
                }
            }
            count += 1;
        }
    }
    println!("metal vs cpu: {count} queries, max relative logit diff {worst:.2e}");
    assert!(worst <= 1e-3, "{worst}");
}
