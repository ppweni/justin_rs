//! Loading from the Hub cache and from local directories, and the errors a wrong pairing or
//! missing input gives. Needs the fixtures (`scripts/make_fixtures.py`) and the cached backbones.

use std::path::{Path, PathBuf};

use justin::source::default_cache;
use justin::{Backbones, Error, Input, Justin, LoadOptions, Output, Query, Source};

fn models() -> Option<serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models.json");
    if !path.is_file() {
        eprintln!("skipped: no fixtures");
        return None;
    }
    Some(
        serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(path).unwrap()).unwrap()
            ["models"]
            .clone(),
    )
}

fn file(models: &serde_json::Value, name: &str, kind: &str) -> PathBuf {
    PathBuf::from(models[name][kind].as_str().unwrap())
}

fn input(queries: Vec<Query>) -> Input {
    let series: Vec<Vec<f32>> = (0..2)
        .map(|c| {
            (0..300)
                .map(|t| ((t * (c + 1)) as f32 * 0.1).sin())
                .collect()
        })
        .collect();
    Input {
        series,
        channel_names: Some(vec!["cartservice_cpu".into(), "redis_mem".into()]),
        step: Some(4.0),
        queries,
        ..Default::default()
    }
}

/// A Hugging Face style cache holding `file` as `owner/justin` at `main`, with the real cached
/// backbone repos linked in, so nothing is downloaded.
fn fake_cache(dir: &Path, file: &Path) -> PathBuf {
    let cache = dir.join("hub");
    let repo = cache.join("models--owner--justin");
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::create_dir_all(repo.join("snapshots/0123abcd")).unwrap();
    std::fs::write(repo.join("refs/main"), "0123abcd").unwrap();
    std::fs::copy(
        file,
        repo.join("snapshots/0123abcd")
            .join(file.file_name().unwrap()),
    )
    .unwrap();
    for backbone in [
        "models--mistralai--Timestral-5M",
        "models--sentence-transformers--all-MiniLM-L6-v2",
    ] {
        std::os::unix::fs::symlink(default_cache().join(backbone), cache.join(backbone)).unwrap();
    }
    cache
}

#[cfg(unix)]
#[test]
fn a_model_loads_from_the_hub_cache_and_from_local_backbones_alike() {
    let Some(models) = models() else { return };
    let export = file(&models, "prod", "export");
    let location = file(&models, "prod", "location");
    let dir = tempfile::tempdir().unwrap();
    let cache = fake_cache(dir.path(), &export);
    let hub = Justin::from_pretrained(
        Source::hub(
            "owner/justin",
            export.file_name().unwrap().to_str().unwrap(),
        ),
        LoadOptions {
            cache_dir: Some(cache),
            location: Some(Source::Local(location.clone())),
            ..Default::default()
        },
    )
    .unwrap();
    let snapshot =
        |repo: &str, revision: &str| default_cache().join(repo).join("snapshots").join(revision);
    let local = Justin::from_pretrained(
        Source::Local(export),
        LoadOptions {
            backbones: Backbones::Local {
                series: snapshot(
                    "models--mistralai--Timestral-5M",
                    "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
                ),
                text: snapshot(
                    "models--sentence-transformers--all-MiniLM-L6-v2",
                    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                ),
            },
            location: Some(Source::Local(location)),
            ..Default::default()
        },
    )
    .unwrap();
    let queries = vec![
        Query::yes_no("Is there an anomaly in this time window?"),
        Query::choice(
            "Which service is the root cause?",
            &["cartservice", "redis", "No service is faulty"],
        ),
        Query::locate("Locate the onset of the spike in redis_mem in this window."),
    ];
    let a = hub.forward(&input(queries.clone())).unwrap();
    let b = local.forward(&input(queries)).unwrap();
    assert_eq!(a, b);
    assert!(matches!(a[0], Output::YesNo { .. }));
    assert!(
        matches!(&a[1], Output::Choice { probabilities, answer: Some(_), .. } if (probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5)
    );
    assert!(matches!(&a[2], Output::Location { logits, .. } if logits.len() == 2 * 10 + 1));
}

#[test]
fn a_location_head_of_another_model_is_refused() {
    let Some(models) = models() else { return };
    let result = Justin::from_pretrained(
        Source::Local(file(&models, "deep", "export")),
        LoadOptions {
            location: Some(Source::Local(file(&models, "prod-pointer", "location"))),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(Error::Format(m)) if m.contains("does not belong")));
}

#[test]
fn missing_inputs_and_heads_are_errors() {
    let Some(models) = models() else { return };
    let model = Justin::from_pretrained(
        Source::Local(file(&models, "prod", "export")),
        LoadOptions::default(),
    )
    .unwrap();
    let located = model.forward(&input(vec![Query::locate("Where does the spike start?")]));
    assert!(matches!(located, Err(Error::Input(m)) if m.contains("no location head")));
    let mut unnamed = input(vec![Query::yes_no("Is it abnormal?")]);
    unnamed.channel_names = None;
    assert!(matches!(model.forward(&unnamed), Err(Error::Input(m)) if m.contains("channel names")));
    let one_option = input(vec![Query::choice("Which?", &["only"])]);
    assert!(matches!(model.forward(&one_option), Err(Error::Input(m)) if m.contains("2 to 255")));
    let mut ragged = input(vec![Query::yes_no("Is it abnormal?")]);
    ragged.series[1].pop();
    assert!(model.forward(&ragged).is_err());
    let not_an_export = Justin::from_pretrained(
        Source::Local(file(&models, "prod", "location")),
        LoadOptions::default(),
    );
    assert!(matches!(not_an_export, Err(Error::Format(m)) if m.contains("not a Justin export")));
}

#[test]
fn an_empty_query_list_gives_no_outputs() {
    let Some(models) = models() else { return };
    let model = Justin::from_pretrained(
        Source::Local(file(&models, "deep", "export")),
        LoadOptions::default(),
    )
    .unwrap();
    assert!(model.forward(&input(vec![])).unwrap().is_empty());
}
