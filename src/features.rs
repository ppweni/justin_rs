//! What a model reads besides the series: per-channel deviation features and the features of the
//! channels a question and its options name. A port of `justin/features.py`,
//! `justin/data/deviation.py` and `justin/data/referents.py` (float64, numpy semantics).

use std::sync::LazyLock;

use regex::Regex;

use crate::error::{Result, input_error};

/// Must equal the `feature_version` an export records.
pub const FEATURE_VERSION: &str = "1";
pub const N_FEATURES: usize = 9;
const FOUND: usize = N_FEATURES;
const AGG: usize = 6;
pub const ENTITY: usize = N_FEATURES + 1 + AGG;
const CONTEXT: usize = 4;
const EXTRA: usize = 8;
pub const OPTION: usize = ENTITY + EXTRA;
const PAIR: usize = 5;
pub const QUESTION: usize = 2 * ENTITY + PAIR + CONTEXT;
const LOG_SCALE: f64 = 8.0;
const SAMPLING: &str = "one point every";

static QUOTED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"'([^']+)'").unwrap());
static TIME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").unwrap());
static NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^-?\d+(\.\d+)?$").unwrap());
static RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"The time range is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) to (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})").unwrap()
});
static SERIES_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^time-series (\d+): ").unwrap());

/// What counts as normal behavior when the series is put in baseline units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Baseline {
    /// The first share of the series (Python's default 0.3).
    Fraction(f64),
    /// An index range `[start, end)` of normal behavior.
    Range(usize, usize),
    /// The series is already in baseline units.
    Given,
}

impl Default for Baseline {
    fn default() -> Self {
        Baseline::Fraction(0.3)
    }
}

/// The inputs of one question as the model takes them.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// [C][T] in baseline units.
    pub series: Vec<Vec<f32>>,
    pub question: String,
    /// [C][N_FEATURES]
    pub channel_features: Vec<[f32; N_FEATURES]>,
    pub referent_question: Vec<f32>,
    /// [options][OPTION]
    pub referent_options: Vec<Vec<f32>>,
}

/// `justin.features.prepare`: baseline units, deviation features, the sampling sentence and the
/// referent features of a question and its options.
pub fn prepare(
    series: &[Vec<f32>],
    question: &str,
    options: &[String],
    channel_names: Option<&[String]>,
    step: Option<f64>,
    baseline: Baseline,
) -> Result<Prepared> {
    if series.is_empty()
        || series[0].is_empty()
        || series.iter().any(|r| r.len() != series[0].len())
    {
        return Err(input_error(
            "series must have shape [channels, time] or [time]",
        ));
    }
    if series.iter().flatten().any(|v| !v.is_finite()) {
        return Err(input_error("series must be finite"));
    }
    if let Some(names) = channel_names
        && names.len() != series.len()
    {
        return Err(input_error("channel_names needs one name per channel"));
    }
    let values: Vec<Vec<f64>> = series
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    let z = normalize(&values, baseline)?;
    let features = deviation_features(&z, step.unwrap_or(1.0));
    let mut question = question.to_string();
    if let Some(step) = step
        && !question.contains(SAMPLING)
    {
        question = format!(
            "{question}\nThe series has one point every {} s.",
            format_g(step)
        );
    }
    let labels: Vec<String> = match channel_names {
        Some(names) => names.to_vec(),
        None => (0..z.len()).map(|i| format!("channel-{i}")).collect(),
    };
    let grouped = labels.iter().any(|n| n.starts_with("time-series "))
        || question.to_lowercase().contains("time-series");
    let (referent_question, referent_options) = if grouped {
        grouped_referents(options, &labels, &features, &question)?
    } else {
        referent_features(&question, options, &labels, &features)
    };
    Ok(Prepared {
        series: z
            .iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect(),
        question,
        channel_features: features,
        referent_question,
        referent_options,
    })
}

/// Python's `{:g}`: six significant digits, no trailing zeros, exponent form outside [1e-4, 1e6).
pub fn format_g(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".into()
        } else if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let exponent = format!("{:.5e}", value);
    let (mantissa, exp) = exponent.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    if (-4..6).contains(&exp) {
        let decimals = (5 - exp).max(0) as usize;
        let fixed = format!("{value:.decimals$}");
        if fixed.contains('.') {
            fixed
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string()
        } else {
            fixed
        }
    } else {
        let mantissa = if mantissa.contains('.') {
            mantissa
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string()
        } else {
            mantissa.to_string()
        };
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// `deviation.normalize`.
pub fn normalize(values: &[Vec<f64>], baseline: Baseline) -> Result<Vec<Vec<f64>>> {
    let length = values[0].len();
    let (start, end) = match baseline {
        Baseline::Given => return Ok(values.to_vec()),
        Baseline::Range(start, end) => {
            if end < 2 || start >= end - 1 || end - 1 > length {
                return Err(input_error(
                    "baseline must be a range of at least two points inside the series",
                ));
            }
            (start, end)
        }
        Baseline::Fraction(fraction) => {
            if !(0.0 < fraction && fraction < 1.0) {
                return Err(input_error("a baseline share must be between 0 and 1"));
            }
            (0, ((length as f64 * fraction) as usize).max(2))
        }
    };
    Ok(values
        .iter()
        .map(|row| {
            let reference = &row[start..end.min(length)];
            let spread = std(reference).max(0.05 * std(row)) + 1e-9;
            let center = mean(reference);
            row.iter().map(|v| (v - center) / spread).collect()
        })
        .collect())
}

fn log_size(x: f64) -> f64 {
    x.ln_1p() / LOG_SCALE
}

/// `deviation.deviation_features`: [C][9] from `z` [C][T] with `step` seconds between points.
pub fn deviation_features(z: &[Vec<f64>], step: f64) -> Vec<[f32; N_FEATURES]> {
    z.iter()
        .map(|row| {
            let z: Vec<f64> = row.iter().map(|v| v.abs()).collect();
            let length = z.len();
            let head = &z[..((length as f64 * 0.4) as usize).max(1)];
            let tail = &z[(length as f64 * 0.6) as usize..];
            let beyond: Vec<bool> = z.iter().map(|&v| v > 3.0).collect();
            let (found, first, last) = if length >= 3 {
                let sustained: Vec<bool> = (0..length - 2)
                    .map(|i| beyond[i] && beyond[i + 1] && beyond[i + 2])
                    .collect();
                let first = sustained.iter().position(|&s| s);
                let last_index = sustained.iter().rposition(|&s| s);
                match (first, last_index) {
                    (Some(f), Some(l)) => (true, f, l + 3),
                    _ => (false, 0, 3),
                }
            } else {
                (false, 0, 0)
            };
            let ended = found && (length as f64 - last as f64) >= (3f64).max(0.03 * length as f64);
            let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            [
                log_size(median(tail)),
                log_size(max),
                log_size(median(head)),
                if found {
                    first as f64 / length as f64
                } else {
                    1.0
                },
                found as u8 as f64,
                log_size(if found { first as f64 * step } else { 0.0 }),
                log_size(length as f64 * step),
                if ended {
                    last as f64 / length as f64
                } else {
                    1.0
                },
                ended as u8 as f64,
            ]
            .map(|v| v as f32)
        })
        .collect()
}

/// Which channels the question quotes (its first two quoted names, as the referents read them).
pub fn quoted_channels(question: &str, names: &[String]) -> Vec<bool> {
    let mut quoted = vec![false; names.len()];
    for capture in QUOTED.captures_iter(question).take(2) {
        for index in resolve(&capture[1], names) {
            quoted[index] = true;
        }
    }
    quoted
}

/// `referents.resolve`: an exact channel name, else every channel of a service prefix.
fn resolve(text: &str, names: &[String]) -> Vec<usize> {
    let text = text.trim();
    let exact: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| *n == text)
        .map(|(i, _)| i)
        .collect();
    if !exact.is_empty() {
        return exact;
    }
    names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.split('_').next().unwrap_or("") == text)
        .map(|(i, _)| i)
        .collect()
}

fn aggregate(indices: &[usize], features: &[[f32; N_FEATURES]]) -> [f32; AGG] {
    let rows: Vec<&[f32; N_FEATURES]> = indices.iter().map(|&i| &features[i]).collect();
    let found: Vec<bool> = rows.iter().map(|r| r[4] > 0.0).collect();
    let n = rows.len() as f64;
    let count = found.iter().filter(|&&f| f).count();
    let earliest = rows
        .iter()
        .zip(&found)
        .filter(|(_, f)| **f)
        .map(|(r, _)| r[3])
        .fold(f32::INFINITY, f32::min);
    [
        (rows.len().min(20) as f64 / 20.0) as f32,
        (count as f64 / n) as f32,
        (count.min(20) as f64 / 20.0) as f32,
        (rows.iter().map(|r| r[1] as f64).sum::<f64>() / n) as f32,
        (rows.iter().map(|r| r[0] as f64).sum::<f64>() / n) as f32,
        if count > 0 { earliest } else { 1.0 },
    ]
}

fn entity(indices: &[usize], features: &[[f32; N_FEATURES]]) -> Vec<f32> {
    if indices.is_empty() {
        return vec![0.0; ENTITY];
    }
    // The first index with the largest maximum deviation, as Python's max().
    let mut best = indices[0];
    for &i in &indices[1..] {
        if features[i][1] > features[best][1] {
            best = i;
        }
    }
    let mut out = features[best].to_vec();
    out.push(1.0);
    out.extend_from_slice(&aggregate(indices, features));
    out
}

fn context(features: &[[f32; N_FEATURES]]) -> [f32; CONTEXT] {
    let n = features.len() as f64;
    let found = features.iter().filter(|r| r[4] > 0.0).count();
    [
        (found as f64 / n) as f32,
        features
            .iter()
            .map(|r| r[1])
            .fold(f32::NEG_INFINITY, f32::max),
        (features.iter().map(|r| r[1] as f64).sum::<f64>() / n) as f32,
        (features.len().min(100) as f64 / 100.0) as f32,
    ]
}

fn summary(features: &[[f32; N_FEATURES]], indices: &[usize]) -> [f32; 4] {
    let rows: Vec<&[f32; N_FEATURES]> = indices
        .iter()
        .filter(|&&i| features[i][4] > 0.0)
        .map(|&i| &features[i])
        .collect();
    if rows.is_empty() {
        return [1.0, 1.0, 0.0, 0.0];
    }
    let max = |k: usize| rows.iter().map(|r| r[k]).fold(f32::NEG_INFINITY, f32::max);
    let end = if rows.iter().all(|r| r[8] > 0.0) {
        max(7)
    } else {
        1.0
    };
    [
        rows.iter().map(|r| r[3]).fold(f32::INFINITY, f32::min),
        end,
        max(0),
        max(1),
    ]
}

fn epoch(text: &str) -> Result<f64> {
    let time = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S")
        .map_err(|e| input_error(format!("invalid timestamp {text}: {e}")))?;
    Ok(time.and_utc().timestamp() as f64)
}

fn option_extras(option: &str, question: &str, facts: &[f32; 4]) -> Result<[f32; EXTRA]> {
    let mut out = [0f32; EXTRA];
    let text = option.trim();
    let window = RANGE.captures(question).filter(|_| TIME.is_match(text));
    if let Some(window) = window {
        let (start, end) = (epoch(&window[1])?, epoch(&window[2])?);
        let fraction = (epoch(text)? - start) / (end - start).max(1.0);
        out[..4].copy_from_slice(&[
            1.0,
            fraction as f32,
            (fraction - facts[0] as f64) as f32,
            (fraction - facts[1] as f64) as f32,
        ]);
    } else if NUMBER.is_match(text) {
        let value: f64 = text
            .parse()
            .map_err(|_| input_error(format!("invalid number {text}")))?;
        let size = value.abs().ln_1p() / LOG_SCALE;
        out[4..].copy_from_slice(&[
            1.0,
            size as f32,
            (size - facts[2] as f64) as f32,
            (size - facts[3] as f64) as f32,
        ]);
    }
    Ok(out)
}

fn question_vector(a: &[f32], b: &[f32], window: Option<&[f32; CONTEXT]>) -> Vec<f32> {
    let both = a[FOUND] * b[FOUND];
    let timed = a[FOUND] * b[FOUND] * a[4] * b[4];
    let pair = [
        both,
        timed,
        timed * 5.0 * (a[3] - b[3]),
        timed * (a[5] - b[5]),
        both * (a[1] - b[1]),
    ];
    let mut out = Vec::with_capacity(QUESTION);
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out.extend_from_slice(&pair);
    match window {
        Some(w) => out.extend_from_slice(w),
        None => out.extend_from_slice(&[0.0; CONTEXT]),
    }
    out
}

fn referent_features(
    question: &str,
    options: &[String],
    names: &[String],
    features: &[[f32; N_FEATURES]],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut quoted: Vec<Vec<f32>> = QUOTED
        .captures_iter(question)
        .take(2)
        .map(|c| entity(&resolve(&c[1], names), features))
        .collect();
    while quoted.len() < 2 {
        quoted.push(vec![0.0; ENTITY]);
    }
    let options = options
        .iter()
        .map(|o| {
            let mut row = entity(&resolve(o, names), features);
            row.extend_from_slice(&[0.0; EXTRA]);
            row
        })
        .collect();
    (
        question_vector(&quoted[0], &quoted[1], Some(&context(features))),
        options,
    )
}

fn grouped_referents(
    options: &[String],
    names: &[String],
    features: &[[f32; N_FEATURES]],
    question: &str,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let series: Vec<u64> = names
        .iter()
        .map(|n| {
            SERIES_PREFIX
                .captures(n)
                .and_then(|c| c[1].parse().ok())
                .unwrap_or(1)
        })
        .collect();
    let mut numbers = series.clone();
    numbers.sort();
    numbers.dedup();
    let groups: Vec<Vec<usize>> = numbers
        .iter()
        .map(|number| {
            series
                .iter()
                .enumerate()
                .filter(|(_, g)| *g == number)
                .map(|(i, _)| i)
                .collect()
        })
        .collect();
    let first = entity(&groups[0], features);
    let second = if groups.len() > 1 {
        entity(&groups[1], features)
    } else {
        vec![0.0; ENTITY]
    };
    let facts = summary(features, &groups[0]);
    let labels: Vec<String> = names
        .iter()
        .map(|n| SERIES_PREFIX.replace(n, "").into_owned())
        .collect();
    let mut rows = Vec::with_capacity(options.len());
    for option in options {
        let parts: Vec<&str> = option.split(", ").map(str::trim).collect();
        let hits: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, label)| parts.contains(&label.as_str()))
            .map(|(i, _)| i)
            .collect();
        let mut row = entity(&hits, features);
        row.extend_from_slice(&option_extras(option, question, &facts)?);
        rows.push(row);
    }
    Ok((
        question_vector(&first, &second, Some(&context(features))),
        rows,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_g_matches_python() {
        for (value, expected) in [
            (1.0, "1"),
            (60.0, "60"),
            (0.5, "0.5"),
            (1e-5, "1e-05"),
            (1234567.0, "1.23457e+06"),
            (15.25, "15.25"),
            (0.0001, "0.0001"),
            (123456.0, "123456"),
        ] {
            assert_eq!(format_g(value), expected, "{value}");
        }
    }

    #[test]
    fn median_and_std_follow_numpy() {
        assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), 2.5);
        assert!((std(&[1.0, 2.0, 3.0, 4.0]) - 1.118_033_988_749_895).abs() < 1e-15);
    }
}
