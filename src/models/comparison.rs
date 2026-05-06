/////////////////////////////////////////////////////////////////////////////////////////////\
//
// Model comparison utilities (metrics, information criteria, and CV summaries).
//
// Created on: 25 Jan 2026     Author: Tobias Kragholm
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Model comparison utilities
//!
//! Provides in-sample metrics, quasi information criteria, and cross-validation
//! summaries for two-part, Tweedie, and log-normal models.

use comfy_table::{
    Attribute, Cell, Color, ContentArrangement, Table, presets::UTF8_FULL_CONDENSED,
};
use std::time::Instant;
use thiserror::Error;
use tracing::info;

use super::lognormal::{
    LogNormalError, LogNormalModel, LogNormalOptions, fit_lognormal_smearing_input,
    log_likelihood as lognormal_log_likelihood,
};
use super::selection::{
    CrossValidationError, CrossValidationOptions, CrossValidationResult, ModelFitMetrics,
    TweedieCandidate, compute_information_criteria, compute_model_fit_metrics,
    cross_validate_models_input, fit_tweedie_candidates_input,
};
use super::tweedie::{
    TweedieError, TweedieModel, TweedieOptions, fit_tweedie_input,
    quasi_log_likelihood as tweedie_quasi_log_likelihood,
};
use super::two_part::{
    FitOptions, TwoPartError, fit_two_part_input, log_likelihood as two_part_log_likelihood,
};
use crate::input::{InputError, ModelInput};
use crate::models::Model;

type TwoPartFit = (
    super::two_part::TwoPartModel,
    super::two_part::TwoPartReport,
);

#[derive(Debug, Clone, Default)]
pub struct FittedComparisonModels {
    pub two_part: Option<TwoPartFit>,
    pub two_part_elastic_net: Option<TwoPartFit>,
    pub tweedie: Option<TweedieModel>,
    pub lognormal: Option<LogNormalModel>,
}

/// Configuration for model comparisons.
#[derive(bon::Builder, Debug, Clone)]
pub struct ModelComparisonOptions {
    #[builder(default = vec![1.2, 1.5, 1.8])]
    pub tweedie_powers: Vec<f64>,
    #[builder(default)]
    pub two_part_options: FitOptions,
    pub two_part_elastic_net_options: Option<FitOptions>,
    #[builder(default)]
    pub tweedie_options: TweedieOptions,
    #[builder(default)]
    pub lognormal_options: LogNormalOptions,
    #[builder(default)]
    pub cv_options: CrossValidationOptions,
    #[builder(default = true)]
    pub include_two_part: bool,
    #[builder(default = true)]
    pub include_tweedie: bool,
    #[builder(default = true)]
    pub include_lognormal: bool,
}

impl Default for ModelComparisonOptions {
    fn default() -> Self {
        Self {
            tweedie_powers: vec![1.2, 1.5, 1.8],
            two_part_options: FitOptions::default(),
            two_part_elastic_net_options: None,
            tweedie_options: TweedieOptions::default(),
            lognormal_options: LogNormalOptions::default(),
            cv_options: CrossValidationOptions::default(),
            include_two_part: true,
            include_tweedie: true,
            include_lognormal: true,
        }
    }
}

/// Errors returned by the comparison workflow.
#[derive(Debug, Error)]
pub enum ModelComparisonError {
    #[error("invalid model input: {0}")]
    Input(#[from] InputError),
    #[error("two-part fit failed: {0}")]
    TwoPart(#[from] TwoPartError),
    #[error("tweedie fit failed: {0}")]
    Tweedie(#[from] TweedieError),
    #[error("log-normal fit failed: {0}")]
    LogNormal(#[from] LogNormalError),
    #[error("cross-validation failed: {0}")]
    CrossValidation(#[from] CrossValidationError),
}

/// Per-model metrics with a descriptive label.
#[derive(Debug, Clone)]
pub struct ModelScore {
    pub name: String,
    pub metrics: ModelFitMetrics,
}

/// Per-model information criteria with a descriptive label.
#[derive(Debug, Clone)]
pub struct ModelInformationCriteria {
    pub name: String,
    pub loglik: f64,
    pub aic: f64,
    pub bic: f64,
}

/// Tweedie CV ranking row (metrics + in-sample IC).
#[derive(Debug, Clone)]
pub struct TweedieRankingRow {
    pub power: f64,
    pub metrics: ModelFitMetrics,
    pub aic: f64,
    pub bic: f64,
}

/// Comparison output for model evaluation.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    pub in_sample: Vec<ModelScore>,
    pub information_criteria: Vec<ModelInformationCriteria>,
    pub cv_summary: Vec<ModelScore>,
    pub cv_ranking: Vec<ModelScore>,
    pub tweedie_ic: Vec<ModelInformationCriteria>,
    pub tweedie_cv_ranking: Vec<TweedieRankingRow>,
}

/// Rendered tables for a comparison report.
#[derive(Debug, Clone)]
pub struct ComparisonTables {
    pub in_sample: String,
    pub information_criteria: String,
    pub tweedie_candidates: String,
    pub cv_summary: String,
    pub cv_ranking: String,
    pub tweedie_cv_ranking: String,
}

#[derive(Debug, Clone, Default)]
struct InSamplePredictions {
    two_part: Option<super::two_part::TwoPartPrediction>,
    two_part_elastic_net: Option<super::two_part::TwoPartPrediction>,
    tweedie: Option<super::tweedie::TweediePrediction>,
    lognormal: Option<super::lognormal::LogNormalPrediction>,
}

/// Run model comparisons for a `ModelInput` dataset.
///
/// Notes:
/// - Tweedie AIC/BIC use quasi-likelihood (-0.5 * deviance).
/// - Log-normal log-likelihood is evaluated on positive outcomes only.
///
/// # Errors
///
/// Returns `ModelComparisonError` if any model fit or cross-validation step fails.
pub fn compare_models_input(
    input: &ModelInput,
    options: &ModelComparisonOptions,
) -> Result<ModelComparison, ModelComparisonError> {
    compare_models_input_with_fits(input, options).map(|(comparison, _)| comparison)
}

#[allow(
    clippy::missing_errors_doc,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    reason = "comparison orchestration logs millisecond timings and returns a single aggregated report"
)]
pub fn compare_models_input_with_fits(
    input: &ModelInput,
    options: &ModelComparisonOptions,
) -> Result<(ModelComparison, FittedComparisonModels), ModelComparisonError> {
    input.validate()?;

    let total_started_at = Instant::now();

    let base_fits_started_at = Instant::now();
    let two_part_default = options
        .include_two_part
        .then(|| fit_two_part_input(input, options.two_part_options))
        .transpose()?;
    let two_part_elastic_net_options = options.two_part_elastic_net_options;
    let two_part_elastic_net = options
        .include_two_part
        .then_some(two_part_elastic_net_options)
        .flatten()
        .map(|elastic_net_options| fit_two_part_input(input, elastic_net_options))
        .transpose()?;

    let tweedie_power = options
        .tweedie_powers
        .iter()
        .copied()
        .find(|power| (*power - 1.5).abs() < 1.0e-12)
        .or_else(|| options.tweedie_powers.first().copied())
        .unwrap_or(1.5);
    let tweedie_model = options
        .include_tweedie
        .then(|| fit_tweedie_input(input, tweedie_power, options.tweedie_options))
        .transpose()?
        .map(|(model, _report)| model);
    let lognormal_model = if options.include_lognormal {
        let (model, _report) = fit_lognormal_smearing_input(input, options.lognormal_options)?;
        Some(model)
    } else {
        None
    };
    info!(
        step = "base_fits",
        rows = input.outcome.nrows(),
        features = input.design_matrix.ncols(),
        include_two_part = options.include_two_part,
        include_tweedie = options.include_tweedie,
        include_lognormal = options.include_lognormal,
        tweedie_powers = options.tweedie_powers.len(),
        duration_ms = base_fits_started_at.elapsed().as_millis() as u64,
        "semicontinuous_models comparison substep complete"
    );

    let selection_started_at = Instant::now();
    let tweedie_candidates = fit_tweedie_candidates_input(input, &options.tweedie_powers);
    info!(
        step = "selection",
        tweedie_candidates = tweedie_candidates.len(),
        duration_ms = selection_started_at.elapsed().as_millis() as u64,
        "semicontinuous_models comparison substep complete"
    );

    let scoring_started_at = Instant::now();
    let predictions = build_in_sample_predictions(
        input,
        two_part_default.as_ref(),
        two_part_elastic_net.as_ref(),
        tweedie_model.as_ref(),
        lognormal_model.as_ref(),
    );
    let in_sample = build_in_sample(input, &predictions, tweedie_model.as_ref());
    let information_criteria = build_information_criteria(
        input,
        &predictions,
        tweedie_model.as_ref(),
        lognormal_model.as_ref(),
    );
    let tweedie_ic = build_tweedie_candidate_ic(&tweedie_candidates);
    info!(
        step = "scoring",
        in_sample_rows = in_sample.len(),
        information_criteria_rows = information_criteria.len(),
        tweedie_ic_rows = tweedie_ic.len(),
        duration_ms = scoring_started_at.elapsed().as_millis() as u64,
        "semicontinuous_models comparison substep complete"
    );

    let cv_default_started_at = Instant::now();
    let cv_default = cross_validate_models_input(
        input,
        &options.tweedie_powers,
        CrossValidationOptions {
            two_part_options: options.two_part_options,
            tweedie_options: options.tweedie_options,
            lognormal_options: options.lognormal_options,
            include_two_part: options.include_two_part,
            include_lognormal: options.include_lognormal,
            ..options.cv_options
        },
    )?;
    info!(
        step = "cv_default",
        folds_used = cv_default.folds_used,
        tweedie_candidates = cv_default.tweedie_candidates.len(),
        include_two_part = cv_default.include_two_part,
        include_lognormal = cv_default.include_lognormal,
        duration_ms = cv_default_started_at.elapsed().as_millis() as u64,
        "semicontinuous_models comparison substep complete"
    );
    let cv_elastic_net_started_at = Instant::now();
    let cv_elastic_net = if let Some(elastic_net_options) = two_part_elastic_net_options {
        Some(cross_validate_models_input(
            input,
            &[],
            CrossValidationOptions {
                two_part_options: elastic_net_options,
                tweedie_options: options.tweedie_options,
                lognormal_options: options.lognormal_options,
                include_two_part: true,
                include_lognormal: options.include_lognormal,
                ..options.cv_options
            },
        )?)
    } else {
        None
    };
    info!(
        step = "cv_elastic_net",
        enabled = cv_elastic_net.is_some(),
        duration_ms = cv_elastic_net_started_at.elapsed().as_millis() as u64,
        "semicontinuous_models comparison substep complete"
    );
    let assembly_started_at = Instant::now();
    let cv_summary = build_cv_summary(&cv_default, cv_elastic_net.as_ref());
    let cv_ranking = build_cv_ranking(&cv_summary);
    let tweedie_cv_ranking = build_tweedie_cv_ranking(&cv_default, &tweedie_candidates);
    info!(
        step = "assemble_report",
        cv_summary_rows = cv_summary.len(),
        cv_ranking_rows = cv_ranking.len(),
        tweedie_cv_ranking_rows = tweedie_cv_ranking.len(),
        duration_ms = assembly_started_at.elapsed().as_millis() as u64,
        "semicontinuous_models comparison substep complete"
    );
    info!(
        step = "total",
        duration_ms = total_started_at.elapsed().as_millis() as u64,
        "semicontinuous_models comparison complete"
    );

    Ok((
        ModelComparison {
            in_sample,
            information_criteria,
            cv_summary,
            cv_ranking,
            tweedie_ic,
            tweedie_cv_ranking,
        },
        FittedComparisonModels {
            two_part: two_part_default,
            two_part_elastic_net,
            tweedie: tweedie_model,
            lognormal: lognormal_model,
        },
    ))
}

/// Render a comparison report to formatted tables using `comfy_table`.
#[must_use]
pub fn render_comparison_tables(report: &ModelComparison) -> ComparisonTables {
    let mut in_sample_table = make_table(&["model", "rmse", "mae", "rmsle", "r2"]);
    let in_sample_best = best_metrics(&report.in_sample);
    for score in &report.in_sample {
        add_metrics_row_highlight(
            &mut in_sample_table,
            &score.name,
            &score.metrics,
            &in_sample_best,
        );
    }

    let mut ic_table = make_table(&["model", "loglik", "aic", "bic"]);
    let ic_best = best_ic(&report.information_criteria);
    for item in &report.information_criteria {
        ic_table.add_row(vec![
            Cell::new(&item.name),
            highlight_metric_cell(item.loglik, ic_best.loglik, 2),
            highlight_metric_cell(item.aic, ic_best.aic, 2),
            highlight_metric_cell(item.bic, ic_best.bic, 2),
        ]);
    }

    let mut tweedie_ic_table = make_table(&["power", "loglik", "aic", "bic"]);
    let tweedie_ic_best = best_ic(&report.tweedie_ic);
    for item in &report.tweedie_ic {
        tweedie_ic_table.add_row(vec![
            Cell::new(&item.name),
            highlight_metric_cell(item.loglik, tweedie_ic_best.loglik, 2),
            highlight_metric_cell(item.aic, tweedie_ic_best.aic, 2),
            highlight_metric_cell(item.bic, tweedie_ic_best.bic, 2),
        ]);
    }

    let mut cv_table = make_table(&["model", "rmse", "mae", "rmsle", "r2"]);
    let cv_best = best_metrics(&report.cv_summary);
    for score in &report.cv_summary {
        add_metrics_row_highlight(&mut cv_table, &score.name, &score.metrics, &cv_best);
    }

    let mut cv_ranking_table = make_table(&["model", "rmse", "mae", "rmsle", "r2"]);
    let ranking_best = best_metrics(&report.cv_ranking);
    for score in &report.cv_ranking {
        add_metrics_row_highlight(
            &mut cv_ranking_table,
            &score.name,
            &score.metrics,
            &ranking_best,
        );
    }

    let mut tweedie_ranking_table =
        make_table(&["power", "rmse", "mae", "rmsle", "r2", "dev", "aic", "bic"]);
    let tweedie_best = best_tweedie_ranking(&report.tweedie_cv_ranking);
    for row in &report.tweedie_cv_ranking {
        tweedie_ranking_table.add_row(vec![
            Cell::new(format!("{:.2}", row.power)),
            highlight_metric_cell(row.metrics.rmse, tweedie_best.rmse, 4),
            highlight_metric_cell(row.metrics.mae, tweedie_best.mae, 4),
            highlight_metric_cell(row.metrics.rmsle, tweedie_best.rmsle, 4),
            highlight_metric_cell(row.metrics.r2, tweedie_best.r2, 4),
            highlight_metric_cell(row.metrics.deviance, tweedie_best.deviance, 4),
            highlight_metric_cell(row.aic, tweedie_best.aic, 2),
            highlight_metric_cell(row.bic, tweedie_best.bic, 2),
        ]);
    }

    ComparisonTables {
        in_sample: in_sample_table.to_string(),
        information_criteria: ic_table.to_string(),
        tweedie_candidates: tweedie_ic_table.to_string(),
        cv_summary: cv_table.to_string(),
        cv_ranking: cv_ranking_table.to_string(),
        tweedie_cv_ranking: tweedie_ranking_table.to_string(),
    }
}

fn build_in_sample_predictions(
    input: &ModelInput,
    two_part: Option<&TwoPartFit>,
    two_part_elastic_net: Option<&TwoPartFit>,
    tweedie: Option<&TweedieModel>,
    lognormal: Option<&LogNormalModel>,
) -> InSamplePredictions {
    InSamplePredictions {
        two_part: two_part.map(|fit| fit.0.predict(&input.design_matrix)),
        two_part_elastic_net: two_part_elastic_net.map(|fit| fit.0.predict(&input.design_matrix)),
        tweedie: tweedie.map(|model| model.predict(&input.design_matrix)),
        lognormal: lognormal.map(|model| model.predict(&input.design_matrix)),
    }
}

fn build_in_sample(
    input: &ModelInput,
    predictions: &InSamplePredictions,
    tweedie: Option<&TweedieModel>,
) -> Vec<ModelScore> {
    let mut scores = Vec::new();
    if let Some(two_part_pred) = predictions.two_part.as_ref() {
        scores.push(ModelScore {
            name: "two_part".to_string(),
            metrics: compute_model_fit_metrics(
                &input.outcome,
                &two_part_pred.expected_outcome,
                None,
            ),
        });
    }
    if let Some(pred) = predictions.two_part_elastic_net.as_ref() {
        scores.push(ModelScore {
            name: "two_part_elastic_net".to_string(),
            metrics: compute_model_fit_metrics(&input.outcome, &pred.expected_outcome, None),
        });
    }
    if let (Some(tweedie), Some(tweedie_pred)) = (tweedie, predictions.tweedie.as_ref()) {
        scores.push(ModelScore {
            name: format!("tweedie p={:.1}", tweedie.power),
            metrics: compute_model_fit_metrics(
                &input.outcome,
                &tweedie_pred.mean,
                Some(tweedie.power),
            ),
        });
    }
    if let Some(pred) = predictions.lognormal.as_ref() {
        scores.push(ModelScore {
            name: "lognormal".to_string(),
            metrics: compute_model_fit_metrics(&input.outcome, &pred.mean, None),
        });
    }
    scores
}

fn build_information_criteria(
    input: &ModelInput,
    predictions: &InSamplePredictions,
    tweedie: Option<&TweedieModel>,
    lognormal: Option<&LogNormalModel>,
) -> Vec<ModelInformationCriteria> {
    let mut rows = Vec::new();
    if let Some(two_part_pred) = predictions.two_part.as_ref() {
        let ll_two_part = two_part_log_likelihood(
            &input.outcome,
            &two_part_pred.prob_positive,
            &two_part_pred.mean_positive,
        );
        let ic_two_part = compute_information_criteria(
            ll_two_part,
            2 * input.design_matrix.ncols(),
            input.outcome.nrows(),
        );
        rows.push(ModelInformationCriteria {
            name: "two_part".to_string(),
            loglik: ic_two_part.loglik,
            aic: ic_two_part.aic,
            bic: ic_two_part.bic,
        });
    }
    if let Some(pred) = predictions.two_part_elastic_net.as_ref() {
        let ll = two_part_log_likelihood(&input.outcome, &pred.prob_positive, &pred.mean_positive);
        let ic = compute_information_criteria(
            ll,
            2 * input.design_matrix.ncols(),
            input.outcome.nrows(),
        );
        rows.push(ModelInformationCriteria {
            name: "two_part_elastic_net".to_string(),
            loglik: ic.loglik,
            aic: ic.aic,
            bic: ic.bic,
        });
    }
    if let (Some(tweedie), Some(tweedie_pred)) = (tweedie, predictions.tweedie.as_ref()) {
        let ll_tweedie =
            tweedie_quasi_log_likelihood(&input.outcome, &tweedie_pred.mean, tweedie.power);
        let ic_tweedie = compute_information_criteria(
            ll_tweedie,
            input.design_matrix.ncols(),
            input.outcome.nrows(),
        );
        rows.push(ModelInformationCriteria {
            name: format!("tweedie p={:.1}", tweedie.power),
            loglik: ic_tweedie.loglik,
            aic: ic_tweedie.aic,
            bic: ic_tweedie.bic,
        });
    }
    if let Some(lognormal) = lognormal {
        let ll = lognormal_log_likelihood(&input.design_matrix, &input.outcome, lognormal);
        if ll.is_finite() {
            let ic = compute_information_criteria(
                ll,
                input.design_matrix.ncols(),
                input.outcome.nrows(),
            );
            rows.push(ModelInformationCriteria {
                name: "lognormal".to_string(),
                loglik: ic.loglik,
                aic: ic.aic,
                bic: ic.bic,
            });
        }
    }
    rows
}

fn build_tweedie_candidate_ic(candidates: &[TweedieCandidate]) -> Vec<ModelInformationCriteria> {
    candidates
        .iter()
        .map(|candidate| ModelInformationCriteria {
            name: format!("{:.2}", candidate.power),
            loglik: candidate.information_criteria.loglik,
            aic: candidate.information_criteria.aic,
            bic: candidate.information_criteria.bic,
        })
        .collect()
}

fn build_cv_summary(
    cv_default: &CrossValidationResult,
    cv_elastic_net: Option<&CrossValidationResult>,
) -> Vec<ModelScore> {
    let mut rows = Vec::new();
    if cv_default.include_two_part {
        rows.push(ModelScore {
            name: "two_part".to_string(),
            metrics: cv_default.two_part_metrics.clone(),
        });
    }
    if let Some(elastic_net) = cv_elastic_net {
        rows.push(ModelScore {
            name: "two_part_elastic_net".to_string(),
            metrics: elastic_net.two_part_metrics.clone(),
        });
    }
    for candidate in &cv_default.tweedie_candidates {
        rows.push(ModelScore {
            name: format!("tweedie p={:.1}", candidate.power),
            metrics: candidate.metrics.clone(),
        });
    }
    if cv_default.include_lognormal
        && let Some(lognormal) = &cv_default.lognormal_metrics
    {
        rows.push(ModelScore {
            name: "lognormal".to_string(),
            metrics: lognormal.clone(),
        });
    }
    rows
}

fn build_cv_ranking(cv_summary: &[ModelScore]) -> Vec<ModelScore> {
    let mut rows = cv_summary.to_vec();
    rows.sort_by(|a, b| a.metrics.rmse.total_cmp(&b.metrics.rmse));
    rows
}

fn build_tweedie_cv_ranking(
    cv_default: &CrossValidationResult,
    tweedie_candidates: &[TweedieCandidate],
) -> Vec<TweedieRankingRow> {
    let mut candidates = cv_default.tweedie_candidates.clone();
    candidates.sort_by(|a, b| a.metrics.rmse.total_cmp(&b.metrics.rmse));
    candidates
        .into_iter()
        .map(|candidate| {
            let in_sample = tweedie_candidates
                .iter()
                .find(|entry| (entry.power - candidate.power).abs() < 1e-12);
            let (aic, bic) = in_sample.map_or((f64::NAN, f64::NAN), |entry| {
                (
                    entry.information_criteria.aic,
                    entry.information_criteria.bic,
                )
            });
            TweedieRankingRow {
                power: candidate.power,
                metrics: candidate.metrics,
                aic,
                bic,
            }
        })
        .collect()
}

fn make_table(headers: &[&str]) -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL_CONDENSED)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(headers.iter().map(|h| Cell::new(*h)).collect::<Vec<_>>());
    table
}

#[derive(Debug, Clone, Copy)]
struct MetricBest {
    rmse: f64,
    mae: f64,
    rmsle: f64,
    r2: f64,
    deviance: f64,
    aic: f64,
    bic: f64,
    loglik: f64,
}

fn best_metrics(rows: &[ModelScore]) -> MetricBest {
    MetricBest {
        rmse: rows
            .iter()
            .map(|s| s.metrics.rmse)
            .fold(f64::INFINITY, f64::min),
        mae: rows
            .iter()
            .map(|s| s.metrics.mae)
            .fold(f64::INFINITY, f64::min),
        rmsle: rows
            .iter()
            .map(|s| s.metrics.rmsle)
            .fold(f64::INFINITY, f64::min),
        r2: rows
            .iter()
            .map(|s| s.metrics.r2)
            .fold(f64::NEG_INFINITY, f64::max),
        deviance: rows
            .iter()
            .map(|s| s.metrics.deviance)
            .fold(f64::INFINITY, f64::min),
        aic: f64::INFINITY,
        bic: f64::INFINITY,
        loglik: f64::NEG_INFINITY,
    }
}

fn best_ic(rows: &[ModelInformationCriteria]) -> MetricBest {
    MetricBest {
        loglik: rows
            .iter()
            .map(|s| s.loglik)
            .fold(f64::NEG_INFINITY, f64::max),
        aic: rows.iter().map(|s| s.aic).fold(f64::INFINITY, f64::min),
        bic: rows.iter().map(|s| s.bic).fold(f64::INFINITY, f64::min),
        rmse: f64::INFINITY,
        mae: f64::INFINITY,
        rmsle: f64::INFINITY,
        r2: f64::NEG_INFINITY,
        deviance: f64::INFINITY,
    }
}

fn best_tweedie_ranking(rows: &[TweedieRankingRow]) -> MetricBest {
    let mut best = MetricBest {
        rmse: f64::INFINITY,
        mae: f64::INFINITY,
        rmsle: f64::INFINITY,
        r2: f64::NEG_INFINITY,
        deviance: f64::INFINITY,
        aic: f64::INFINITY,
        bic: f64::INFINITY,
        loglik: f64::NEG_INFINITY,
    };
    for row in rows {
        best.rmse = best.rmse.min(row.metrics.rmse);
        best.mae = best.mae.min(row.metrics.mae);
        best.rmsle = best.rmsle.min(row.metrics.rmsle);
        best.r2 = best.r2.max(row.metrics.r2);
        best.deviance = best.deviance.min(row.metrics.deviance);
        best.aic = best.aic.min(row.aic);
        best.bic = best.bic.min(row.bic);
    }
    best
}

fn add_metrics_row_highlight(
    table: &mut Table,
    label: &str,
    metrics: &ModelFitMetrics,
    best: &MetricBest,
) {
    table.add_row(vec![
        Cell::new(label),
        highlight_metric_cell(metrics.rmse, best.rmse, 4),
        highlight_metric_cell(metrics.mae, best.mae, 4),
        highlight_metric_cell(metrics.rmsle, best.rmsle, 4),
        highlight_metric_cell(metrics.r2, best.r2, 4),
    ]);
}

fn highlight_metric_cell(value: f64, best: f64, precision: usize) -> Cell {
    let is_best = (value - best).abs() < 1e-12;
    if is_best {
        Cell::new(format!("{value:.precision$}"))
            .fg(Color::Green)
            .add_attribute(Attribute::Bold)
    } else {
        Cell::new(format!("{value:.precision$}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::two_part::Regularization;
    use crate::utils::usize_to_f64;
    use faer::Mat;

    fn sample_input(n: usize) -> ModelInput {
        let design_matrix = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 25.0 },
        );
        let outcome = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.15f64.mul_add(usize_to_f64(i), 1.2)
            }
        });
        ModelInput::new(design_matrix, outcome)
    }

    #[test]
    fn compare_models_input_runs_and_renders_tables() {
        let input = sample_input(80);
        let report =
            compare_models_input(&input, &ModelComparisonOptions::default()).expect("comparison");
        assert!(!report.in_sample.is_empty());
        assert!(!report.information_criteria.is_empty());
        assert!(!report.cv_summary.is_empty());
        let tables = render_comparison_tables(&report);
        assert!(tables.in_sample.contains("model"));
        assert!(tables.information_criteria.contains("aic"));
        assert!(tables.cv_summary.contains("rmse"));
    }

    #[test]
    fn compare_models_input_excludes_lognormal_when_disabled() {
        let input = sample_input(70);
        let options = ModelComparisonOptions {
            include_lognormal: false,
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(report.in_sample.iter().all(|row| row.name != "lognormal"));
        assert!(
            report
                .information_criteria
                .iter()
                .all(|row| row.name != "lognormal")
        );
        assert!(report.cv_summary.iter().all(|row| row.name != "lognormal"));
        assert!(report.cv_ranking.iter().all(|row| row.name != "lognormal"));
    }

    #[test]
    fn compare_models_input_excludes_two_part_when_disabled() {
        let input = sample_input(70);
        let options = ModelComparisonOptions {
            include_two_part: false,
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(report.in_sample.iter().all(|row| row.name != "two_part"));
        assert!(
            report
                .information_criteria
                .iter()
                .all(|row| row.name != "two_part")
        );
        assert!(report.cv_summary.iter().all(|row| row.name != "two_part"));
        assert!(report.cv_ranking.iter().all(|row| row.name != "two_part"));
    }

    #[test]
    fn compare_models_input_handles_empty_tweedie_power_grid() {
        let input = sample_input(60);
        let options = ModelComparisonOptions {
            include_tweedie: false,
            tweedie_powers: Vec::new(),
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(report.tweedie_ic.is_empty());
        assert!(report.tweedie_cv_ranking.is_empty());
        assert!(
            report
                .in_sample
                .iter()
                .all(|row| !row.name.starts_with("tweedie"))
        );
    }

    #[test]
    fn compare_models_input_adds_elastic_net_two_part_results() {
        let input = sample_input(80);
        let options = ModelComparisonOptions {
            two_part_elastic_net_options: Some(FitOptions {
                regularization: Regularization::ElasticNet {
                    lambda: 0.05,
                    alpha: 0.5,
                    exclude_intercept: true,
                },
                ..FitOptions::default()
            }),
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(
            report
                .in_sample
                .iter()
                .any(|row| row.name == "two_part_elastic_net")
        );
        assert!(
            report
                .information_criteria
                .iter()
                .any(|row| row.name == "two_part_elastic_net")
        );
        assert!(
            report
                .cv_summary
                .iter()
                .any(|row| row.name == "two_part_elastic_net")
        );
    }

    #[test]
    fn render_comparison_tables_handles_degenerate_empty_report() {
        let report = ModelComparison {
            in_sample: Vec::new(),
            information_criteria: Vec::new(),
            cv_summary: Vec::new(),
            cv_ranking: Vec::new(),
            tweedie_ic: Vec::new(),
            tweedie_cv_ranking: Vec::new(),
        };
        let tables = render_comparison_tables(&report);
        assert!(!tables.in_sample.is_empty());
        assert!(!tables.information_criteria.is_empty());
        assert!(!tables.cv_summary.is_empty());
        assert!(!tables.tweedie_cv_ranking.is_empty());
    }
}
