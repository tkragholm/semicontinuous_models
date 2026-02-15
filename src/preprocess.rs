use std::collections::HashSet;

use faer::Mat;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutcomeDiagnostics {
    pub n_rows: usize,
    pub n_finite: usize,
    pub n_non_finite: usize,
    pub n_negative: usize,
    pub n_zero: usize,
    pub n_positive: usize,
    pub positive_share: f64,
    pub zero_share: f64,
}

#[must_use]
pub fn outcome_diagnostics(outcome: &Mat<f64>) -> OutcomeDiagnostics {
    let n_rows = outcome.nrows();
    let mut n_finite = 0usize;
    let mut n_negative = 0usize;
    let mut n_zero = 0usize;
    let mut n_positive = 0usize;

    for row in 0..n_rows {
        let value = outcome[(row, 0)];
        if !value.is_finite() {
            continue;
        }
        n_finite += 1;
        if value < 0.0 {
            n_negative += 1;
        } else if value == 0.0 {
            n_zero += 1;
        } else {
            n_positive += 1;
        }
    }

    let n_non_finite = n_rows.saturating_sub(n_finite);
    let positive_share = if n_finite > 0 {
        usize_to_f64(n_positive) / usize_to_f64(n_finite)
    } else {
        0.0
    };
    let zero_share = if n_finite > 0 {
        usize_to_f64(n_zero) / usize_to_f64(n_finite)
    } else {
        0.0
    };

    OutcomeDiagnostics {
        n_rows,
        n_finite,
        n_non_finite,
        n_negative,
        n_zero,
        n_positive,
        positive_share,
        zero_share,
    }
}

#[must_use]
pub fn column_has_variation(x: &Mat<f64>, column: usize, tolerance: f64) -> bool {
    if column >= x.ncols() || x.nrows() < 2 {
        return false;
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for row in 0..x.nrows() {
        let value = x[(row, column)];
        min = min.min(value);
        max = max.max(value);
    }
    (max - min).abs() > tolerance.abs()
}

#[must_use]
fn nonconstant_column_indices(x: &Mat<f64>, tolerance: f64, always_keep: &[usize]) -> Vec<usize> {
    let forced = always_keep
        .iter()
        .copied()
        .filter(|idx| *idx < x.ncols())
        .collect::<HashSet<_>>();
    let mut cols = Vec::new();
    for col in 0..x.ncols() {
        if forced.contains(&col) || column_has_variation(x, col, tolerance) {
            cols.push(col);
        }
    }
    cols
}

#[must_use]
fn select_columns(x: &Mat<f64>, columns: &[usize]) -> Mat<f64> {
    Mat::from_fn(x.nrows(), columns.len(), |row, col| x[(row, columns[col])])
}

#[must_use]
pub fn drop_constant_columns(
    x: &Mat<f64>,
    tolerance: f64,
    always_keep: &[usize],
) -> (Mat<f64>, Vec<usize>) {
    let kept_columns = nonconstant_column_indices(x, tolerance, always_keep);
    (select_columns(x, &kept_columns), kept_columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_diagnostics_counts_values() {
        let y = Mat::from_fn(5, 1, |row, _| match row {
            0 => -1.0,
            1 => 0.0,
            2 => 2.0,
            3 => 3.0,
            _ => f64::NAN,
        });
        let diag = outcome_diagnostics(&y);
        assert_eq!(diag.n_rows, 5);
        assert_eq!(diag.n_finite, 4);
        assert_eq!(diag.n_non_finite, 1);
        assert_eq!(diag.n_negative, 1);
        assert_eq!(diag.n_zero, 1);
        assert_eq!(diag.n_positive, 2);
        assert!((diag.positive_share - 0.5).abs() < 1e-12);
        assert!((diag.zero_share - 0.25).abs() < 1e-12);
    }

    #[test]
    fn drop_constant_columns_keeps_forced_and_variable_columns() {
        let x = Mat::from_fn(4, 4, |row, col| match col {
            0 => 1.0,
            1 => {
                if row < 2 {
                    0.0
                } else {
                    1.0
                }
            }
            2 => 3.0,
            _ => usize_to_f64(row),
        });

        let (trimmed, kept) = drop_constant_columns(&x, 1e-12, &[0, 1]);
        assert_eq!(kept, vec![0, 1, 3]);
        assert_eq!(trimmed.ncols(), 3);
        assert!((trimmed[(3, 2)] - 3.0).abs() < 1e-12);
    }
}
