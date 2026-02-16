//! Input preprocessing helpers for longitudinal MTP models.

use std::collections::BTreeMap;

use crate::input::LongitudinalModelInput;

use super::types::MtpError;

/// Subject-wise row index bundle.
#[derive(Debug, Clone)]
pub(crate) struct SubjectRows {
    pub subject_id: u64,
    pub rows: Vec<usize>,
}

/// Prepared input view reused by sampler components.
#[derive(Debug, Clone)]
pub(crate) struct MtpPreparedInput<'a> {
    pub input: &'a LongitudinalModelInput,
    pub subjects: Vec<SubjectRows>,
}

impl MtpPreparedInput<'_> {
    #[must_use]
    pub(crate) const fn n_subjects(&self) -> usize {
        self.subjects.len()
    }
}

/// # Errors
///
/// Returns `MtpError` if the longitudinal input is invalid.
pub(crate) fn prepare_input(
    input: &LongitudinalModelInput,
) -> Result<MtpPreparedInput<'_>, MtpError> {
    input.validate()?;

    let mut grouped: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (row, subject_id) in input.subject_ids.iter().copied().enumerate() {
        grouped.entry(subject_id).or_default().push(row);
    }

    let subjects = grouped
        .into_iter()
        .map(|(subject_id, rows)| SubjectRows { subject_id, rows })
        .collect();

    Ok(MtpPreparedInput { input, subjects })
}

#[cfg(test)]
mod tests {
    use faer::Mat;

    use super::*;

    #[test]
    fn prepare_input_groups_rows_by_subject() {
        let outcome = Mat::from_fn(4, 1, |_i, _| 1.0);
        let x_binary = Mat::from_fn(4, 1, |_i, _| 1.0);
        let x_mean = Mat::from_fn(4, 1, |_i, _| 1.0);

        let input = LongitudinalModelInput::new(
            outcome,
            x_binary,
            x_mean,
            vec![10, 11, 10, 11],
            vec![0.0, 0.0, 1.0, 1.0],
        );

        let prepared = prepare_input(&input).expect("input should be valid");
        assert_eq!(prepared.n_subjects(), 2);
        assert_eq!(prepared.subjects[0].subject_id, 10);
        assert_eq!(prepared.subjects[0].rows, vec![0, 2]);
    }
}
