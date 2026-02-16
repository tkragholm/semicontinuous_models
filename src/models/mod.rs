//! # Models
//!
//! Model implementations for semi-continuous, right-skewed semi-continuous outcome data.
//! Includes two-part models, Tweedie GLM, log-normal with smearing, and
//! a selection workflow to compare candidate fits.
//!
//! Also includes a modular scaffold for correlated marginalized two-part (MTP)
//! longitudinal modeling.

pub mod comparison;
pub mod lognormal;
pub mod matrix_ops;
pub mod mtp;
pub mod selection;
pub mod tweedie;
pub mod two_part;
