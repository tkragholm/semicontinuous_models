use faer::Mat;

#[must_use]
pub fn select_rows(matrix: &Mat<f64>, indices: &[usize]) -> Mat<f64> {
    Mat::from_fn(indices.len(), matrix.ncols(), |i, j| {
        matrix[(indices[i], j)]
    })
}

#[must_use]
pub fn select_values(vector: &Mat<f64>, indices: &[usize]) -> Mat<f64> {
    Mat::from_fn(indices.len(), 1, |i, _| vector[(indices[i], 0)])
}

#[must_use]
pub fn map_mat(values: &Mat<f64>, f: impl Fn(f64) -> f64) -> Mat<f64> {
    Mat::from_fn(values.nrows(), values.ncols(), |i, j| f(values[(i, j)]))
}
