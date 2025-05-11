use ndarray::{Array1, Array2};

use crate::common::metrics::sigmoid;

pub fn initialize_weights(n_features: usize) -> (Array1<f64>, f64) {
    let weights: Array1<f64> = Array1::from_elem(n_features, 0.0);
    let bias: f64 = 0.0;
    (weights, bias)
}

pub fn predict_proba(x: &Array2<f64>, weights: &Array1<f64>, bias: f64) -> Array1<f64> {
    if x.shape()[1] != weights.len() {
        panic!(
            "Dimension mismatch: x has length {}, weights has length {}",
            x.len(),
            weights.len()
        );
    }
    let z = x.dot(weights) + bias;
    z.mapv(sigmoid)
}

pub fn update_weights(
    weights: &Array1<f64>,
    bias: f64,
    dw: Array1<f64>,
    db: f64,
    learning_rate: f64,
) -> (Array1<f64>, f64) {
    let weights = weights - learning_rate * dw;
    let bias = bias - learning_rate * db;
    (weights, bias)
}
