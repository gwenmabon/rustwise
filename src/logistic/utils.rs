use ndarray::{Array1, Array2};

use crate::common::metrics::sigmoid;

pub fn initialize_weights(n_features: usize) -> (Array1<f64>, f64) {
    let weights: Array1<f64> = Array1::from_elem(n_features, 0.0);
    let bias: f64 = 0.0;
    (weights, bias)
}

pub fn predict_proba(X: &Array2<f64>, weights: &Array1<f64>, bias: f64) -> Array1<f64> {
    if X.len() != weights.len() {
        panic!(
            "Dimension mismatch: X has length {}, weights has length {}",
            X.len(),
            weights.len()
        );
    }
    let z= X.dot(weights) + bias;
    let y_pred = z.mapv(|v| sigmoid(v));
    y_pred
}

pub fn log_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let term1 = y_true * &y_pred.mapv(|x| x.ln());              // y_true * log(y_pred)
    let term2 = (1.0 - y_true) * &(1.0 - y_pred).mapv(|x| x.ln()); // (1 - y_true) * log(1 - y_pred)
    
    let loss = -(term1 + term2).sum() / y_true.len() as f64; 
    loss
}

pub fn compute_gradients(X: &Array2<f64>, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> (Array1<f64>,f64){
    let n_sample = X.shape()[0] as f64;
    let error = y_pred - y_true;
    let dw=(X.t().dot(&error))/n_sample;
    let db = error.sum() / n_sample;
    (dw, db)
}

pub fn update_weights(weights: &Array1<f64>, bias: f64, dw: Array1<f64>, db: f64, learning_rate: f64) -> (Array1<f64>,f64) {
    let weights = weights - learning_rate *dw;
    let bias=bias -learning_rate*db;
    (weights, bias)
}

