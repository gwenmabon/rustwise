use ndarray::{Array1, Array2};

use crate::{
    common::traits::Model,
    logistic::utils::{predict_proba, update_weights},
};

#[derive(Default, Clone)]
pub struct Regularization {
    pub l1: bool,
    pub l2: bool,
    pub lambda: f64,
}
pub fn compute_gradients(
    x: &Array2<f64>,
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    weights: &Array1<f64>,
    reg: Regularization,
) -> (Array1<f64>, f64) {
    let n_sample = x.shape()[0] as f64;
    let error = y_pred - y_true;
    let mut dw = (x.t().dot(&error)) / n_sample;
    let db = error.sum() / n_sample;
    if reg.l1 {
        dw = dw + (reg.lambda / n_sample) * weights.mapv(|w| w.signum())
    } else if reg.l2 {
        dw = dw + (reg.lambda / n_sample) * weights
    }

    (dw, db)
}

pub fn log_loss(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    weights: &Array1<f64>,
    reg: Regularization,
) -> f64 {
    let n = y_true.len() as f64;
    let eps = 1e-15; // pour éviter log(0)

    let base_loss = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, p)| {
            let p = p.clamp(eps, 1.0 - eps);
            y * p.ln() + (1.0 - y) * (1.0 - p).ln()
        })
        .sum::<f64>()
        / -n;

    let l2_penalty = if reg.l2 {
        reg.lambda / (2.0 * n) * weights.iter().map(|w| w * w).sum::<f64>()
    } else {
        0.0
    };

    let l1_penalty = if reg.l1 {
        reg.lambda / n * weights.iter().map(|w| w.abs()).sum::<f64>()
    } else {
        0.0
    };

    base_loss + l1_penalty + l2_penalty
}

pub struct LogisticRegression {
    pub learning_rate: f64,
    pub n_iters: usize,
    pub reg: Regularization,
    weights: Array1<f64>,
    bias: f64,
}

impl LogisticRegression {
    pub fn new(learning_rate: f64, n_iters: usize, reg: Regularization) -> Self {
        Self {
            learning_rate,
            n_iters,
            reg,
            weights: Array1::<f64>::zeros(0),
            bias: 0.0,
        }
    }

    fn init_weights(&mut self, n_features: usize) {
        if self.weights.len() != n_features {
            self.weights = Array1::<f64>::zeros(n_features);
            self.bias = 0.0;
        }
    }
}

impl Model for LogisticRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<u8>) -> (Array1<f64>, f64, f64) {
        println!("Fitting Logistic Regression model...");
        let y_true = y.mapv(|v| v as f64);
        let n_features = x.shape()[1];

        self.init_weights(n_features);

        let mut weights = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0;
        let mut loss = 0.0;

        for _ in 0..self.n_iters {
            let y_pred = predict_proba(x, &weights, bias);
            loss = log_loss(&y_true, &y_pred, &weights, self.reg.clone());
            //println!("Loss: {}", loss);

            let (dw, db) = compute_gradients(x, &y_true, &y_pred, &weights, self.reg.clone());
            let (new_weights, new_bias) =
                update_weights(&weights, bias, dw, db, self.learning_rate);
            weights = new_weights;
            bias = new_bias;
        }
        self.weights = weights.clone();
        self.bias = bias;
        (weights, bias, loss)
    }

    fn predict(&self, x: &Array2<f64>) -> Vec<u8> {
        println!("Predicting with Logistic Regression model...");
        let y_prob = predict_proba(x, &self.weights, self.bias);
        y_prob.mapv(|p| if p > 0.5 { 1 } else { 0 }).to_vec()
    }
}
