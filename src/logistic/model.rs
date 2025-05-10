use ndarray::{Array1, Array2};

use crate::{common::traits::Model, logistic::utils::{compute_gradients, initialize_weights, log_loss, predict_proba, update_weights}};

pub struct LogisticRegression {
    weights: Vec<f64>, 
    bias: f64,
    learning_rate: f64,
    pub n_iters: usize,
}

impl LogisticRegression {
    // Constructeur pour LogisticRegression
    pub fn new(learning_rate: f64, n_iters: usize) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            n_iters,
        }
    }
}
impl Model for LogisticRegression {
    fn fit(&mut self, X: &Array2<f64>, y: &Array1<u8>) -> (Array1<f64>, f64) {
        println!("Fitting Logistic Regression model...");
        let y_true = y.mapv(|v| v as f64);
        let n_features= X.shape()[1];

        let mut weights = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0;

        for _ in 0..self.n_iters {
            let y_pred = predict_proba(X, &weights, bias);
            let loss = log_loss(&y_true, &y_pred);
            let (dw, db)= compute_gradients(X, &y_true, &y_pred);
            let (new_weights, new_bias) = update_weights(&weights, bias, dw, db, self.learning_rate);
            weights = new_weights;
            bias = new_bias;
        }
        (weights, bias)
    }

    fn predict(&self, X: &Array2<f64>) -> Vec<u8> {
        println!("Predicting with Logistic Regression model...");
        vec![]
        // function predict(X, weights, bias):
//     y_prob = predict_proba(X, weights, bias)
//     return [1 if p > 0.5 else 0 for p in y_prob]
    }
}
