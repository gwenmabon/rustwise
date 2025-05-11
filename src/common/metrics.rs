// Common metrics like accuracy, precision, etc.

pub fn accuracy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| (*a > &0.5) == (*b > &0.5))
        .count();
    correct as f32 / y_true.len() as f32
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}
pub fn sigmoid_derivative(z: f64) -> f64 {
    let sig = sigmoid(z);
    sig * (1.0 - sig)
}
