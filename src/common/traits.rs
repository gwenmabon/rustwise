use ndarray::{Array1, Array2};

pub trait Model {
    fn fit(&mut self, X: &Array2<f64>, y: &Array1<u8>) -> (Array1<f64>, f64) ;
    fn predict(&self, X: &Array2<f64>) -> Vec<u8>;
}
