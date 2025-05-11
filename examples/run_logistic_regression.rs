use ndarray::array;
use rustwise::{
    common::traits::Model,
    logistic::model::{LogisticRegression, Regularization},
};

fn main() {
    // === Fake data ===
    // x = 4 samples, 2 features
    let x = array![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]];
    // y = binary labels (u8)
    let y = array![0, 0, 1, 1];

    // === Model setting ===
    let mut model = LogisticRegression::new(5.0, 1000, Regularization::default());

    let l2_reg = Regularization {
        l1: false,
        l2: true,
        lambda: 0.0001,
    };
    let mut model_l2reg = LogisticRegression::new(5.0, 1000, l2_reg);

    // === Training ===
    let (weights, bias, loss) = model.fit(&x, &y);

    // === Predictions ===
    //let y_pred = model.predict(&x, &weights, bias);

    println!("Learned weights: {:?}", weights);
    println!("Learned bias: {:?}", bias);
    println!("LogLoss: {:?}", loss);
    //println!("Prédictions: {:?}", y_pred);
    println!("_______________________________");
    let (weights2, bias2, loss2) = model_l2reg.fit(&x, &y);
    println!("PLearned weights: {:?}", weights2);
    println!("Learned bias: {:?}", bias2);
    println!("LogLoss: {:?}", loss2);
}
