use ndarray::{array, Array1, Array2};
use rustwise::logistic::model::LogisticRegression;

fn main() {
    // === Données factices ===
    // X = 4 échantillons, 2 features
    let X = array![[0.0, 1.0],
                   [1.0, 1.0],
                   [2.0, 1.0],
                   [3.0, 1.0]];
    // y = labels binaires (u8)
    let y = array![0, 0, 1, 1];

    // === Création du modèle ===
    let mut model = LogisticRegression {
        weights: Array1::zeros(2),
        bias: 0.0,
        learning_rate: 0.1,
        n_iters: 1000,
    };

    // === Entraînement ===
    let (weights, bias) = model.fit(&X, &y);

    // === Prédictions ===
    let y_pred = predict(&X, &weights, bias);

    println!("Poids appris: {:?}", weights);
    println!("Biais appris: {:?}", bias);
    println!("Prédictions: {:?}", y_pred);
}
