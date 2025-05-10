# RustWise 🦀📚

![Rust](https://github.com/gwenmabob/rustwise/actions/workflows/ci.yml/badge.svg)

> A lightweight collection of machine learning models implemented in pure Rust.

RustWise is an educational and experimental library for implementing classic machine learning models in Rust.  
The goal is to provide a clean, minimal foundation for learning, extending, and experimenting with ML in Rust.

---

## 🚀 Features

- 📈 Logistic Regression
- 🔁 Bayesian Logistic Regression (based on Chapelle's paper)
- 🧠 Perceptron
- 🐦 Naive Bayes
- 🧰 Shared tools: metrics, traits, utilities

---

## 🗂 Project Structure
```bash
rustwise/
├── Cargo.toml
├── src/
│   ├── common/               # Shared utilities and traits
│   ├── logistic/             # Classic logistic regression
│   ├── bayesian_logistic/    # Bayesian logistic regression
│   ├── perceptron/           # Perceptron model
│   └── naive_bayes/          # Naive Bayes classifier
├── examples/                 # Example training scripts
└── tests/                    # Integration tests
```



---

## 🛠 Usage

### Build

```bash
make build
```
```bash
make test
```
```bash
make run example=run_logistic
```

## 💡 Contributing

RustWise is an open playground — feel free to contribute improvements, new models, or refactorings.
If you're learning Rust or ML, this repo is a great starting point!

## 📚 References

Chapelle, O. (2009). Semi-supervised learning with Bayesian logistic regression.
Elements of Statistical Learning, Hastie et al.
🐛 License

MIT License.
Feel free to use and modify — just mention the source if it helps you!