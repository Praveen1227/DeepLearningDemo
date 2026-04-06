# MNIST Digit Classification — Perceptron vs ANN vs CNN

Comparing three neural network architectures on the MNIST handwritten digit dataset, from a simple linear classifier up to a full CNN. The goal is to demonstrate how architectural decisions — hidden layers, spatial convolutions, regularization — directly translate into measurable accuracy gains on the same task.

---

## Results

| Model | Test Accuracy | Gain over Previous |
|---|---|---|
| Perceptron | 92.84% | — |
| ANN | 98.25% | +5.41% |
| CNN | **99.52%** | +1.27% |

---

## Dataset

- **Source:** [Kaggle — MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Train:** `mnist_train.csv` — 60,000 samples
- **Test:** `mnist_test.csv` — 10,000 samples
- **Input:** 28×28 grayscale images, flattened to 784 pixel values
- **Classes:** 10 (digits 0–9)
- **Preprocessing:** Pixel values normalized to [0, 1] by dividing by 255

---

## Models

- **Perceptron** — Flatten → Dense(10, softmax). Linear baseline, no hidden layers.
- **ANN** — Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, softmax). Adds non-linear feature learning.
- **CNN** — Two Conv2D + MaxPooling blocks → Dense(128, ReLU) → Dropout(0.5) → Dense(10, softmax). Exploits spatial structure for best performance.

All models trained with Adam optimizer, categorical crossentropy loss, 5 epochs, batch size 32.

---

## Visualizations

- **Side-by-side comparison** — 5 random test images predicted simultaneously by all three models.
- **CNN-only correct analysis** — Scans the test set to find cases where the CNN predicts correctly but both the Perceptron and ANN fail. Highlights the practical advantage of convolutional feature extraction on ambiguous or stylistically varied digits.

---

## Project Structure

```
mnist-cnn-comparison/
│
├── CNN.ipynb           # All models, training, visualizations, and CNN-only analysis
├── mnist_train.csv     # Training data (download separately)
├── mnist_test.csv      # Test data (download separately)
└── README.md
```

---

## Getting Started

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib
```

Download `mnist_train.csv` and `mnist_test.csv` from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv), place them in the root directory, then run:

```bash
jupyter notebook CNN.ipynb
```

---

## Stack
Python · TensorFlow/Keras · scikit-learn · NumPy · Pandas · Matplotlib

---

## License
This project is open-source.
