# 🧠 Microstructure-Predictions

This repository contains the code and resources for predicting the effective Young’s modulus (`E_eff`) of binary microstructure images using a convolutional neural network (CNN).

## 📂 Repository Structure

.
├── data/
│   ├── models/
│   ├── npy_images/
│   └── properties/
│       └── material_property.csv
├── notebooks/
│   ├── run_tests.ipynb
│   └── train_colab.ipynb
├── src/
│   ├── image_loader.py
│   ├── inference.py
│   ├── model.py
│   ├── pipeline.py
│   ├── targets_loader.py
│   └── train.py
├── .gitignore
├── README.md
├── requirements.txt
└── train_model.ipynb


## 📊 Problem Overview

The task is to train a neural network model to predict effective Young's modulus values from microstructure images. These images represent binary composites, and the regression target (\(E_{\text{eff}}\)) is a scalar in units of Pa.

## 🛠️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/ianmm2d/Microstructure-Predictions.git
   cd Microstructure-Predictions
    ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebook**
   Open `train_model.ipynb` and run all cells to preprocess data, train the model, and evaluate performance.

## 🔢 Normalization Details

The target values ($E_{\text{eff}}$) were normalized using the mean and standard deviation of the training set only:

* $y_{\text{mean}} = 2.7590 \times 10^9$
* $y_{\text{std}} = 1.0724 \times 10^9$

These values are saved together with the model in `model_weights.pth`, so inference can be performed without requiring access to the original training set.

## 🧠 Model Architecture

* Convolutional Layers: 2 layers (32 and 64 filters, kernel size 3)
* Activation: ReLU
* Pooling: Max pooling after each conv layer
* Fully Connected: One FC layer with 128 neurons (ReLU), followed by final regression output
* Optimizer: Adam (LR: $1 \times 10^{-4}$, weight decay: $1 \times 10^{-5}$)
* Loss: Mean Squared Error (MSE)

## 📈 Results

The dataset was split into:

* 8% Training
* 2% Validation
* 90% Testing

### 📊 Prediction Accuracy

| Set        | Mean Relative Error |
| ---------- | ------------------- |
| Training   | 0.6146 %            |
| Validation | 2.7316 %            |
| Test       | 2.5396 %            |

## 🔍 Example Predictions

| Set            | True $E_{\text{eff}}$ $[10^9]$ | Predicted $[10^9]$ | Relative Error \[%] |
| -------------- | ------------------------------ | ------------------ | ------------------- |
| **Training**   | 2.149                          | 2.190              | 1.87                |
|                | 1.747                          | 1.728              | 1.12                |
|                | 2.329                          | 2.325              | 0.20                |
| **Validation** | 3.997                          | 3.875              | 3.05                |
|                | 2.465                          | 2.535              | 2.88                |
|                | 4.398                          | 4.276              | 2.81                |
| **Test**       | 1.588                          | 1.642              | 3.39                |
|                | 2.177                          | 2.192              | 0.67                |
|                | 1.594                          | 1.629              | 2.23                |
|                | 4.123                          | 4.085              | 0.91                |

## ✅ Notes

* The model and normalization statistics are saved together in `data/models/model_weights.pth`.
* All training, evaluation, and prediction code is in `train_model.ipynb`.
* To predict on new images, ensure they are preprocessed as described in the notebook.

## 📎 License

This project is part of an academic assignment and is intended for educational use only.