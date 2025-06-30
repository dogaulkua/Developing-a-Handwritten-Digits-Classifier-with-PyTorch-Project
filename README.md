# 🧠 MNIST Handwritten Digit Recognition

A complete deep learning pipeline for handwritten digit classification using the **MNIST dataset** and **PyTorch**. This project implements a multi-layer feedforward neural network with dropout, batch normalization, and model persistence.

![MNIST Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

## 📌 Project Goals

- Build an image classification model for the MNIST dataset.
- Understand and implement data preprocessing techniques.
- Design a deep neural network with regularization techniques.
- Train and evaluate the model using PyTorch.
- Visualize results, tune performance, and save the trained model.

---

## 🗂️ Contents

- `data/` – Automatically downloaded MNIST dataset.
- `mnist_classifier_model.pth` – Saved PyTorch model (after training).
- Python script / notebook (this code) – Full training and evaluation pipeline.

---

## 🧰 Libraries Used

- `torch`, `torchvision`, `numpy`
- `matplotlib`, `collections`
- `torch.nn`, `torch.optim`

---

## 📥 Dataset

- **Source:** [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/)
- **Description:** 70,000 28x28 grayscale images of handwritten digits (0–9).
- **Split:**
  - 60,000 training samples
  - 10,000 test samples

---

## 🔄 Data Preprocessing

- Converted images to PyTorch tensors using `ToTensor()`.
- Applied normalization using MNIST mean = `0.1307`, std = `0.3081`.
- Data visualization using unnormalized samples.
- Visualized class distribution and sample images.

---

## 🧠 Model Architecture

| Layer Type        | Output Shape | Activation | Notes                  |
|------------------|--------------|------------|------------------------|
| Input Layer       | (784,)       | –          | Flattened 28x28 image  |
| Linear (fc1)      | 512          | ReLU       | Dropout + BatchNorm    |
| Linear (fc2)      | 256          | ReLU       | Dropout + BatchNorm    |
| Linear (fc3)      | 128          | ReLU       | Dropout                |
| Output (fc4)      | 10           | LogSoftmax | Multi-class prediction |

> Includes dropout for regularization and learning rate scheduling (StepLR).

---

## 🧪 Training & Evaluation

- **Loss Function:** `NLLLoss` (with LogSoftmax)
- **Optimizer:** `Adam` with weight decay and learning rate scheduler
- **Epochs:** 25–30
- **Batch Size:** 64
- **Training Accuracy:** Tracked and visualized per epoch
- **Test Accuracy:** Calculated overall + per class

### 🧾 Results

| Metric            | Value        |
|-------------------|--------------|
| Final Test Accuracy | **>90%** ✅ |
| Per-Class Accuracy  | Printed in output |
| Total Parameters    | ~500K+     |

---

## 📊 Visualizations

- Training Loss & Accuracy curves
- Dataset Sample Images
- Class Distribution Bar Chart
- Model Predictions (Correct / Incorrect Highlighted)

---

## 🔧 Hyperparameter Tuning

If model accuracy < 90%, an **improved network** is trained:
- Larger hidden layers
- Additional dropout and batch normalization
- Tuned learning rate schedule

---

## 💾 Saving the Model

Trained model is saved using `torch.save()`:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': model.__class__.__name__,
    'test_accuracy': test_accuracy
}, 'mnist_classifier_model.pth')
