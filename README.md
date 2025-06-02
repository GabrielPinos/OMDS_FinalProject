# Optimization Methods for Data Science ⚙️  
**Final Project (2024–2025)**

---
## 📘 Project Overview

This repository contains the final project for the *Optimization Methods for Data Science* course. The project explores the application of optimization techniques in machine learning, focusing on two core tasks:

1. **Age Prediction using Multi-Layer Perceptron (MLP) with Regularized L2 Loss**  
2. **Gender and Ethnicity Classification using Support Vector Machines (SVMs)**

We employ real-world features extracted from facial images via a ResNet backbone, and optimize model parameters using custom routines (e.g., `scipy.optimize`, `CVXOPT`) without relying on automatic differentiation libraries such as PyTorch or TensorFlow.

---

## 📂 Project Structure

```bash
OMDS_Final_Project/
│
├── dataset/
│   ├── AGE REGRESSION.csv
│   ├── GENDER CLASSIFICATION.csv
│   └── ETHNICITY CLASSIFICATION.csv
│
├── src/
│   ├── mlp.py         # MLP utilities and L2 loss optimizer
│   ├── svm.py         # SVM optimization routines
│   └── ...                                       # other helper scripts
│
├── final_project.ipynb                           # Main notebook combining all parts
├── OMDS_exam_2025.pdf                            # Project assignment and specifications
├── Report_Pinos_Lattanzio.pdf                    # Final report of the assignment
└── README.md                                     # This file
```

---

## 📊 Datasets

All datasets are derived from the **UTKFace** dataset, containing facial features extracted via ResNet:

- **AGE REGRESSION.csv**:  
  - **Task**: Age prediction (regression)  
  - **Target**: Continuous values in \[0, 100\]  
- **GENDER CLASSIFICATION.csv**:  
  - **Task**: Binary classification (Male/Female)  
  - **Target**: 0 or 1  
- **ETHNICITY CLASSIFICATION.csv**:  
  - **Task**: Multiclass classification among 5 ethnicities (Bonus Task)  
  - **Target**: Integer from 0 to 4

Each file includes feature vectors (`feat i`) and a ground truth label (`gt`).

---

## 🧠 Theoretical Background

### Multi-Layer Perceptron (MLP)

The first part focuses on optimizing an L2-regularized loss function:

```math
E(\omega, \beta) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \lambda \sum_{l=1}^L \|\omega^{(l)}\|^2
```

We use `scipy.optimize.minimize()` to solve this minimization, tuning hyperparameters like:
- Number of layers (L)
- Hidden units per layer
- Activation function
- Regularization parameter λ

### Support Vector Machines (SVM)

The second part trains SVM classifiers using:
- **Gaussian (RBF)** and **Polynomial kernels**
- Dual SVM formulation solved via `CVXOPT`
- Most Violating Pair (MVP) decomposition for a subproblem of fixed size (q = 2)

Cross-validation is used to select optimal hyperparameters like kernel parameters and C.

---

## 📑 Report

The full technical report is available in `Report_Pinos_Lattanzio.pdf`. It includes:
- Description of the models and methods
- Optimization details
- Cross-validation and performance metrics
- Comparison between different models

---
## 🔗 References
**Author**: Gabriel Pinos, Federico Lattanzio

**Email**: [pinos.1965035@studenti.uniroma1.it], [lattanzio.1886519@studenti.uniroma1.it]

**Course**: Optimization Methods for Data Science

MSc. in Data Science, Sapienza University of Rome
