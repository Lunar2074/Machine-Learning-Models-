# Machine Learning Models Portfolio

A collection of machine learning implementations created by Chad R. to demonstrate a deep understanding of core algorithms and modern deep learning frameworks. 

This repository highlights the ability to implement algorithms from scratch using NumPy to understand the underlying mathematics, as well as proficiency with production-level libraries like PyTorch and Scikit-Learn.

## Project Overview

### 1. Convolutional Neural Network (CIFAR-10)
**File:** `Convolutional Neural Network and Classification of CIFAR-10 Set.ipynb`

*   **Objective:** Classify images from the CIFAR-10 dataset using a custom deep learning architecture.
*   **Key Skills:** Deep Learning, PyTorch, Computer Vision, Model Optimization.
*   **Implementation Details:**
    *   Designed a custom CNN architecture (`CIFARNet`) utilizing Convolutional layers, Batch Normalization, ReLU activation, and Dropout for regularization.
    *   Implemented data augmentation pipelines (RandomCrop, RandomHorizontalFlip) to improve model generalization.
    *   Utilized CUDA for GPU acceleration to optimize training time.
    *   Achieved a **Top-1 Accuracy of 91.14%** through hyperparameter tuning (learning rate scheduling).

### 2. Optical Character Recognition (KNN & Naive Bayes)
**File:** `K-Nearest Neighbors & Gaussian Naive Bayes for Optical Character Recognition.ipynb`

*   **Objective:** Recognize handwritten digits (Optical Character Recognition) using statistical classification methods.
*   **Key Skills:** Algorithm Implementation from Scratch, NumPy, Statistical Modeling.
*   **Implementation Details:**
    *   **K-Nearest Neighbors (KNN):** Implemented the KNN algorithm manually using Euclidean distance calculations in NumPy (no pre-built classifiers). Analyzed model performance across differing `k` values.
    *   **Gaussian Naive Bayes:** Built a probabilistic classifier from scratch by calculating prior probabilities, means, and variances for each class to compute Gaussian likelihoods.
    *   **Evaluation:** Detailed error analysis using Confusion Matrices to identify common misclassifications (e.g., confusing 8s with 1s).

### 3. Regression Analysis (Linear, Lasso, Ridge)
**File:** `Linear Regression, Types of Regression, and the Progression of Diabetes within a Year.ipynb`

*   **Objective:** Predict disease progression using various regression techniques to understand bias-variance tradeoff.
*   **Key Skills:** Linear Algebra, Regularization (L1 & L2), Optimization, Gradient Descent.
*   **Implementation Details:**
    *   **Linear Regression:** Solved for coefficients using the closed-form Normal Equation (Ordinary Least Squares).
    *   **Lasso (L1) & Ridge (L2):** Implemented regularization manually to prevent overfitting.
    *   **Optimization:** Wrote custom Gradient Descent loops to minimize loss functions iteratively.
    *   **Visualization:** Plotted Ridge coefficients as a function of the regularization parameter (lambda) to visualize feature shrinkage.

### 4. Support Vector Machine (SVM)
**File:** `Support Vector Machine.ipynb`

*   **Objective:** Create a maximum margin classifier to separate linearly separable data.
*   **Key Skills:** convex optimization, Margin Maximization, Hinge Loss.
*   **Implementation Details:**
    *   Implemented a Linear SVM from scratch using Gradient Descent to optimize the hinge loss function.
    *   Visualized the decision boundary and the separating hyperplane relative to the support vectors.
    *   Demonstrated the mathematical concept of the "margin" by programmatically removing the closest support vector and retraining the model to observe the shift in the decision boundary.

## Technologies Used
*   **Languages:** Python
*   **Deep Learning:** PyTorch, Torchvision
*   **Scientific Computing:** NumPy, Pandas
*   **Visualization:** Matplotlib
*   **Utilities:** Scikit-Learn (for dataset loading and metrics)
