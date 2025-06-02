"""
SVM Utility Functions
This module contains all utility functions for implementing SVM with dual quadratic optimization.
"""

import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import itertools


def kernel(x, y, params):
    """
    Compute kernel function between two vectors.
    
    Args:
        x (np.array): First vector of shape (n_features,)
        y (np.array): Second vector of shape (n_features,)
        params (dict): Kernel parameters including 'type' and kernel-specific params
    
    Returns:
        float: Kernel value between x and y
    """
    if params['type'] == 'rbf':
        gamma = params.get('gamma', 1.0)
        return np.exp(-gamma * np.linalg.norm(x - y)**2)
    
    elif params['type'] == 'poly':
        p = params.get('p', 2)
        return (np.dot(x, y) + 1) ** p
    
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf' or 'poly'.")


def build_dual_problem(X, y, kernel_params, C=1.0):
    """
    Build the dual quadratic optimization problem for SVM.
    
    Args:
        X (np.array): Training data of shape (n_samples, n_features)
        y (np.array): Training labels of shape (n_samples,)
        kernel_params (dict): Kernel parameters
        C (float): Regularization parameter
    
    Returns:
        tuple: (P, q, G, h, A, b) matrices for CVXOPT quadratic solver
    """
    N = X.shape[0]
    
    # Build kernel matrix K
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j], kernel_params)
    
    # Quadratic form matrix P = y_i * y_j * K(x_i, x_j)
    P = matrix(np.outer(y, y) * K)
    
    # Linear term q = -1 (maximize sum of lambdas)
    q = matrix(-np.ones(N))

    # Inequality constraints: 0 <= lambda_i <= C
    G_std = np.vstack((-np.eye(N), np.eye(N)))
    h_std = np.hstack((np.zeros(N), np.ones(N) * C))
    G = matrix(G_std)
    h = matrix(h_std)

    # Equality constraint: sum(lambda_i * y_i) = 0
    A = matrix(y.astype(np.double), (1, N))
    b = matrix(0.0)

    return P, q, G, h, A, b


def solve_svm_dual(P, q, G, h, A, b):
    """
    Solve the SVM dual quadratic problem using CVXOPT.
    
    Args:
        P, q, G, h, A, b: Matrices defining the quadratic problem
    
    Returns:
        tuple: (lambdas, solution) where lambdas are Lagrange multipliers
    """
    # Disable CVXOPT verbose output
    solvers.options['show_progress'] = False
    
    # Solve the quadratic problem
    solution = solvers.qp(P, q, G, h, A, b)
    
    # Extract optimal lambda values (Lagrange multipliers)
    lambdas = np.ravel(solution['x'])
    
    return lambdas, solution


def compute_bias(X, y, lambdas, kernel_params, C, tol=1e-5):
    """
    Compute the bias term for SVM using support vectors.
    
    Args:
        X (np.array): Training data
        y (np.array): Training labels
        lambdas (np.array): Lagrange multipliers
        kernel_params (dict): Kernel parameters
        C (float): Regularization parameter
        tol (float): Tolerance for numerical precision
    
    Returns:
        tuple: (bias, support_vector_indices)
    """
    # Support vectors: 0 < lambda_i < C
    sv_mask = (lambdas > tol) & (lambdas < C - tol)
    sv_indices = np.where(sv_mask)[0]

    if len(sv_indices) == 0:
        raise ValueError("No support vectors found.")

    # Use first support vector to calculate bias
    idx = sv_indices[0]
    x_k = X[idx]
    y_k = y[idx]

    # Calculate sum term for bias computation
    sum_term = 0
    for i in range(len(X)):
        if lambdas[i] > tol:
            sum_term += lambdas[i] * y[i] * kernel(X[i], x_k, kernel_params)

    bias = y_k - sum_term
    return bias, sv_indices


def predict(X_train, y_train, lambdas, bias, kernel_params, X_test, tol=1e-5):
    """
    Make predictions using trained SVM model.
    
    Args:
        X_train (np.array): Training data
        y_train (np.array): Training labels
        lambdas (np.array): Lagrange multipliers
        bias (float): Bias term
        kernel_params (dict): Kernel parameters
        X_test (np.array): Test data
        tol (float): Tolerance for numerical precision
    
    Returns:
        np.array: Predicted labels for test data
    """
    # Use only support vectors (lambda > tolerance)
    sv = lambdas > tol
    X_sv = X_train[sv]
    y_sv = y_train[sv]
    lambda_sv = lambdas[sv]

    y_pred = []

    # Make prediction for each test sample
    for x in X_test:
        sum_term = 0
        for i in range(len(X_sv)):
            sum_term += lambda_sv[i] * y_sv[i] * kernel(X_sv[i], x, kernel_params)
        y_pred.append(np.sign(sum_term + bias))

    return np.array(y_pred)


def cross_validate_svm(X, y, C_values, gamma_values, k=5):
    """
    Perform k-fold cross validation to find optimal hyperparameters.
    
    Args:
        X (np.array): Training data
        y (np.array): Training labels
        C_values (list): List of C values to test
        gamma_values (list): List of gamma values to test
        k (int): Number of folds for cross validation
    
    Returns:
        list: Results sorted by accuracy (best first)
    """
    results = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Test all combinations of C and gamma
    for C, gamma in itertools.product(C_values, gamma_values):
        accs = []

        # Perform k-fold cross validation
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                # Build and solve SVM dual problem
                P, q, G, h, A, b_eq = build_dual_problem(
                    X_train, y_train, {'type': 'rbf', 'gamma': gamma}, C
                )
                lambdas, _ = solve_svm_dual(P, q, G, h, A, b_eq)
                
                # Compute bias and make predictions
                bias, _ = compute_bias(
                    X_train, y_train, lambdas, {'type': 'rbf', 'gamma': gamma}, C
                )
                y_pred = predict(
                    X_train, y_train, lambdas, bias, 
                    {'type': 'rbf', 'gamma': gamma}, X_val
                )

                # Calculate accuracy
                acc = accuracy_score(y_val, y_pred)
                accs.append(acc)
                
            except Exception as e:
                print(f"Error with C={C}, gamma={gamma}: {e}")
                accs.append(0)

        # Store average accuracy for this parameter combination
        avg_acc = np.mean(accs)
        results.append((C, gamma, avg_acc))
        print(f"C={C}, gamma={gamma}, CV accuracy={avg_acc:.4f}")

    # Sort by accuracy (best first)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def train_final_svm(X, y, kernel_params, C):
    """
    Train final SVM model with given parameters.
    
    Args:
        X (np.array): Training data
        y (np.array): Training labels  
        kernel_params (dict): Kernel parameters
        C (float): Regularization parameter
    
    Returns:
        tuple: (lambdas, bias, solution_info)
    """
    # Build and solve dual problem
    P, q, G, h, A, b_eq = build_dual_problem(X, y, kernel_params, C)
    lambdas, solution = solve_svm_dual(P, q, G, h, A, b_eq)
    
    # Compute bias
    bias, sv_indices = compute_bias(X, y, lambdas, kernel_params, C)
    
    return lambdas, bias, solution