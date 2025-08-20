import numpy as np
from sklearn.metrics import confusion_matrix , accuracy_score
from cvxopt import matrix, solvers
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def linear_kernel(X1, X2):
    """
    Computes the linear kernel K(x, x') = x · x'
    
    Args:
        X1: shape (n_samples_1, n_features)
        X2: shape (n_samples_2, n_features)

    Returns:
        Kernel matrix of shape (n_samples_1, n_samples_2)
    """
    return np.dot(X1, X2.T)


def polynomial_kernel(X1, X2, p=3):
    """
    Computes the polynomial kernel K(x, x') = (x · x' + 1)^p
    
    Args:
        X1: shape (n_samples_1, n_features)
        X2: shape (n_samples_2, n_features)
        p: degree of the polynomial

    Returns:
        Kernel matrix of shape (n_samples_1, n_samples_2)
    """
    return (np.dot(X1, X2.T) + 1) ** p


def rbf_kernel(X1, X2, gamma=0.1):
    """
    Computes the RBF (Gaussian) kernel:
    K(x, x') = exp(-gamma ||x - x'||^2)

    Args:
        X1: shape (n_samples_1, n_features)
        X2: shape (n_samples_2, n_features)
        gamma: width of the Gaussian (scalar)

    Returns:
        Kernel matrix of shape (n_samples_1, n_samples_2)
    """
    # Expand dimensions for broadcasting
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    sq_dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dist)

def compute_kernel_matrix(X1, X2, kernel_function):
    """
    Computes the kernel matrix between two datasets using a specified kernel function.

    Args:
        X1: shape (n_samples_1, n_features)
        X2: shape (n_samples_2, n_features)
        kernel_function: callable (e.g., linear_kernel, rbf_kernel, etc.)

    Returns:
        Kernel matrix K of shape (n_samples_1, n_samples_2)
    """
    return kernel_function(X1, X2)



def train_svm_dual_cvxopt(X, y, kernel, C=1.0, tol=1e-5):
    n_samples = X.shape[0]
    y = y.astype(float)

    K = compute_kernel_matrix(X, X, kernel)
    P_np = (np.outer(y, y) * K).astype('double')
    P_np = 0.5 * (P_np + P_np.T)
    q_np = (-np.ones(n_samples)).astype('double')

    G_std = (-np.eye(n_samples)).astype('double')
    h_std = (np.zeros(n_samples)).astype('double')
    G_slack = (np.eye(n_samples)).astype('double')
    h_slack = (C * np.ones(n_samples)).astype('double')

    G_np = np.vstack((G_std, G_slack))
    h_np = np.hstack((h_std, h_slack))

    A_np = y.reshape(1, -1).astype('double')
    b_np = np.array([0.0], dtype='double')

    P, q = matrix(P_np), matrix(q_np)
    G, h = matrix(G_np), matrix(h_np)
    A, b_cvx = matrix(A_np), matrix(b_np)

    solvers.options['show_progress'] = False
    t0 = time.perf_counter()
    result = solvers.qp(P, q, G, h, A, b_cvx)
    opt_time = time.perf_counter() - t0

    alpha = np.ravel(result['x'])
    alpha = np.clip(alpha, 0.0, C)

    obj = result['primal objective']
    n_iter = result['iterations']
    status = result['status']

    sv_idx = np.where(alpha > tol)[0]
    free_idx = np.where((alpha > tol) & (alpha < C - tol))[0]
    idx_for_b = free_idx if len(free_idx) > 0 else sv_idx

    b = np.mean([
        y[i] - np.sum(alpha * y * K[i])
        for i in idx_for_b
    ])

    return alpha, sv_idx, b, obj, n_iter, status, opt_time




def predict_svm_dual(X_train, y_train, X_test, alpha, b, kernel, tol=1e-5):
    """
    Predict labels for test points using the trained SVM dual solution.

    Args:
        X_train: training data (used to define support vectors)
        y_train: training labels
        X_test: test data
        alpha: vector of Lagrange multipliers
        b: bias term
        kernel: kernel function
        tol: threshold for selecting support vectors (usually same as in training)

    Returns:
        predictions: array of predicted labels (+1 or -1)
    """
    support_indices = np.where(alpha > tol)[0]
    alpha_sv = alpha[support_indices]
    y_sv = y_train[support_indices]
    X_sv = X_train[support_indices]

    K = compute_kernel_matrix(X_test, X_sv, kernel)
    decision = K @ (alpha_sv * y_sv) + b
    pred = np.sign(decision)
    pred[pred == 0] = 1
    return pred



def cross_validate_svm_cvxopt(X, y, C_values, gamma_values, k_folds=5):
    """
    Cross-validation using CVXOPT for RBF kernel SVM.

    Returns:
        best_config: tuple (C, gamma, avg_val_accuracy)
        all_results: list of tuples with (C, gamma, avg_val_accuracy)
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_results = []

    for C in C_values:
        for gamma in gamma_values:
            accs = []
            for tr_idx, va_idx in skf.split(X, y):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)

                kernel = lambda A, B: rbf_kernel(A, B, gamma=gamma)
                alpha, sv_idx, b, obj, iters, status, _ = train_svm_dual_cvxopt(
                    X_tr, y_tr, kernel, C
                )
                y_pred = predict_svm_dual(X_tr, y_tr, X_va, alpha, b, kernel)
                accs.append(accuracy_score(y_va, y_pred))

            avg_acc = float(np.mean(accs))
            print(f"C={C}, gamma={gamma} → avg val acc: {avg_acc:.4f}")
            all_results.append((C, gamma, avg_acc))

    best_config = max(all_results, key=lambda x: x[2])
    return best_config, all_results



def evaluate_svm_all(X_train, y_train, X_test, y_test, alpha, b, kernel, threshold=1e-5):
    """
    Full evaluation of SVM model: accuracy + confusion matrices + support vectors.
    """
    support_indices = np.where(alpha > threshold)[0]
    n_sv = len(support_indices)
    percent_sv = 100 * n_sv / len(alpha)

    y_train_pred = predict_svm_dual(X_train, y_train, X_train, alpha, b, kernel)
    y_test_pred  = predict_svm_dual(X_train, y_train, X_test, alpha, b, kernel)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test  = accuracy_score(y_test, y_test_pred)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test  = confusion_matrix(y_test, y_test_pred)

    print("\n--- FULL SVM EVALUATION ---")
    print(f"Train Accuracy         : {acc_train:.4f}")
    print(f"Test Accuracy          : {acc_test:.4f}")
    print(f"Support Vectors        : {n_sv} / {len(alpha)} ({percent_sv:.2f}%)")
    print("Confusion Matrix (Train):")
    print(cm_train)
    print("Confusion Matrix (Test):")
    print(cm_test)

    return {
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "n_support_vectors": n_sv,
        "percent_support_vectors": percent_sv,
        "confusion_matrix_train": cm_train,
        "confusion_matrix_test": cm_test
    }





def decision_function_svm_dual(X_train, y_train, X_test, alpha, b, kernel, tol=1e-5):
    """
    Restituisce i punteggi (decision values) del SVM binario:
    f(x) = sum_i alpha_i * y_i * K(x, x_i) + b
    """
    support_indices = np.where(alpha > tol)[0]
    alpha_sv = alpha[support_indices]
    y_sv = y_train[support_indices]
    X_sv = X_train[support_indices]

    K = compute_kernel_matrix(X_test, X_sv, kernel)
    decision = K @ (alpha_sv * y_sv) + b
    return decision


class OneVsAllSVM:
    def __init__(self, kernel, C=1.0):
        self.kernel = kernel
        self.C = C
        self.models_ = {}
        self.classes_ = None
        self.X_train_ = None
        self.y_train_bin_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_train_ = X.copy()
        for cls in self.classes_:
            y_bin = np.where(y == cls, 1, -1).astype(float)
            alpha, sv_idx, b, obj, n_iter, status, opt_time = train_svm_dual_cvxopt(
                X, y_bin, self.kernel, C=self.C
            )
            self.models_[cls] = (alpha, sv_idx, b)
            self.y_train_bin_[cls] = y_bin

    def decision_function(self, X):
        scores = []
        for cls in self.classes_:
            alpha, sv_idx, b = self.models_[cls]
            y_bin = self.y_train_bin_[cls]
            dec = decision_function_svm_dual(self.X_train_, y_bin, X, alpha, b, self.kernel)
            scores.append(dec)
        return np.vstack(scores).T

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]
