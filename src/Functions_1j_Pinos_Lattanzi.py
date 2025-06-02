import numpy as np
from sklearn.model_selection import KFold
import time
from scipy.optimize import minimize

def flatten_params(params):
    """
    Flattens the param dict {W0, b0, W1, b1, ...} into a 1D array,
    and records each variable's shape and name.

    Returns:
        - flat_params: concatenated 1D array
        - metadata: list of (key, shape, size)
    """
    flat_list = []
    metadata = []

    for key, value in params.items():
        flat_value = value.ravel()
        flat_list.append(flat_value)
        metadata.append((key, value.shape, flat_value.size))

    flat_params = np.concatenate(flat_list)
    return flat_params, metadata

def unflatten_params(flat_array, metadata):
    """
    Reconstructs the original param dict from the flat array and metadata.

    Args:
        flat_array: 1D numpy array with all params
        metadata: list of (key, shape, size) as returned by flatten_params

    Returns:
        Dictionary with reshaped weight/bias matrices
    """
    params = {}
    index = 0
    for key, shape, size in metadata:
        param = flat_array[index : index + size].reshape(shape)
        params[key] = param
        index += size

    return params

def objective_function(w_flat, metadata, mlp_model, X, y):
    """
    Computes the loss and gradient for scipy.optimize.minimize.

    Args:
        w_flat: Flattened parameter vector
        metadata: Shapes and names of original parameters
        mlp_model: MLP model instance
        X: Input data (n_samples × n_features)
        y: Target values (n_samples,)

    Returns:
        loss: Scalar loss value
        grad_flat: Flattened gradient vector
    """
    # 1. Reconstruct parameter dictionary from flat vector
    params = unflatten_params(w_flat, metadata)
    mlp_model.params = params  # update model with current weights

    # 2. Forward pass to compute predictions
    y_pred = mlp_model.forward(X)

    # 3. Compute loss (MSE + L2)
    loss = mlp_model.compute_loss(y, y_pred)

    # 4. Backward pass to compute gradients
    grads = mlp_model.backward(X, y)

    # 5. Flatten gradients to return to optimizer
    grad_flat, _ = flatten_params(grads)

    return loss, grad_flat


class MLP:
    def __init__(self, layer_sizes, activation='sigmoid', lambda_reg=0.0):
        self.layer_sizes = layer_sizes      # e.g. [D, 64, 32, 1]
        self.lambda_reg = lambda_reg
        self.activation_name = activation
        self.params = self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initialize weights and biases for all layers using Xavier initialization.

        For each layer l:
            - Weight matrix W[l] ∈ ℝ^{n_l x n_{l-1}} initialized with N(0, 1/n_{l-1})
            - Bias vector b[l] initialized as zeros

        Returns:
            A dictionary containing all initialized weights and biases:
            {
                "W0": ..., "b0": ...,
                "W1": ..., "b1": ...,
                ...
            }
        """
        np.random.seed(0)  # For reproducibility
        params = {}
        for i in range(1, len(self.layer_sizes)):
            input_size = self.layer_sizes[i - 1]
            output_size = self.layer_sizes[i]
            
            #  Initialization : : preserves variance of signals
            weight = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
            bias = np.zeros((output_size, 1))
            
            params[f"W{i-1}"] = weight
            params[f"b{i-1}"] = bias
            
        return params

    def activation(self, x):
        """
        Applies the chosen activation function element-wise.

        Args:
            x: Pre-activation input (numpy array)

        Returns:
            Activated output (numpy array)
        """
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function: choose 'sigmoid' or 'tanh'")


    
    def activation_derivative(self, x):
        """
        Computes the derivative of the chosen activation function.

        Args:
            x: Pre-activation input (numpy array)

        Returns:
            Derivative of activation function evaluated at x
        """
        if self.activation_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError("Unsupported activation function: choose 'sigmoid' or 'tanh'")


    def forward(self, X):
        """
        Performs the forward pass of the MLP.

        Stores:
            - z_list: pre-activation values (W·a + b)
            - a_list: activations (after non-linearity)

        Args:
            X: Input matrix (shape: n_features × n_samples)

        Returns:
            Output prediction (shape: 1 × n_samples)
        """
        a = X.T  # Each column is a sample
        self.z_list = []
        self.a_list = [a]  # First activation is the input

        num_layers = len(self.layer_sizes) - 1  # Exclude input layer

        for i in range(num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            z = np.dot(W, a) + b  # pre-activation
            self.z_list.append(z)

            # Apply activation for hidden layers only
            if i < num_layers - 1:
                a = self.activation(z)
            else:
                a = z  # Linear output for regression

            self.a_list.append(a)

        return a  # final output (predictions)

    
    def compute_loss(self, y_true, y_pred):
        """
        Computes the L2-regularized Mean Squared Error loss.

        Args:
            y_true: Ground truth target values (shape: n_samples,)
            y_pred: Predicted values (shape: 1 × n_samples)

        Returns:
            Scalar value of the loss
        """
        m = y_true.shape[0]
        error = y_pred.flatten() - y_true  # (n_samples,)
        mse = np.mean(error ** 2)

        # L2 regularization (only on weights, not biases)
        l2_penalty = 0
        for key in self.params:
            if key.startswith("W"):
                l2_penalty += np.sum(self.params[key] ** 2)

        loss = mse + self.lambda_reg * l2_penalty
        return loss

    
    def backward(self, X, y):
        """
        Performs backpropagation and computes gradients of the loss w.r.t. all weights and biases.

        Returns:
            Dictionary of gradients:
            {
                "dW0": ..., "db0": ...,
                "dW1": ..., "db1": ...,
                ...
            }
        """
        m = y.shape[0]
        y = y.reshape(1, -1)  # Shape: (1, n_samples)
        grads = {}

        # Initialize gradient from output layer
        a_final = self.a_list[-1]  # Output predictions
        dz = (a_final - y) * (2 / m)  # Derivative of MSE w.r.t. output z

        for i in reversed(range(len(self.layer_sizes) - 1)):
            a_prev = self.a_list[i]
            W = self.params[f"W{i}"]

            dW = np.dot(dz, a_prev.T) + self.lambda_reg * W  # gradient + L2
            db = np.sum(dz, axis=1, keepdims=True)

            grads[f"dW{i}"] = dW
            grads[f"db{i}"] = db

            if i != 0:
                z_prev = self.z_list[i - 1]
                da = np.dot(W.T, dz)
                dz = da * self.activation_derivative(z_prev)  # chain rule

        return grads


    def predict(self, X):
        """
        Runs a forward pass through the network and returns predicted outputs.

        Args:
            X: Input matrix of shape (n_samples, n_features)

        Returns:
            1D NumPy array of predicted values (n_samples,)
        """
        output = self.forward(X)  # forward already transposes input
        return output.flatten()   # convert from (1, n_samples) to (n_samples,)







def train_model(X, y, layer_sizes, activation='tanh', lambda_reg=1e-3, max_iter=200):
    """
    Trains an MLP using scipy.optimize.minimize and returns the trained model and metadata.

    Args:
        X: Training features (n_samples × n_features)
        y: Target values (n_samples,)
        layer_sizes: List defining the architecture, e.g. [32, 64, 1]
        activation: Activation function ('sigmoid' or 'tanh')
        lambda_reg: L2 regularization coefficient
        max_iter: Maximum number of iterations for the optimizer

    Returns:
        model: Trained MLP instance
        result: Optimizer result object
        training_time: Elapsed time in seconds
    """
    # 1. Initialize model
    model = MLP(layer_sizes, activation=activation, lambda_reg=lambda_reg)

    # 2. Flatten initial weights and get metadata
    w0, metadata = flatten_params(model.params)

    # 3. Define objective function with fixed args
    def wrapped_objective(w_flat):
        return objective_function(w_flat, metadata, model, X, y)

    # 4. Run optimizer
    start_time = time.time()
    result = minimize(
        fun=wrapped_objective,
        x0=w0,
        method='L-BFGS-B',
        jac=True,                   # we return loss and grad
        options={'maxiter': max_iter}
    )
    end_time = time.time()

    # 5. Update model with final weights
    final_params = unflatten_params(result.x, metadata)
    model.params = final_params

    training_time = end_time - start_time

    return model, result, training_time





def cross_validate_model(X, y, k_folds, configs, max_iter=200, seed=42, scoring='mse'):
    """
    Performs k-fold cross-validation to evaluate different MLP configurations,
    with normalization of the target variable inside each fold.

    Args:
        X, y: Full training data
        k_folds: Number of folds (e.g., 5)
        configs: List of dicts with keys: 'layers', 'activation', 'lambda'
        max_iter: Max iterations per training
        seed: Random seed for reproducibility
        scoring: 'mse' or 'mape'

    Returns:
        best_config: Config with lowest average validation error
        results: List of tuples (config, avg_val_error)
        best_score: Lowest average validation error (for report use)
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    results = []

    for config in configs:
        val_errors = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Normalize target in training set
            y_min = y_train.min()
            y_max = y_train.max()
            y_train_norm = (y_train - y_min) / (y_max - y_min)
            y_val_norm = (y_val - y_min) / (y_max - y_min)

            # Train model on normalized target
            model, _, _ = train_model(
                X_train, y_train_norm,
                layer_sizes=config['layers'],
                activation=config['activation'],
                lambda_reg=config['lambda'],
                max_iter=max_iter
            )

            # Predict and denormalize output
            y_val_pred_norm = model.predict(X_val)
            y_val_pred = y_val_pred_norm * (y_max - y_min) + y_min

            # Compute validation error
            if scoring == 'mse':
                err = np.mean((y_val_pred - y_val) ** 2)
            elif scoring == 'mape':
                err = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
            else:
                raise ValueError("Unsupported scoring")

            val_errors.append(err)

        avg_val_error = np.mean(val_errors)
        results.append((config, avg_val_error))
        print(f"Config: {config} → Avg Val {scoring.upper()}: {avg_val_error:.4f}")

    # Select best config based on lowest avg val error
    best_config, best_score = min(results, key=lambda x: x[1])
    return best_config, results, best_score
