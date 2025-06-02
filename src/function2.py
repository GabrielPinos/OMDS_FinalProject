import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, layer_sizes, activation='relu', lambda_reg=0.0, dropout_rate=0.0):
        self.layer_sizes = layer_sizes
        self.lambda_reg = lambda_reg
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.training = True  # Training mode flag for dropout
        self.params = self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initialize weights and biases for all layers using improved initialization.
        
        Uses:
        - He initialization for ReLU/LeakyReLU
        - Xavier initialization for sigmoid/tanh
        - Small random initialization for biases

        Returns:
            A dictionary containing all initialized weights and biases
        """
        np.random.seed(42)  # Better seed for reproducibility
        params = {}
        
        for i in range(1, len(self.layer_sizes)):
            input_size = self.layer_sizes[i - 1]
            output_size = self.layer_sizes[i]
            
            # Choose initialization based on activation function
            if self.activation_name in ['relu']:
                # He initialization for ReLU variants
                std = np.sqrt(2.0 / input_size)
            else:
                # Xavier initialization for sigmoid/tanh
                std = np.sqrt(1.0 / input_size)
            
            weight = np.random.randn(output_size, input_size) * std
            # Small random bias initialization instead of zeros
            bias = np.random.randn(output_size, 1) * 0.01
            
            params[f"W{i-1}"] = weight
            params[f"b{i-1}"] = bias
            
        return params

    def activation(self, x):
        """
        Applies the chosen activation function element-wise with numerical stability.
        """
        if self.activation_name == 'sigmoid':
            # Numerically stable sigmoid
            return np.where(x >= 0, 
                          1 / (1 + np.exp(-x)), 
                          np.exp(x) / (1 + np.exp(x)))
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation function: choose 'sigmoid', 'tanh', 'relu'")

    def activation_derivative(self, x):
        """
        Computes the derivative of the chosen activation function with numerical stability.
        """
        if self.activation_name == 'sigmoid':
            sig = self.activation(x)
            return sig * (1 - sig)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == 'relu':
            return (x > 0).astype(float)

        else:
            raise ValueError("Unsupported activation function")

    def apply_dropout(self, a):
        """Apply dropout during training"""
        if self.training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape) / (1 - self.dropout_rate)
            return a * mask
        return a

    def forward(self, X):
        """
        Performs the forward pass of the MLP with improved numerical stability.
        """
        a = X.T  # Each column is a sample
        self.z_list = []
        self.a_list = [a]  # First activation is the input

        num_layers = len(self.layer_sizes) - 1  # Exclude input layer

        for i in range(num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            
            # Apply dropout to previous layer's activations (except input)
            if i > 0:
                a = self.apply_dropout(a)
            
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
        Computes the L2-regularized Mean Squared Error loss with improved regularization.
        """
        m = y_true.shape[0]
        error = y_pred.flatten() - y_true  # (n_samples,)
        mse = np.mean(error ** 2)

        # Improved L2 regularization (only on weights, not biases)
        # Scale regularization by number of parameters to make it more interpretable
        l2_penalty = 0
        total_params = 0
        for key in self.params:
            if key.startswith("W"):
                weight_matrix = self.params[key]
                l2_penalty += np.sum(weight_matrix ** 2)
                total_params += weight_matrix.size

        # Normalize regularization by total number of parameters
        if total_params > 0:
            l2_penalty = l2_penalty / total_params

        loss = mse + self.lambda_reg * l2_penalty
        return loss

    def backward(self, X, y):
        """
        Performs backpropagation with improved gradient computation.
        """
        m = y.shape[0]
        y = y.reshape(1, -1)  # Shape: (1, n_samples)
        grads = {}

        # Initialize gradient from output layer
        a_final = self.a_list[-1]  # Output predictions
        dz = (a_final - y) / m  # Derivative of MSE w.r.t. output z (removed factor of 2)

        for i in reversed(range(len(self.layer_sizes) - 1)):
            a_prev = self.a_list[i]
            W = self.params[f"W{i}"]

            # Improved gradient computation with proper regularization scaling
            total_params = sum(p.size for key, p in self.params.items() if key.startswith("W"))
            reg_term = (self.lambda_reg / total_params) * W if total_params > 0 else 0
            
            dW = np.dot(dz, a_prev.T) + reg_term
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
        Sets training mode to False to disable dropout during prediction.
        """
        original_training = self.training
        self.training = False  # Disable dropout for prediction
        
        output = self.forward(X)  # forward already transposes input
        result = output.flatten()   # convert from (1, n_samples) to (n_samples,)
        
        self.training = original_training  # Restore original training mode
        return result


def train_model(X, y, layer_sizes, activation='relu', lambda_reg=1e-3, dropout_rate=0.0, 
                max_iter=500, method='L-BFGS-B'):
    """
    Trains an MLP using scipy.optimize.minimize with improved settings.
    """
    # 1. Initialize model
    model = MLP(layer_sizes, activation=activation, lambda_reg=lambda_reg, dropout_rate=dropout_rate)

    # 2. Flatten initial weights and get metadata
    w0, metadata = flatten_params(model.params)

    # 3. Define objective function with fixed args
    def wrapped_objective(w_flat):
        return objective_function(w_flat, metadata, model, X, y)

    # 4. Run optimizer with improved settings
    start_time = time.time()
    
    optimizer_options = {
        'maxiter': max_iter,
        'ftol': 1e-9,    # Tighter convergence tolerance
        'gtol': 1e-8,    # Gradient tolerance
    }
    
    result = minimize(
        fun=wrapped_objective,
        x0=w0,
        method=method,
        jac=True,                   # we return loss and grad
        options=optimizer_options
    )
    end_time = time.time()

    # 5. Update model with final weights
    final_params = unflatten_params(result.x, metadata)
    model.params = final_params

    training_time = end_time - start_time

    return model, result, training_time


def cross_validate_model(X, y, k_folds, configs, max_iter=500, seed=42, scoring='mse'):
    """
    Performs k-fold cross-validation that matches your existing workflow.
    Note: X should already be scaled (as in your notebook workflow).
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    results = []

    print(f"Starting {k_folds}-fold cross-validation with {len(configs)} configurations...")
    
    for config_idx, config in enumerate(configs):
        print(f"\nTesting config {config_idx + 1}/{len(configs)}: {config}")
        val_errors = []
        fold_times = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_start = time.time()
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Target normalization (same as your workflow)
            y_min, y_max = y_train.min(), y_train.max()
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1  # Avoid division by zero
            
            y_train_norm = (y_train - y_min) / y_range

            # Train model on normalized data
            try:
                model, result, train_time = train_model(
                    X_train, y_train_norm,
                    layer_sizes=config['layers'],
                    activation=config['activation'],
                    lambda_reg=config['lambda'],
                    dropout_rate=config.get('dropout', 0.0),
                    max_iter=max_iter
                )

                # Predict and denormalize output
                y_val_pred_norm = model.predict(X_val)
                y_val_pred = y_val_pred_norm * y_range + y_min

                # Compute validation error
                if scoring == 'mse':
                    err = np.mean((y_val_pred - y_val) ** 2)
                elif scoring == 'mape':
                    # Avoid division by zero in MAPE
                    mask = np.abs(y_val) > 0.1  # Only consider non-zero targets
                    if np.sum(mask) > 0:
                        err = np.mean(np.abs((y_val[mask] - y_val_pred[mask]) / y_val[mask])) * 100
                    else:
                        err = float('inf')
                else:
                    raise ValueError("Unsupported scoring method")

                val_errors.append(err)
                fold_times.append(time.time() - fold_start)
                
                print(f"  Fold {fold_idx + 1}: {scoring.upper()} = {err:.4f}, "
                      f"Train time = {train_time:.2f}s, Converged = {result.success}")
                
            except Exception as e:
                print(f"  Fold {fold_idx + 1}: Error during training - {str(e)}")
                val_errors.append(float('inf'))
                fold_times.append(time.time() - fold_start)

        if val_errors:
            # Filter out infinite errors for average calculation
            valid_errors = [e for e in val_errors if not np.isinf(e)]
            if valid_errors:
                avg_val_error = np.mean(valid_errors)
                std_val_error = np.std(valid_errors)
                avg_time = np.mean(fold_times)
            else:
                avg_val_error = float('inf')
                std_val_error = 0
                avg_time = np.mean(fold_times)
        else:
            avg_val_error = float('inf')
            std_val_error = 0
            avg_time = 0

        results.append((config, avg_val_error, std_val_error))
        print(f"  Average {scoring.upper()}: {avg_val_error:.4f} ± {std_val_error:.4f}, "
              f"Avg time per fold: {avg_time:.2f}s")

    # Select best config based on lowest avg val error
    valid_results = [(c, e, s) for c, e, s in results if not np.isinf(e)]
    if valid_results:
        best_config, best_score, best_std = min(valid_results, key=lambda x: x[1])
        print(f"\nBest configuration: {best_config}")
        print(f"Best {scoring.upper()}: {best_score:.4f} ± {best_std:.4f}")
    else:
        print("\nNo valid configurations found!")
        best_config, best_score = None, float('inf')

    # Return format matching your original function
    return best_config, results


def evaluate_model(model, X, y, scaler=None, y_normalizer=None, dataset_name="Dataset"):
    """
    Evaluate model performance with proper scaling and denormalization.
    
    Args:
        model: Trained MLP model
        X: Input features
        y: True targets
        scaler: Fitted StandardScaler for features (optional)
        y_normalizer: Dict with 'min' and 'range' for target denormalization (optional)
        dataset_name: Name for printing results
    
    Returns:
        dict: Dictionary containing MSE and MAPE scores
    """
    # Scale features if scaler provided
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Get predictions
    y_pred_norm = model.predict(X_scaled)
    
    # Denormalize predictions if normalizer provided
    if y_normalizer is not None:
        y_pred = y_pred_norm * y_normalizer['range'] + y_normalizer['min']
    else:
        y_pred = y_pred_norm
    
    # Calculate metrics
    mse = np.mean((y_pred - y) ** 2)
    
    # MAPE calculation with zero-handling
    mask = np.abs(y) > 0.1
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
    else:
        mape = float('inf')
    
    print(f"{dataset_name} Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'mse': mse, 'mape': mape, 'predictions': y_pred}


# Example usage and improved hyperparameter configurations
def get_improved_configs():
    """
    Returns a set of improved hyperparameter configurations for age prediction.
    """
    configs = []
    
    # Define base architectures (assuming input size will be added dynamically)
    architectures = [
        [128, 64, 1],           # Medium network
        [256, 128, 64, 1],      # Deeper network
        [512, 256, 128, 1],     # Even deeper
        [256, 256, 128, 1],     # Wider hidden layers
        [128, 128, 64, 32, 1],  # More layers
    ]
    
    activations = ['relu', 'tanh']
    lambda_values = [1e-5, 1e-4, 1e-3, 1e-2]
    dropout_values = [0.0, 0.1, 0.2]
    
    # Generate combinations (limited to avoid too many configs)
    for arch in architectures[:3]:  # Test top 3 architectures
        for activation in activations:
            for lambda_reg in lambda_values:
                for dropout in dropout_values[:2]:  # Test 2 dropout values
                    configs.append({
                        'layers': arch,  # Will be updated with input size
                        'activation': activation,
                        'lambda': lambda_reg,
                        'dropout': dropout
                    })
    
    return configs[:24]  # Limit to reasonable number for testing