"""
Logistic Regression Implementation from Scratch
================================================
Binary classification using gradient descent optimization.
"""

import numpy as np
from typing import Optional, Tuple


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch.
    
    Uses sigmoid activation and cross-entropy loss with gradient descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 regularization: float = 0.0, verbose: bool = False):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of training iterations
            regularization: L2 regularization strength (0 = no regularization)
            verbose: If True, print loss during training
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list = []
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        Args:
            z: Linear combination of inputs and weights
            
        Returns:
            Probability values between 0 and 1
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities
            
        Returns:
            Average cross-entropy loss
        """
        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add L2 regularization term
        if self.regularization > 0 and self.weights is not None:
            loss += (self.regularization / (2 * len(y_true))) * np.sum(self.weights ** 2)
        
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
            
        Returns:
            self (for method chaining)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Add regularization gradient
            if self.regularization > 0:
                dw += (self.regularization / n_samples) * self.weights
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Track loss
            loss = self._cross_entropy_loss(y, y_pred)
            self.loss_history.append(loss)
            
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class 1.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Probability of class 1 for each sample
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features, shape (n_samples, n_features)
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> dict:
        """
        Get feature importance based on absolute weight values.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of feature name -> (weight, abs_importance)
        """
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.weights))]
        
        importance = {}
        for name, weight in zip(feature_names, self.weights):
            importance[name] = {
                'weight': weight,
                'abs_importance': abs(weight)
            }
        
        # Sort by absolute importance
        importance = dict(sorted(importance.items(), 
                                  key=lambda x: x[1]['abs_importance'], 
                                  reverse=True))
        return importance


# Alias for backward compatibility
LogisticRegressionScratch = LogisticRegression


def train_test_split_receipts(X: np.ndarray, y: np.ndarray, 
                               receipt_ids: np.ndarray,
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                                 np.ndarray, np.ndarray]:
    """
    Split data at receipt level to prevent data leakage.
    
    Args:
        X: Features
        y: Labels
        receipt_ids: Receipt IDs for each sample
        test_size: Proportion of receipts to use for testing
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    
    unique_ids = np.unique(receipt_ids)
    np.random.shuffle(unique_ids)
    
    split_idx = int(len(unique_ids) * (1 - test_size))
    train_ids = unique_ids[:split_idx]
    
    train_mask = np.isin(receipt_ids, train_ids)
    
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    return X_train, X_test, y_train, y_test
