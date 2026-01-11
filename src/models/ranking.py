"""
Naive Bayes Classifier Implementation from Scratch
===================================================
For ranking/recommendation in restaurant sales prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class NaiveBayesClassifier:
    """
    Gaussian Naive Bayes classifier implemented from scratch.
    
    Assumes continuous features follow Gaussian distribution.
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize the classifier.
        
        Args:
            var_smoothing: Portion of the largest variance added for stability
        """
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[Dict[int, float]] = None
        self.mean_: Optional[Dict[int, np.ndarray]] = None
        self.var_: Optional[Dict[int, np.ndarray]] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesClassifier':
        """
        Fit the Naive Bayes model.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Labels, shape (n_samples,)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        # Calculate class priors
        self.class_prior_ = {}
        self.mean_ = {}
        self.var_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            self.class_prior_[c] = len(X_c) / n_samples
            self.mean_[c] = np.mean(X_c, axis=0)
            self.var_[c] = np.var(X_c, axis=0) + self.var_smoothing
            
        return self
    
    def _log_likelihood(self, X: np.ndarray, c: int) -> np.ndarray:
        """
        Calculate log-likelihood of X given class c (Gaussian distribution).
        
        Args:
            X: Feature matrix
            c: Class label
            
        Returns:
            Log-likelihood for each sample
        """
        mean = self.mean_[c]
        var = self.var_[c]
        
        # Log of Gaussian PDF
        # log(P(x|c)) = -0.5 * (log(2*pi*var) + (x-mean)^2/var)
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, 
            axis=1
        )
        return log_likelihood
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log-probability for each class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Log-probabilities, shape (n_samples, n_classes)
        """
        log_proba = []
        
        for c in self.classes_:
            log_prior = np.log(self.class_prior_[c])
            log_likelihood = self._log_likelihood(X, c)
            log_proba.append(log_prior + log_likelihood)
        
        return np.column_stack(log_proba)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability for each class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probabilities, shape (n_samples, n_classes)
        """
        log_proba = self.predict_log_proba(X)
        
        # Softmax to convert log-proba to proba
        exp_log_proba = np.exp(log_proba - np.max(log_proba, axis=1, keepdims=True))
        proba = exp_log_proba / np.sum(exp_log_proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class KNNClassifier:
    """
    K-Nearest Neighbors classifier implemented from scratch.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """
        Initialize KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: 'uniform' or 'distance'
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the KNN model (just store the data).
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _get_neighbors(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get k nearest neighbors for a single sample.
        
        Returns:
            Tuple of (distances, labels) for k nearest neighbors
        """
        distances = np.array([self._euclidean_distance(x, x_train) 
                             for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]
        return k_distances, k_labels
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability for each class
        """
        probas = []
        
        for x in X:
            distances, labels = self._get_neighbors(x)
            
            # Calculate weights
            if self.weights == 'distance':
                # Avoid division by zero
                weights = 1 / (distances + 1e-10)
            else:
                weights = np.ones_like(distances)
            
            # Calculate weighted votes
            class_weights = {}
            for c in self.classes_:
                class_weights[c] = np.sum(weights[labels == c])
            
            total_weight = sum(class_weights.values())
            proba = [class_weights.get(c, 0) / total_weight for c in self.classes_]
            probas.append(proba)
        
        return np.array(probas)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class ProductRanker:
    """
    Product ranking for upselling using various ML algorithms.
    
    Score(p | cart) = P(p | cart) * price(p)
    """
    
    def __init__(self, algorithm: str = 'naive_bayes'):
        """
        Initialize ranker.
        
        Args:
            algorithm: 'naive_bayes' or 'knn'
        """
        self.algorithm = algorithm
        self.models: Dict[str, object] = {}
        self.product_prices: Dict[str, float] = {}
        self.products: List[str] = []
        self.feature_names: List[str] = []
        
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
            feature_names: List[str], product_prices: Dict[str, float]):
        """
        Train a model for each product.
        
        Args:
            X: Feature matrix (n_carts x n_features)
            y_dict: Dict of product_name -> binary labels
            feature_names: Names of features
            product_prices: Dict of product_name -> price
        """
        self.feature_names = feature_names
        self.product_prices = product_prices
        self.products = list(y_dict.keys())
        
        for product, y in y_dict.items():
            if self.algorithm == 'naive_bayes':
                model = NaiveBayesClassifier()
            else:
                model = KNNClassifier(n_neighbors=5, weights='distance')
            
            model.fit(X, y)
            self.models[product] = model
    
    def rank(self, X: np.ndarray, exclude_products: Optional[List[str]] = None,
             top_k: int = 5) -> List[List[Tuple[str, float, float]]]:
        """
        Rank products for each cart.
        
        Args:
            X: Feature matrix for carts
            exclude_products: Products already in cart (per cart)
            top_k: Number of top products to return
            
        Returns:
            List of rankings, each ranking is [(product, score, prob), ...]
        """
        if exclude_products is None:
            exclude_products = [[] for _ in range(len(X))]
        
        rankings = []
        
        for i, x in enumerate(X):
            scores = []
            x_reshaped = x.reshape(1, -1)
            
            for product in self.products:
                if product in exclude_products[i]:
                    continue
                
                model = self.models[product]
                # Get probability of class 1 (product purchased)
                proba = model.predict_proba(x_reshaped)[0]
                
                if len(proba) > 1:
                    prob = proba[1]  # Probability of purchase
                else:
                    prob = proba[0]
                
                price = self.product_prices.get(product, 0)
                score = prob * price  # Expected value
                
                scores.append((product, score, prob))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            rankings.append(scores[:top_k])
        
        return rankings


def calculate_hit_at_k(rankings: List[List[Tuple]], actual_products: List[str], 
                       k: int = 5) -> float:
    """
    Calculate Hit@K metric.
    
    Args:
        rankings: List of rankings from ranker
        actual_products: List of actual products purchased
        k: Number of top predictions to consider
        
    Returns:
        Hit rate (proportion of hits)
    """
    hits = 0
    for ranking, actual in zip(rankings, actual_products):
        top_k_products = [r[0] for r in ranking[:k]]
        if actual in top_k_products:
            hits += 1
    return hits / len(rankings)


def calculate_precision_at_k(rankings: List[List[Tuple]], 
                             actual_products: List[List[str]], 
                             k: int = 5) -> float:
    """
    Calculate Precision@K metric.
    
    Args:
        rankings: List of rankings from ranker
        actual_products: List of lists of actual products purchased
        k: Number of top predictions to consider
        
    Returns:
        Average precision at K
    """
    precisions = []
    for ranking, actuals in zip(rankings, actual_products):
        top_k_products = [r[0] for r in ranking[:k]]
        relevant = sum(1 for p in top_k_products if p in actuals)
        precisions.append(relevant / k)
    return np.mean(precisions)


def calculate_mrr(rankings: List[List[Tuple]], actual_products: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        rankings: List of rankings from ranker
        actual_products: List of actual products purchased
        
    Returns:
        Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    for ranking, actual in zip(rankings, actual_products):
        products = [r[0] for r in ranking]
        if actual in products:
            rank = products.index(actual) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)
