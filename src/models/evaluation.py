"""
Evaluation Metrics Module
=========================
Classification metrics for model evaluation.
"""

import numpy as np
from typing import Tuple, Dict


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        2x2 confusion matrix: [[TN, FP], [FN, TP]]
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision score.
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Precision score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity) score.
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Recall score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        F1 score
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (true negative rate).
    
    Specificity = TN / (TN + FP)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Generate a comprehensive classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with all metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'recall': recall(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'specificity': specificity(y_true, y_pred),
        'confusion_matrix': cm,
        'true_negatives': cm[0, 0],
        'false_positives': cm[0, 1],
        'false_negatives': cm[1, 0],
        'true_positives': cm[1, 1],
        'total_samples': len(y_true),
        'positive_samples': int(np.sum(y_true == 1)),
        'negative_samples': int(np.sum(y_true == 0))
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print a formatted classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    report = classification_report(y_true, y_pred)
    
    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(f"\nTotal Samples: {report['total_samples']}")
    print(f"  - Positive: {report['positive_samples']} ({100*report['positive_samples']/report['total_samples']:.1f}%)")
    print(f"  - Negative: {report['negative_samples']} ({100*report['negative_samples']/report['total_samples']:.1f}%)")
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg  {report['true_negatives']:5d}  {report['false_positives']:5d}")
    print(f"  Actual Pos  {report['false_negatives']:5d}  {report['true_positives']:5d}")
    print("\nMetrics:")
    print(f"  Accuracy:    {report['accuracy']:.4f}")
    print(f"  Precision:   {report['precision']:.4f}")
    print(f"  Recall:      {report['recall']:.4f}")
    print(f"  F1 Score:    {report['f1_score']:.4f}")
    print(f"  Specificity: {report['specificity']:.4f}")
    print("=" * 50)


def roc_curve(y_true: np.ndarray, y_proba: np.ndarray, n_thresholds: int = 100):
    """
    Calculate ROC curve data points.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        n_thresholds: Number of threshold points
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # True positives and false positives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        # TPR = TP / (TP + FN)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds


def roc_auc_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate ROC-AUC score using trapezoidal rule.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        ROC-AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    # Sort by FPR for proper integration
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Trapezoidal rule for AUC
    auc = np.trapz(tpr_sorted, fpr_sorted)
    
    return auc


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                   title: str = "ROC Curve", save_path: str = None):
    """
    Plot ROC curve (requires matplotlib).
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        title: Plot title
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.3)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax

