"""
Preprocessing Module
====================
Feature engineering for the restaurant sales dataset.
Creates receipt-level features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
try:
    from .data_loader import SAUCES
except ImportError:
    from data_loader import SAUCES


def create_receipt_features(df: pd.DataFrame, 
                            use_binary: bool = True,
                            exclude_products: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create receipt-level features from raw transaction data.
    
    Args:
        df: DataFrame with raw transaction data (row per product)
        use_binary: If True, use binary (has/not) instead of counts
        exclude_products: List of products to exclude from features
        
    Returns:
        DataFrame with one row per receipt and features as columns
    """
    if exclude_products is None:
        exclude_products = []
    
    receipts = df.groupby('id_bon')
    
    # Get all unique products (excluding specified ones)
    all_products = df['retail_product_name'].unique()
    products_for_features = [p for p in all_products if p not in exclude_products]
    
    features_list = []
    
    for receipt_id, receipt_data in receipts:
        features = {'id_bon': receipt_id}
        
        # Product features (count or binary)
        product_counts = receipt_data['retail_product_name'].value_counts()
        for product in products_for_features:
            if product in product_counts.index:
                if use_binary:
                    features[f'has_{product}'] = 1
                else:
                    features[f'count_{product}'] = product_counts[product]
            else:
                if use_binary:
                    features[f'has_{product}'] = 0
                else:
                    features[f'count_{product}'] = 0
        
        # Aggregate features
        features['cart_size'] = len(receipt_data)
        features['distinct_products'] = receipt_data['retail_product_name'].nunique()
        features['total_value'] = receipt_data['SalePriceWithVAT'].sum()
        
        # Time features
        timestamp = receipt_data['data_bon'].iloc[0]
        features['day_of_week'] = timestamp.dayofweek + 1  # 1-7 (Monday=1)
        features['hour'] = timestamp.hour
        features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    return features_df


def prepare_lr_data(df: pd.DataFrame, 
                    target_sauce: str,
                    filter_product: Optional[str] = None,
                    exclude_all_sauces: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare X and y for Logistic Regression.
    
    Args:
        df: DataFrame with raw transaction data
        target_sauce: The sauce to predict (y=1 if present)
        filter_product: If provided, only use receipts containing this product
        exclude_all_sauces: If True, exclude all sauces from features
        
    Returns:
        Tuple of (X, y, features_df) where features_df contains id_bon for reference
    """
    # Filter to receipts with specific product if requested
    if filter_product:
        receipt_ids = df[df['retail_product_name'] == filter_product]['id_bon'].unique()
        df = df[df['id_bon'].isin(receipt_ids)].copy()
    
    # Determine which products to exclude from features
    if exclude_all_sauces:
        exclude_products = SAUCES
    else:
        exclude_products = [target_sauce]
    
    # Create features
    features_df = create_receipt_features(df, use_binary=True, 
                                          exclude_products=exclude_products)
    
    # Create target variable
    receipts_with_sauce = df[df['retail_product_name'] == target_sauce]['id_bon'].unique()
    features_df['target'] = features_df['id_bon'].isin(receipts_with_sauce).astype(int)
    
    # Separate features and target
    feature_cols = [col for col in features_df.columns 
                    if col not in ['id_bon', 'target']]
    
    X = features_df[feature_cols].values
    y = features_df['target'].values
    
    return X, y, features_df


def normalize_features(X: np.ndarray, 
                       mean: Optional[np.ndarray] = None,
                       std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        X: Feature matrix
        mean: Pre-computed mean (for test data, use training mean)
        std: Pre-computed std (for test data, use training std)
        
    Returns:
        Tuple of (X_normalized, mean, std)
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    X_normalized = (X - mean) / std
    
    return X_normalized, mean, std


def get_feature_names(features_df: pd.DataFrame) -> List[str]:
    """
    Get list of feature column names (excluding id_bon and target).
    
    Args:
        features_df: DataFrame from prepare_lr_data
        
    Returns:
        List of feature names
    """
    return [col for col in features_df.columns if col not in ['id_bon', 'target']]


def create_interaction_features(features_df: pd.DataFrame,
                                 interactions: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between products.
    
    Args:
        features_df: DataFrame with receipt-level features
        interactions: List of tuples like [('has_Fries', 'has_Cola')]
        
    Returns:
        DataFrame with additional interaction columns
    """
    df = features_df.copy()
    
    for feat1, feat2 in interactions:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f'{feat1}_x_{feat2}'
            df[interaction_name] = df[feat1] * df[feat2]
    
    return df


if __name__ == "__main__":
    # Quick test
    from data_loader import load_raw_data, SAUCES
    import sys
    
    if len(sys.argv) > 1:
        df = load_raw_data(sys.argv[1])
        
        # Test basic feature creation
        features = create_receipt_features(df)
        print(f"Created {len(features)} receipts with {len(features.columns)} features")
        print(f"\nFeature columns: {features.columns.tolist()[:10]}...")
        
        # Test LR data preparation for Crazy Sauce
        X, y, features_df = prepare_lr_data(df, "Crazy Sauce", 
                                            filter_product="Crazy Schnitzel")
        print(f"\nCrazy Sauce prediction (filtered by Crazy Schnitzel):")
        print(f"  X shape: {X.shape}")
        print(f"  Positive samples: {y.sum()} ({100*y.mean():.1f}%)")
