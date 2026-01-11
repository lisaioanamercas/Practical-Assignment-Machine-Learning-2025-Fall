"""
Data Loader Module
==================
Functions for loading and validating the restaurant sales dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List


# List of standalone sauces (not "with ... sauce" products)
SAUCES = [
    "Crazy Sauce",
    "Cheddar Sauce", 
    "Extra Cheddar Sauce",
    "Garlic Sauce",
    "Tomato Sauce",
    "Blueberry Sauce",
    "Spicy Sauce",
    "Pink Sauce"
]

# Required columns in the dataset
REQUIRED_COLUMNS = ['id_bon', 'data_bon', 'retail_product_name', 'SalePriceWithVAT']


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw restaurant sales dataset from CSV.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame with the raw data
    """
    df = pd.read_csv(path)
    
    # Convert date column to datetime
    if 'data_bon' in df.columns:
        df['data_bon'] = pd.to_datetime(df['data_bon'])
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset has all required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Get quick statistics about the dataset.
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        Dictionary with statistics
    """
    return {
        'total_rows': len(df),
        'unique_receipts': df['id_bon'].nunique(),
        'unique_products': df['retail_product_name'].nunique(),
        'date_range': (df['data_bon'].min(), df['data_bon'].max()),
        'avg_cart_size': len(df) / df['id_bon'].nunique()
    }


def get_receipts_with_product(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """
    Filter dataset to only include receipts containing a specific product.
    
    Args:
        df: DataFrame with raw data
        product_name: Name of the product to filter by
        
    Returns:
        DataFrame with only receipts containing the product
    """
    # Get receipt IDs that contain the product
    receipt_ids = df[df['retail_product_name'] == product_name]['id_bon'].unique()
    
    # Return all rows from those receipts
    return df[df['id_bon'].isin(receipt_ids)].copy()


def get_receipts_without_product(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """
    Filter dataset to only include receipts NOT containing a specific product.
    
    Args:
        df: DataFrame with raw data
        product_name: Name of the product to exclude
        
    Returns:
        DataFrame with only receipts NOT containing the product
    """
    # Get receipt IDs that contain the product
    receipt_ids_with = df[df['retail_product_name'] == product_name]['id_bon'].unique()
    
    # Return all rows from receipts NOT in that list
    return df[~df['id_bon'].isin(receipt_ids_with)].copy()


def split_by_receipt(df: pd.DataFrame, train_ratio: float = 0.8, 
                     random_state: int = 42) -> tuple:
    """
    Split data at receipt level (not row level).
    
    Args:
        df: DataFrame with raw data
        train_ratio: Ratio for training set (default 0.8)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    import numpy as np
    np.random.seed(random_state)
    
    # Get unique receipt IDs
    receipt_ids = df['id_bon'].unique()
    np.random.shuffle(receipt_ids)
    
    # Split IDs
    split_idx = int(len(receipt_ids) * train_ratio)
    train_ids = receipt_ids[:split_idx]
    test_ids = receipt_ids[split_idx:]
    
    # Create train/test DataFrames
    train_df = df[df['id_bon'].isin(train_ids)].copy()
    test_df = df[df['id_bon'].isin(test_ids)].copy()
    
    return train_df, test_df


def split_by_time(df: pd.DataFrame, split_date: str) -> tuple:
    """
    Split data by date (train on older, test on newer).
    
    Args:
        df: DataFrame with raw data
        split_date: Date string for split point (format: 'YYYY-MM-DD')
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_dt = pd.to_datetime(split_date)
    
    train_df = df[df['data_bon'] < split_dt].copy()
    test_df = df[df['data_bon'] >= split_dt].copy()
    
    return train_df, test_df


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        df = load_raw_data(sys.argv[1])
        validate_data(df)
        stats = get_data_stats(df)
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
