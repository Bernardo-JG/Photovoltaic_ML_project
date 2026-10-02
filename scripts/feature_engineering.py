"""
Este documento `feature_engineering.py` contém funções auxiliares para criar e transformar 
features no dataset, para a entrada nos modelos ser melhor :) 
"""

import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def handle_missing_values(data):
    """
    Handle missing values in the dataset.

    Parameters:
    data (DataFrame): Input dataset.

    Returns:
    DataFrame: Dataset with missing values handled.
    """
    # Example: Fill missing values with the median value of each column
    return data.fillna(data.median())

def create_new_features(data):
    """
    Create new features from existing data.

    Parameters:
    data (DataFrame): Input dataset.

    Returns:
    DataFrame: Dataset with new features.
    """
    # Example: Create a feature that combines 'temperature' and 'irradiance'
    data['temp_irradiance_interaction'] = data['temperature'] * data['irradiance']
    
    # Example: Create a feature that indicates whether the PV module type is a specific type
    data['is_type_A'] = (data['pv_module_type'] == 'Type A').astype(int)
    
    return data

def encode_categorical_features(data):
    """
    Convert categorical variables to dummy variables.

    Parameters:
    data (DataFrame): Input dataset.

    Returns:
    DataFrame: Dataset with categorical variables encoded.
    """
    # Convert categorical variables to dummy variables
    return pd.get_dummies(data, columns=['pv_module_type', 'hotspot', 'birddrop', 'soiling', 'junction_box'])

def feature_engineering_pipeline(filepath):
    """
    Run the entire feature engineering pipeline.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame: Dataset with engineered features.
    """
    data = load_data(filepath)
    data = handle_missing_values(data)
    data = create_new_features(data)
    data = encode_categorical_features(data)
    return data

if __name__ == "__main__":
    # Example usage
    filepath = "data/raw/pv_module_efficiency_dataset.csv"
    processed_data = feature_engineering_pipeline(filepath)
    processed_data.to_csv("data/processed/processed_data.csv", index=False)
    print("Feature engineering completed and data saved to 'data/processed/processed_data.csv'")