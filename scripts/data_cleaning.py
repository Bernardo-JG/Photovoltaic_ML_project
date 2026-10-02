"""
Este documento "data_cleaning.py" tem funções auxiliares para carregar e limpar o dataset. 
Servem para remover inconsistências.
"""

import pandas as pd

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
    data_filled = data.fillna(data.median())
    
    # Optionally, drop rows with too many missing values
    # threshold = len(data.columns) * 0.5  # If more than 50% of the values in a row are missing, drop the row
    # data_cleaned = data.dropna(thresh=threshold)
    
    return data_filled

def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.

    Parameters:
    data (DataFrame): Input dataset.

    Returns:
    DataFrame: Dataset with duplicate rows removed.
    """
    return data.drop_duplicates()

def convert_data_types(data):
    """
    Convert data types of certain columns if necessary.

    Parameters:
    data (DataFrame): Input dataset.

    Returns:
    DataFrame: Dataset with converted data types.
    """
    # Example: Convert a column to datetime if it contains date information
    # data['date_column'] = pd.to_datetime(data['date_column'])
    
    return data

def clean_data(filepath):
    """
    Run the entire data cleaning pipeline.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame: Cleaned dataset.
    """
    data = load_data(filepath)
    data = handle_missing_values(data)
    data = remove_duplicates(data)
    data = convert_data_types(data)
    return data

if __name__ == "__main__":
    # Example usage
    filepath = "data/raw/pv_module_efficiency_dataset.csv"
    cleaned_data = clean_data(filepath)
    cleaned_data.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Data cleaning completed and data saved to 'data/processed/cleaned_data.csv'")