import pandas as pd
import torch

def get_y_values(file_path: str, filenames: list[str]) -> torch.Tensor:
    """
    Loads the y values from a CSV file containing material properties.
    The CSV file should have a column named 'E_eff' which contains the 
    effective modulus values
    Args:
        file_path (str): Path to the CSV file containing material properties.
    Returns:
        torch.Tensor: A tensor containing the effective modulus values.
    """
    
    # Load y values from material_property.csv

    df = pd.read_csv(file_path)
    df = df.sort_values(by='filename')
    df = df.set_index('filename').loc[filenames].reset_index()
    if df.empty:
        raise ValueError("No matching filenames found in the CSV file.")    
    # Transform the DataFrame to tensors
    if 'E_eff' not in df.columns:
        raise ValueError("CSV file must contain 'E_eff' column.")
    y = df['E_eff'].values
    y = torch.tensor(y, dtype=torch.float32)
    return y