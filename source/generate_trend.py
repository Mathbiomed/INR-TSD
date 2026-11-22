import numpy as np
import pandas as pd

from typing import Optional

def generate_trend(x: np.ndarray, trend_type: int, df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Generates or loads trend data based on the given trend_type.
    
    Args:
        x (np.ndarray): Time steps
        trend_type (int): 1, 2, or 3
        df (pd.DataFrame, optional): DataFrame to reference when trend_type is 3
        
    Returns:
        np.ndarray: Calculated trend values
    """
    if trend_type == 1:
        return 0.2 * x
    elif trend_type == 2:
        return np.exp(-0.05 * x + 2)
    elif trend_type == 3:
        if df is None:
            raise ValueError("A DataFrame 'df' is required when trend_type is 3.")
        return df.iloc[:, 2].values
    else:
        raise ValueError(f"Invalid trend_type: {trend_type}. Must be 1, 2, or 3.")