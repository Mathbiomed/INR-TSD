import numpy as np

from typing import Optional

def generate_season(x: np.ndarray, season_type: int) -> np.ndarray:
    """
    Generates seasonality data based on the given season_type.
    
    Args:
        x (np.ndarray): Time steps
        season_type (int): 1, 2, or 3
        
    Returns:
        np.ndarray: Calculated seasonality values
    """
    if season_type == 1:
        return np.sin(2 * x)
    elif season_type == 2:
        return np.sin(x) + np.sin(2 * x)
    elif season_type == 3:
        return np.sin(0.5 * x) + np.sin(np.sqrt(2) * x)
    else:
        raise ValueError(f"Invalid season_type: {season_type}. Must be 1, 2, or 3.")

