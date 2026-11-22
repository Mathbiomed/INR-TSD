import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional

from source.generate_season import *
from source.generate_trend import *

def main():
    
    # Set save path and create directory (auto-create if it doesn't exist)
    dataset_dir = Path("./dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base X-axis data (declare once outside the loop)
    x = np.linspace(0, 50, 100)
    
    # Load external trend data
    trend3_path = dataset_dir / "trend3.csv"
    try:
        df_trend3 = pd.read_csv(trend3_path)
    except FileNotFoundError:
        print(f"Warning: '{trend3_path}' not found. Trend type 3 will raise an error.")
        df_trend3 = None

    # Initialize figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharey=True)

    # Loop for data generation and saving
    for season_type in [1, 2, 3]:
        for trend_type in [1, 2, 3]:
            
            y_season = generate_season(x, season_type)
            y_trend = generate_trend(x, trend_type, df=df_trend3)
            y_total = y_season + y_trend
            
            # Calculate subplot index and plot the data
            ax = axes[season_type - 1, trend_type - 1]
            ax.plot(x, y_total, '-o', markersize=4)
            ax.set_title(f"Season: {season_type} | Trend: {trend_type}", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Create DataFrame to save as CSV
            df_out = pd.DataFrame({
                "time": x,
                "value": y_total,
                "trend": y_trend,
                "seasonality": y_season
            })
            
            filename = dataset_dir / f"season{season_type}_trend{trend_type}.csv"
            df_out.to_csv(filename, index=False)

    # Adjust overall graph layout and save
    plt.tight_layout()
    fig.savefig(dataset_dir / "generated_dataset.eps", format="eps", dpi=300, bbox_inches="tight")
    fig.savefig(dataset_dir / "generated_dataset.png", format="png", dpi=300, bbox_inches="tight")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()