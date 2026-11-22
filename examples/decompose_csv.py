"""Fit one named time/value CSV; run from the repository root."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_SEED, MODEL_DIR
from source.environment import configure_environment, seed_data_pipeline
from source.train import decompose_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose an output path different from the input.")
    frame = pd.read_csv(args.input)
    if not {"time", "value"}.issubset(frame.columns):
        parser.error("Input must have named time and value columns.")
    x = pd.to_numeric(frame["time"], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        parser.error("Provide at least three finite observations.")
    if not np.all(np.diff(x) > 0) or np.ptp(y) == 0:
        parser.error("Times must strictly increase and values must not be constant.")
    configure_environment()
    seed_data_pipeline(DATA_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trend, season = decompose_series(x - x[0], y)
    pd.DataFrame({
        "time": x,
        "actual_value": y,
        "predicted_trend": trend,
        "predicted_seasonality": season,
    }).to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
