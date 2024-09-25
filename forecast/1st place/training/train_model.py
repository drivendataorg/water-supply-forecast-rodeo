#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[0]))

from loguru import logger

from train_yearly_model import train_yearly_model
from train_monthly_model import train_monthly_model


def main(model_dir: Path,
         preprocessed_dir: Path):
    logger.info(f"Beginning model training.")

    train_features = pd.read_csv(preprocessed_dir / "train_features.csv")

    # Null out the monthly naturalized flow on the 1st of the month to simulate forecast scenario
    # Only want to null out the month_volume_label for the month that corresponds to the month prior to the issue_date
    null_monthly_nat_flow = True
    logger.info(f"null_monthly_nat_flow={null_monthly_nat_flow}")

    train_yearly_model(model_dir, train_features, null_monthly_nat_flow)
    train_monthly_model(model_dir, train_features, null_monthly_nat_flow)


if __name__ == "__main__":
    from pathlib import Path

    import sys
    import warnings
    import numpy as np
    import pandas as pd

    sys.path.append(str(Path(__file__).parent.resolve()))

    MODEL_DIR = Path.cwd() / "training/models"
    PREPROCESSED_DIR = Path.cwd() / "training/preprocessed_data"

    warnings.filterwarnings('ignore')

    main(MODEL_DIR, PREPROCESSED_DIR)
