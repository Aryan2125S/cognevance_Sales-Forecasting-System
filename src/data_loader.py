import os
from pathlib import Path
import pandas as pd

def get_dataset_path() -> Path:
    """
    Returns the absolute Path to the stores sales forecasting CSV dataset.
    """
    # This file is in src/, so the project root is parent directory
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data" / "stores_sales_forecasting.csv"

def load_data() -> pd.DataFrame:
    """
    Loads the sales forecasting dataset with robust encoding handling.
    Tries standard UTF-8 first and falls back to latin1 if required.
    
    Returns:
        pd.DataFrame: Loaded dataset
    Raises:
        FileNotFoundError: If the dataset file is missing.
    """
    path = get_dataset_path()
    if not path.exists():
        raise FileNotFoundError(
            f"The sales forecasting dataset is missing! "
            f"Expected file at: '{path}'\n"
            f"Please verify that 'stores_sales_forecasting.csv' is in your 'data/' folder."
        )
    
    try:
        # Try loading with standard UTF-8
        df = pd.read_csv(path, encoding="utf-8")
        return df
    except (UnicodeDecodeError, UnicodeError):
        # Fall back to latin1 if UTF-8 fails
        df = pd.read_csv(path, encoding="latin1")
        return df
