import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the transaction-level sales dataset.
    
    Tasks performed:
    1. Converts 'Order Date' and 'Ship Date' to datetime objects.
    2. Sorts the transactions chronologically by 'Order Date'.
    3. Handles missing or invalid numeric values in Sales, Profit, Quantity, and Discount.
    4. Computes useful time-based columns: Year, Month, Month Number, Quarter, and Year-Month.
    5. Computes 'Shipping Duration' (Ship Date - Order Date) in days.
    
    Args:
        df (pd.DataFrame): Raw transactional DataFrame.
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    # Create a copy to prevent warning on assignments
    df = df.copy()
    
    # Convert dates to datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    
    # Sort chronologically by Order Date
    df = df.sort_values(by="Order Date").reset_index(drop=True)
    
    # Ensure numeric columns are formatted properly
    numeric_cols = ["Sales", "Profit", "Quantity", "Discount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Drop records with invalid/null Sales or Order Date values
    df = df.dropna(subset=["Order Date", "Sales"])
    df = df[df["Sales"] > 0]
    
    # Feature Engineering
    df["Year"] = df["Order Date"].dt.year
    df["Month Number"] = df["Order Date"].dt.month
    df["Month"] = df["Order Date"].dt.strftime("%B")
    df["Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)
    df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
    
    # Order shipping duration in days
    df["Shipping Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days
    
    return df
