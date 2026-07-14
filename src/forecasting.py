import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any

def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactional sales into a monthly time series.
    Ensures that all months in the date range are represented chronologically.
    
    Args:
        df (pd.DataFrame): Preprocessed transactional DataFrame.
    Returns:
        pd.DataFrame: Monthly time-series DataFrame with 'Order Date' and 'Sales'.
    """
    # Group by Year-Month and sum Sales
    df_monthly = df.groupby(df["Order Date"].dt.to_period("M")).agg({"Sales": "sum"}).reset_index()
    # Convert period back to timestamp (representing start of month)
    df_monthly["Order Date"] = df_monthly["Order Date"].dt.to_timestamp()
    
    # Sort chronologically
    df_monthly = df_monthly.sort_values(by="Order Date").reset_index(drop=True)
    
    # Reindex to handle any potential missing months in the sequence
    min_date = df_monthly["Order Date"].min()
    max_date = df_monthly["Order Date"].max()
    
    if pd.isnull(min_date) or pd.isnull(max_date):
        return df_monthly
        
    full_range = pd.date_range(start=min_date, end=max_date, freq="MS")
    df_monthly = df_monthly.set_index("Order Date").reindex(full_range, fill_value=0.0).reset_index()
    df_monthly = df_monthly.rename(columns={"index": "Order Date"})
    
    return df_monthly

def engineer_features(df_monthly: pd.DataFrame, start_time_index: int = 1) -> pd.DataFrame:
    """
    Engineers trend and cyclical seasonal features for the monthly sales.
    
    Features created:
    1. Time Index (linear trend)
    2. Month Sin & Month Cos (captures 12-month seasonality cyclically)
    
    Args:
        df_monthly (pd.DataFrame): Monthly time-series DataFrame.
        start_time_index (int): Starting index for the Time Index feature.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df_features = df_monthly.copy()
    
    # 1. Time Index
    n_rows = len(df_features)
    df_features["Time Index"] = np.arange(start_time_index, start_time_index + n_rows)
    
    # 2. Cyclical Monthly Features
    months = df_features["Order Date"].dt.month
    df_features["Month Sin"] = np.sin(2 * np.pi * months / 12)
    df_features["Month Cos"] = np.cos(2 * np.pi * months / 12)
    
    return df_features

def train_forecast_model(df_monthly: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[LinearRegression, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Trains a Linear Regression model using chronological train-test splitting.
    
    Args:
        df_monthly (pd.DataFrame): Monthly time-series DataFrame.
        train_ratio (float): Chronological split ratio.
    Returns:
        Tuple containing:
            - LinearRegression: Trained model.
            - pd.DataFrame: Train set with features.
            - pd.DataFrame: Test set with features and predictions.
            - Dict[str, float]: Evaluation metrics (MAE, RMSE, R²).
    """
    # 1. Engineer features
    df_features = engineer_features(df_monthly)
    
    # 2. Chronological split
    n = len(df_features)
    train_size = int(n * train_ratio)
    
    train_data = df_features.iloc[:train_size].copy()
    test_data = df_features.iloc[train_size:].copy()
    
    # 3. Model Training
    feature_cols = ["Time Index", "Month Sin", "Month Cos"]
    X_train = train_data[feature_cols]
    y_train = train_data["Sales"]
    X_test = test_data[feature_cols]
    y_test = test_data["Sales"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Model Predictions
    train_data["Predicted_Sales"] = model.predict(X_train)
    test_data["Predicted_Sales"] = model.predict(X_test)
    
    # 5. Evaluation Metrics
    y_pred_test = test_data["Predicted_Sales"]
    
    mae = float(mean_absolute_error(y_test, y_pred_test))
    # Calculate RMSE in a version-safe manner
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    
    return model, train_data, test_data, metrics

def generate_future_forecast(model: LinearRegression, last_date: pd.Timestamp, last_time_index: int, horizon: int = 6) -> pd.DataFrame:
    """
    Forecasts future sales for a specified monthly horizon.
    
    Args:
        model (LinearRegression): Trained forecasting model.
        last_date (pd.Timestamp): Date of the last historical data point.
        last_time_index (int): Time index of the last historical data point.
        horizon (int): Number of months to forecast into the future.
    Returns:
        pd.DataFrame: Forecast results containing 'Order Date', 'Time Index',
                      'Month Sin', 'Month Cos', and predicted 'Sales'.
    """
    # 1. Generate future dates (monthly offsets from the last date)
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
    
    # 2. Build feature matrix
    future_df = pd.DataFrame({"Order Date": future_dates})
    future_df["Time Index"] = np.arange(last_time_index + 1, last_time_index + 1 + horizon)
    
    months = future_df["Order Date"].dt.month
    future_df["Month Sin"] = np.sin(2 * np.pi * months / 12)
    future_df["Month Cos"] = np.cos(2 * np.pi * months / 12)
    
    # 3. Predict future values
    feature_cols = ["Time Index", "Month Sin", "Month Cos"]
    future_df["Sales"] = model.predict(future_df[feature_cols])
    
    # Keep output format clean
    return future_df
