import pandas as pd
import numpy as np
from typing import Dict, Tuple

def get_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes key performance indicators (KPIs) from the preprocessed DataFrame.
    """
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    total_quantity = int(df["Quantity"].sum())
    total_orders = int(df["Order ID"].nunique())
    
    # Avoid division by zero
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0.0
    avg_order_value = (total_sales / total_orders) if total_orders > 0 else 0.0
    
    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_quantity": total_quantity,
        "total_orders": total_orders,
        "profit_margin": profit_margin,
        "avg_order_value": avg_order_value
    }

def get_monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates Sales and Profit on a monthly basis.
    """
    trend = df.groupby("Year-Month").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Order ID": "nunique"
    }).rename(columns={"Order ID": "Orders"}).reset_index()
    
    # Ensure chronological order
    trend = trend.sort_values("Year-Month").reset_index(drop=True)
    return trend

def get_yearly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates Sales and Profit on a yearly basis.
    """
    return df.groupby("Year").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Order ID": "nunique"
    }).rename(columns={"Order ID": "Orders"}).reset_index()

def get_quarterly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates Sales and Profit on a quarterly basis.
    """
    return df.groupby("Quarter").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Order ID": "nunique"
    }).rename(columns={"Order ID": "Orders"}).reset_index().sort_values("Quarter").reset_index(drop=True)

def get_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates performance metrics by Region.
    """
    perf = df.groupby("Region").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum"
    }).reset_index()
    return perf.sort_values(by="Sales", ascending=False).reset_index(drop=True)

def get_segment_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates performance metrics by Customer Segment.
    """
    perf = df.groupby("Segment").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum"
    }).reset_index()
    return perf.sort_values(by="Sales", ascending=False).reset_index(drop=True)

def get_subcategory_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates performance metrics by Sub-Category.
    """
    perf = df.groupby("Sub-Category").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum"
    }).reset_index()
    return perf.sort_values(by="Sales", ascending=False).reset_index(drop=True)

def get_top_products(df: pd.DataFrame, metric: str = "Sales", n: int = 10) -> pd.DataFrame:
    """
    Identifies top products based on Sales or Profit.
    """
    if metric not in ["Sales", "Profit"]:
        metric = "Sales"
        
    perf = df.groupby("Product Name").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum"
    }).reset_index()
    return perf.sort_values(by=metric, ascending=False).head(n).reset_index(drop=True)

def get_discount_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the relationship between Discount and Profit Margin.
    """
    # Group by standard discount rates present in dataset
    perf = df.groupby("Discount").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "mean",
        "Row ID": "count"
    }).rename(columns={"Row ID": "Transactions"}).reset_index()
    
    # Calculate margin per discount level
    perf["Profit Margin (%)"] = (perf["Profit"] / perf["Sales"] * 100).round(2)
    return perf.sort_values("Discount").reset_index(drop=True)
