# Sales Forecasting System

An interactive, end-to-end Data Science and Machine Learning web application built to analyze historical furniture transactions and forecast future monthly sales.

---

## Project Overview
This project is an intermediate Level 2 Data Science and Analytics internship submission. It ingests historical retail transaction records, cleans and preprocesses the dataset, conducts interactive Exploratory Data Analysis (EDA) across various business dimensions, aggregates transaction-level sales into monthly time-series intervals, and implements a Linear Regression model with cyclical trigonometric seasonal features to predict future sales. Finally, the system is wrapped in a Streamlit dashboard that provides dynamic business insights and actionable suggestions.

## Problem Statement
Retail stores often struggle to optimize inventory due to demand variability. Under-stocking results in missed revenue, while over-stocking locks up capital and increases warehousing costs. By building a reliable demand forecasting system utilizing historical order records, businesses can anticipate sales fluctuations, prepare supply chains for seasonal surges, and identify operational leaks (e.g. loss-making products or inefficient discounting strategies).

## Objectives
* Develop a robust data loading pipeline that handles formatting and encoding issues gracefully.
* Clean raw transactional data and engineer time-based features (shipping durations, quarters, cyclical sine/cosine months).
* Perform multi-dimensional analysis on segments, regions, sub-categories, and discounts.
* Train an explainable machine learning forecasting model using a chronologically correct splitting strategy.
* Build a polished, interactive Streamlit dashboard deployable to the Streamlit Community Cloud.
* Produce a detailed, calculated business insights report with practical suggestions.

---

## Dataset
The system operates on the `stores_sales_forecasting.csv` dataset located in `data/stores_sales_forecasting.csv`.
* **Total Transactions:** 2,121 records
* **Total Fields:** 21 columns
* **Historical Period:** 2014 to 2017 (~48 months)
* **Scope:** Focused exclusively on the **Furniture** category.
* **Target Forecasting Column:** `Sales`
* **Time Column:** `Order Date`
* **Additional Columns:** Row ID, Order ID, Ship Date, Ship Mode, Customer ID, Customer Name, Segment, Country, City, State, Postal Code, Region, Product ID, Sub-Category, Product Name, Quantity, Discount, Profit.

---

## Key Features
* **Interactive KPI Grid:** Dynamic cards displaying Total Sales, Profit, Orders, Quantity, and Profit Margin.
* **Descriptive Analysis Tabs:** Filterable line, bar, and pie charts analyzing monthly trend lines, regional performances, customer segments, top products, and discount structures.
* **Out-of-Sample Evaluation:** Out-of-sample visual charts comparing predicted sales against actual sales on the test set.
* **Future Forecast horizon slider:** A slider letting users dynamically forecast future monthly sales from 3 to 12 months ahead.
* **Trigonometric cyclical Seasonality:** Utilizes mathematical cyclical mapping (Sin/Cos) to capture quarterly and annual fluctuations.
* **Dynamic Business Insights:** Computes highest/lowest performance margins and detects loss-making categories automatically.

---

## Project Workflow
```
Data Collection 
  → Data Cleaning 
  → Exploratory Data Analysis 
  → Monthly Aggregation 
  → Feature Engineering 
  → Chronological Train-Test Split 
  → Linear Regression 
  → Model Evaluation 
  → Future Forecasting 
  → Business Insights 
  → Streamlit Dashboard
```

---

## Technologies Used
* **Python** (Core Logic)
* **Pandas** (Data Wrangling & Time-Series Preparation)
* **NumPy** (Mathematical & Cyclical Operations)
* **Matplotlib & Seaborn** (Data Visualizations)
* **Scikit-Learn** (Linear Regression Modeling & Metrics)
* **Streamlit** (Interactive Dashboard UI)

---

## Project Structure
```text
cognevance_Sales-Forecasting-System/  (Repository Root)
│
├── app.py                     # Streamlit Main Dashboard Entry Point
├── requirements.txt           # Dependency Requirements
├── README.md                  # Project Documentation
├── .gitignore                 # Version Control Ignore Rules
│
├── data/
│   └── stores_sales_forecasting.csv    # Extracted Transaction Dataset
│
├── notebooks/
│   └── sales_forecasting_analysis.ipynb # Step-by-Step Jupyter Notebook
│
├── src/
│   ├── __init__.py            # Python Package Initialization
│   ├── data_loader.py         # Robust Data Loading & Encoding Fallback
│   ├── data_preprocessing.py  # Cleaning, Sorting & Time Feature Engineering
│   ├── analyzer.py            # Aggregate Statistics & KPI Computations
│   └── forecasting.py         # Time-Series ML Modeling & Future Projections
│
└── reports/
    └── business_insights_report.md     # Detailed Calculated Business Report
```

---

## Data Preprocessing
1. **Datetime Conversion:** Parses `Order Date` and `Ship Date` into datetime objects.
2. **Chronological Sorting:** Sorts all records by `Order Date` to ensure time continuity.
3. **Data Type Correction:** Ensures Sales, Profit, Quantity, and Discount are parsed as float/int values.
4. **shipping Duration:** Computes duration in days (`Ship Date` - `Order Date`) for order fulfillment analysis.
5. **Feature Creation:** Extracts Year, Month, Month Number, Quarter, and Year-Month (e.g., `2014-01`) columns.

---

## Exploratory Data Analysis
Through exploratory analysis, we identified the following structural insights:
* **Sub-Category Breakdown:** **Chairs** is the largest revenue driver (**$328,449.10** sales).
* **Margin Drains:** **Tables** generates **$206,965.53** in sales but incurs a massive net loss of **-$17,725.48**.
* **Discount Erosion:** Discounts of **30% or higher** consistently erode profit margins into negative percentages. Cap guidelines should restrict discounts to under 20%.
* **Regional Risk:** The **Central** region is net loss-making (**-$2,871.05**), whereas the **West** and **South** regions are highly profitable.

---

## Forecasting Methodology
1. **Monthly Aggregation:** Transaction-level sales are aggregated by calendar months (48 total observation periods).
2. **Feature Engineering:**
   - **Trend Feature (`Time Index`):** Sequential integers (1, 2, 3...) to capture linear growth.
   - **cyclical Seasonal Features (`Month Sin`, `Month Cos`):** Since January (Month 1) and December (Month 12) are adjacent calendar periods, treating Month only as a simple linear variable is mathematically incorrect. We map the calendar months to a 2D circular wave using:
     $$\text{Month Sin} = \sin\left(\frac{2 \pi \times \text{Month}}{12}\right)$$
     $$\text{Month Cos} = \cos\left(\frac{2 \pi \times \text{Month}}{12}\right)$$
3. **Chronological Train-Test Split:** Splitting time-series data using random shuffles violates chronological dependency, introducing target leakage. We split the data chronologically:
   - **Training Set:** First **80%** of historical months (38 observations)
   - **Testing Set:** Latest **20%** of historical months (10 observations)

---

## Model Evaluation
The Linear Regression model trained on cyclical features achieved the following metrics on the out-of-sample chronological test set:
* **Mean Absolute Error (MAE):** $5,438.61  
* **Root Mean Squared Error (RMSE):** $6,483.93  
* **R² Score:** 0.4784 (Explains 47.84% of the variance in monthly furniture sales)

These metrics represent a stable, robust baseline suitable for a student project.

---

## Streamlit Dashboard
The interface is structured into six pages:
1. **Home:** High-level project summary, date range indicators, and total sales, profit, orders, quantity, and profit margin cards.
2. **Dataset Explorer:** Transaction previews, summary statistics tables, and schema detail descriptions.
3. **Sales Analysis:** Interactive filters (Year, Region, Segment) with trend lines, sub-category sales/profit bars, segment pie charts, discount curve lines, and top products.
4. **Sales Forecasting:** Displays model performance metrics, out-of-sample prediction plots, a forecasting horizon slider (3-12 months), and prediction outputs.
5. **Business Insights:** Dynamic insights reflecting highest/lowest sales periods, regional and segment alerts, and a summary list of operational business recommendations.
6. **About Project:** Project metadata, intern information, and technical details.

---

## Installation
Ensure Python 3.8+ is installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

## Run Locally
To run the Streamlit app locally, run the following command in your terminal:
```bash
streamlit run app.py
```

## Deployment on Streamlit Community Cloud
To host this system on Streamlit Community Cloud:
1. Push this workspace folder to a public GitHub repository.
2. Log in to [Streamlit Share](https://share.streamlit.io/).
3. Click **New app**, select your repository, branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit will automatically read `requirements.txt` and set up the hosting environment.

---

## Limitations
* **Sample Size:** Monthly aggregation yields 48 historical data points.
* **Extrapolation:** Linear regression extrapolates trends linearly, which may over-estimate growth over long horizons.
* **Internal Factors Only:** Does not account for marketing campaigns, macroeconomic changes, or stock outs.

---

## Future Improvements
* Implement a recursive forecasting model that supports lagged sales variables.
* Incorporate confidence intervals based on residual distributions.
* Collect additional external indicators (e.g. ad spend) to train a multivariate regression model.

---

## Author
* **Name:** Vinit Sahani  
* **Role:** Data Science Intern  
* **Internship Level:** Level 2 Intermediate  
