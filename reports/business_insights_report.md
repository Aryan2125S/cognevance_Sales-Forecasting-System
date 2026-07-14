# Sales Forecasting System - Business Insights Report

**Author:** Vinit Sahani  
**Role:** Senior Data Scientist & Project Reviewer (Internship Submission)  
**Date:** July 12, 2026  

---

## 1. Executive Summary
This report presents a thorough analysis and forecasting of the furniture sales dataset spanning the years 2014 through 2017. Over this 48-month period, the business generated **$741,999.80** in total sales, but realized only **$18,451.27** in net profit, leading to an overall profit margin of **2.49%**. This low profit margin is primarily driven by significant losses in specific sub-categories and regions, combined with excessive discounting policies. A Linear Regression model was implemented with cyclical seasonal variables, achieving an R² score of **0.4784** on out-of-sample testing, and was used to forecast future demand to support data-driven inventory and pricing adjustments.

---

## 2. Dataset Overview
The dataset contains transaction-level historical sales records for the **Furniture** category of a retail store.
* **Total Transactions:** 2,121 records
* **Total Columns:** 21 variables
* **Historical Timeframe:** January 6, 2014, to December 30, 2017
* **Primary Features:** Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name, Segment, Country, City, State, Postal Code, Region, Product ID, Category, Sub-Category, Product Name, Sales, Quantity, Discount, Profit.
* **Target Forecasting Variable:** Sales (aggregated monthly)
* **Time Index Variable:** Order Date

---

## 3. Data Quality
The raw dataset was cleaned and processed without loss of transaction volume.
* **Missing Values:** No null values were present in the critical date (`Order Date`) or target (`Sales`) columns.
* **Duplicates:** No duplicate rows were identified.
* **Encoding:** Standard UTF-8 decoding encountered special characters in the text fields (e.g. Product Name). The file was loaded using robust `latin1` and `cp1252` encoding to prevent crashes.
* **Date Formats:** Order Date and Ship Date columns were parsed to datetime formats. Sorting was performed chronologically.

---

## 4. Historical Sales Performance
Historically, the store has experienced steady revenue growth, but profit has not scaled proportionally:
* **Total Revenue:** $741,999.80
* **Total Profit:** $18,451.27
* **Total Orders:** 1,764 unique transactions
* **Total Quantity Sold:** 8,028 items
* **Average Order Value (AOV):** $420.63

---

## 5. Monthly Sales Trends
A monthly aggregation reveals clear annual seasonality:
* **Holiday Peaks:** November and December show dramatic spikes in sales due to Q4 holiday shopping and end-of-year corporate budgets.
* **Post-Holiday Slumps:** Sales drop to their lowest levels in January and February.
* **Mid-Year Recovery:** Minor sales increases occur in September, followed by a stabilizing period in October.

---

## 6. Yearly Sales Trends
Year-over-year revenue demonstrates a clear upward trend:
* **2014:** Sales: **$157,192.85** | Profit: **$5,457.73** | Margins: **3.47%**
* **2015:** Sales: **$170,518.24** | Profit: **$3,015.20** | Margins: **1.77%**
* **2016:** Sales: **$198,901.44** | Profit: **$6,959.95** | Margins: **3.50%**
* **2017:** Sales: **$215,387.27** | Profit: **$3,018.39** | Margins: **1.40%**

> [!WARNING]
> While sales increased by **37.02%** from 2014 to 2017, net profit actually *decreased* by **44.70%** in the same period, indicating a major cost, discounting, or margin erosion problem.

---

## 7. Regional Performance
Performance varies heavily across the four defined geographic regions:
* **West:** Sales: **$252,612.74** | Profit: **$11,504.95** | Margin: **4.55%** (Strongest region)
* **East:** Sales: **$208,291.20** | Profit: **$3,046.17** | Margin: **1.46%**
* **South:** Sales: **$117,298.68** | Profit: **$6,771.21** | Margin: **5.77%** (Most efficient margin)
* **Central:** Sales: **$163,797.16** | Profit: **-$2,871.05** | Margin: **-1.75%** (Severe loss-maker)

---

## 8. Segment Performance
Customer segment analysis indicates that corporate transactions are highly profitable:
* **Consumer:** Sales: **$391,049.31** | Profit: **$6,991.08** | Margin: **1.79%**
* **Corporate:** Sales: **$229,019.79** | Profit: **$7,584.82** | Margin: **3.31%**
* **Home Office:** Sales: **$121,930.70** | Profit: **$3,875.38** | Margin: **3.18%**

---

## 9. Sub-Category Performance
Since the dataset is restricted to the **Furniture** category, we inspect the four sub-categories:
1. **Chairs:** Sales: **$328,449.10** | Profit: **$26,590.17** (Highly profitable anchor segment)
2. **Tables:** Sales: **$206,965.53** | Profit: **-$17,725.48** (Massive drain on margins)
3. **Bookcases:** Sales: **$114,880.00** | Profit: **-$3,472.56** (Minor loss-maker)
4. **Furnishings:** Sales: **$91,705.16** | Profit: **$13,059.14** (High profitability relative to volume)

---

## 10. Profitability Analysis
The primary culprit behind low overall profits is **discounting**.
* **Discount Impact:** Average transaction profits remain positive for discounts up to **20%**.
* **Erosion Curve:** Transactions with discounts of **30% or higher** generate massive, compounding losses. Tables and Bookcases are frequently subjected to high promotional discount rates (often 30% to 50%), which explains their negative net profit contribution.

---

## 11. Forecasting Methodology
We aggregate transactional records into monthly total sales, producing **48 data points** (Jan 2014 to Dec 2017).
* **Split Type:** Chronological (First 80% of months for training, last 20% for testing) to prevent target leakage.
* **Features:**
  1. `Time Index` (Linear trend)
  2. `Month Sin` (Seasonal sine component)
  3. `Month Cos` (Seasonal cosine component)
* **Model Type:** Scikit-Learn `LinearRegression`

---

## 12. Model Performance
Model evaluations on the chronological test set are:
* **Mean Absolute Error (MAE):** $5,438.61 (average absolute monthly prediction error)
* **Root Mean Squared Error (RMSE):** $6,483.93
* **R² Score:** 0.4784 (the model explains 47.84% of the monthly sales variance)

These metrics represent a stable, robust baseline suitable for a student project.

---

## 13. Future Sales Outlook
For a 6-month forecast horizon (January 2018 to June 2018):
* **Demand Trend:** The model forecasts an overall upward trend in sales compared to the recent historical average.
* **Expected Seasonality:** Sales are projected to decline in Jan/Feb 2018 (matching historical holiday slumps) before climbing steadily through spring.

---

## 14. Business Recommendations
1. **Discontinue or Reprice Tables:** Implement immediate price controls or reduce shipping/manufacturing costs on tables. Tables are draining **$17,725.48** in profit.
2. **Cap Discounts at 20%:** Establish corporate policies preventing retail representatives from applying discounts above 20% on furniture items.
3. **Target Corporate Sales:** Allocate marketing budgets toward the **Corporate** segment, which yields the highest margin (3.31%).
4. **Optimize Central Region Logistics:** Overhaul distribution and sales processes in the Central region to turn the **-$2,871.05** loss into a profit.
5. **Implement Seasonal Staffing:** Increase warehouse and shipping staff in September-October to handle the Q4 surge.

---

## 15. Limitations
* **Aggregation Level:** Monthly aggregation reduces noise but provides only 48 data points for training.
* **Linear Assumptions:** Linear Regression assumes linear trend growth, which may not capture sudden economic shifts or supply constraints.
* **External Factors:** The model does not incorporate marketing spend, competitor pricing, or inventory stock-outs.

---

## 16. Conclusion
This Sales Forecasting System successfully cleans historical furniture transactions, details structural margin problems (Tables and Central region discounting), and provides a reliable monthly demand forecasting baseline. Implementing the recommendations above will assist in turning low margins into double-digit profitability.
