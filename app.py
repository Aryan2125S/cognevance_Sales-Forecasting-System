import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Set Streamlit page configuration first
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
import src.analyzer as analyzer
import src.forecasting as forecasting

# Set matplotlib style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.family'] = 'sans-serif'

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    /* Styling headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    /* Card design */
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.02);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Dark mode adjustments if detected */
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background-color: #1F2937;
            border-color: #374151;
        }
        .kpi-value {
            color: #F9FAFB;
        }
        .kpi-label {
            color: #9CA3AF;
        }
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading in Streamlit
@st.cache_data
def get_dataset():
    try:
        raw_df = load_data()
        cleaned_df = preprocess_data(raw_df)
        return cleaned_df, None
    except Exception as e:
        return None, str(e)

# Load data
df, error_msg = get_dataset()

if error_msg:
    st.error(f"⚠️ Error initializing data: {error_msg}")
    st.info("Please make sure the dataset is placed at `data/stores_sales_forecasting.csv` relative to the project root directory.")
    st.stop()

# Sidebar Navigation System
st.sidebar.markdown("<h2 style='text-align: center; font-weight: 800;'>📊 Menu</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🔍 Dataset Explorer", "📈 Sales Analysis", "🔮 Sales Forecasting", "💡 Business Insights", "ℹ️ About Project"]
)

# Helper function to format currency
def format_currency(val: float) -> str:
    return f"${val:,.2f}"

# =====================================================================
# PAGE 1: HOME PAGE
# =====================================================================
if page == "🏠 Home":
    st.markdown("<div class='main-header'>Sales Forecasting System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>An interactive Data Science & Machine Learning platform for analyzing furniture sales and forecasting demand.</div>", unsafe_allow_html=True)
    
    # Calculate KPIs
    kpi_dict = analyzer.get_kpis(df)
    min_date = df["Order Date"].min().strftime('%B %Y')
    max_date = df["Order Date"].max().strftime('%B %Y')
    
    # Hero description
    st.markdown(f"""
    Welcome to the **Sales Forecasting System** dashboard. This application aggregates historical transaction-level 
    furniture sales data and uses a Time-Series forecasting model (**Linear Regression with Cyclical Seasonal Features**) 
    to project future sales.
    
    * **Historical Range**: {min_date} to {max_date} (~48 months)
    * **Historical Transactions**: {len(df):,} orders
    """)
    
    st.write("---")
    
    # KPI Grid
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{format_currency(kpi_dict['total_sales'])}</div>
            <div class="kpi-label">Total Sales</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{format_currency(kpi_dict['total_profit'])}</div>
            <div class="kpi-label">Total Profit</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpi_dict['total_orders']:,}</div>
            <div class="kpi-label">Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpi_dict['total_quantity']:,}</div>
            <div class="kpi-label">Total Items Sold</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        color = "#10B981" if kpi_dict['profit_margin'] >= 0 else "#EF4444"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: {color};">{kpi_dict['profit_margin']:.2f}%</div>
            <div class="kpi-label">Profit Margin</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    # Brief insights or summary details
    st.subheader("Key Findings Quick-View")
    c_left, c_right = st.columns(2)
    
    with c_left:
        subcat_perf = analyzer.get_subcategory_performance(df)
        best_sub = subcat_perf.iloc[0]
        worst_sub = subcat_perf.iloc[-1]
        
        st.markdown(f"""
        * **Top Performing Category Segment**: Chairs (Sales: **{format_currency(best_sub['Sales'])}**, Profit: **{format_currency(best_sub['Profit'])}**)
        * **Low Profitability Alert**: Bookcases and Tables have high sales volume but consistently low profits (Tables show net loss).
        """)
        
    with c_right:
        st.markdown("""
        * **Seasonal Trends**: Peak sales occur in November and December due to Q4 holidays.
        * **Regional Strength**: The **West** region generates the highest sales and profit margin, followed closely by the **East**.
        """)

# =====================================================================
# PAGE 2: DATASET EXPLORER PAGE
# =====================================================================
elif page == "🔍 Dataset Explorer":
    st.markdown("<div class='main-header'>Dataset Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Detailed view of the historical furniture transactional records.</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Summary Statistics", "🔍 Data Quality & Schema"])
    
    with tab1:
        st.subheader("Transactional Records Preview")
        st.write(f"Displaying first 100 of {len(df):,} preprocessed transactions. Column `Shipping Duration` calculates days between Order Date and Ship Date.")
        st.dataframe(df.head(100), use_container_width=True)
        
    with tab2:
        st.subheader("Numeric Summary Statistics")
        st.dataframe(df[["Sales", "Quantity", "Discount", "Profit", "Shipping Duration"]].describe(), use_container_width=True)
        
    with tab3:
        col_types = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.subheader("Dataset Schema Details")
        st.dataframe(col_types, use_container_width=True)
        
        st.subheader("Category Breakdown Notice")
        st.info("Category column has only 1 unique value: **Furniture**. Therefore, descriptive analysis focuses on **Sub-Category** levels.")

# =====================================================================
# PAGE 3: SALES ANALYSIS PAGE
# =====================================================================
elif page == "📈 Sales Analysis":
    st.markdown("<div class='main-header'>Sales & Business Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive insights of historical sales, profit, regions, customer segments, and discount structures.</div>", unsafe_allow_html=True)
    
    # Dynamic dashboard filters (Do not modify forecasting calculations)
    st.sidebar.markdown("### 🛠️ Descriptive Filters")
    years = ["All"] + list(df["Year"].unique().astype(str))
    selected_year = st.sidebar.selectbox("Filter by Year:", years)
    
    regions = ["All"] + list(df["Region"].unique())
    selected_region = st.sidebar.selectbox("Filter by Region:", regions)
    
    segments = ["All"] + list(df["Segment"].unique())
    selected_segment = st.sidebar.selectbox("Filter by Segment:", segments)
    
    # Filter dataset for analysis only
    analysis_df = df.copy()
    if selected_year != "All":
        analysis_df = analysis_df[analysis_df["Year"] == int(selected_year)]
    if selected_region != "All":
        analysis_df = analysis_df[analysis_df["Region"] == selected_region]
    if selected_segment != "All":
        analysis_df = analysis_df[analysis_df["Segment"] == selected_segment]
        
    if analysis_df.empty:
        st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
        st.stop()
        
    # KPI Grid for descriptive analysis
    kpi_dict = analyzer.get_kpis(analysis_df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Sales", format_currency(kpi_dict["total_sales"]))
    col2.metric("Filtered Profit", format_currency(kpi_dict["total_profit"]))
    col3.metric("Filtered Orders", f"{kpi_dict['total_orders']:,}")
    col4.metric("Profit Margin", f"{kpi_dict['profit_margin']:.2f}%")
    
    st.write("---")
    
    # Layout with columns for charts
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Monthly Sales & Profit Trend")
        monthly_trend = analyzer.get_monthly_sales_trend(analysis_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_trend["Year-Month"], monthly_trend["Sales"], label="Sales", marker='o', color="#4F46E5", linewidth=2.5)
        ax.plot(monthly_trend["Year-Month"], monthly_trend["Profit"], label="Profit", marker='x', color="#EF4444", linewidth=1.5)
        
        # Display monthly labels nicely
        n_ticks = len(monthly_trend)
        step = max(1, n_ticks // 8)
        ax.set_xticks(monthly_trend["Year-Month"][::step])
        ax.set_xticklabels(monthly_trend["Year-Month"][::step], rotation=45)
        
        ax.set_ylabel("USD ($)")
        ax.legend()
        st.pyplot(fig)
        
    with row1_col2:
        st.subheader("Performance by Sub-Category")
        subcat = analyzer.get_subcategory_performance(analysis_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(subcat))
        width = 0.35
        
        ax.bar(x - width/2, subcat["Sales"], width, label="Sales", color="#636EFA")
        ax.bar(x + width/2, subcat["Profit"], width, label="Profit", color="#EF553B")
        
        ax.set_xticks(x)
        ax.set_xticklabels(subcat["Sub-Category"])
        ax.set_ylabel("USD ($)")
        ax.legend()
        st.pyplot(fig)
        
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("Sales and Profit by Region")
        reg_perf = analyzer.get_regional_performance(analysis_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=reg_perf, x="Region", y="Sales", color="#06B6D4", label="Sales", ax=ax)
        # Overlay line plot for profit
        ax2 = ax.twinx()
        ax2.plot(reg_perf["Region"], reg_perf["Profit"], color="#B91C1C", marker="o", label="Profit", linewidth=2)
        ax2.set_ylabel("Profit ($)")
        ax.set_ylabel("Sales ($)")
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
        st.pyplot(fig)
        
    with row2_col2:
        st.subheader("Sales by Customer Segment")
        seg_perf = analyzer.get_segment_performance(analysis_df)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(seg_perf["Sales"], labels=seg_perf["Segment"], autopct='%1.1f%%', colors=["#4F46E5", "#3B82F6", "#93C5FD"], startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
    st.write("---")
    
    # Row 3: Discount Analysis and Top Products
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.subheader("Impact of Discounts on Profit Margin")
        disc_df = analyzer.get_discount_impact(analysis_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=disc_df, x="Discount", y="Profit Margin (%)", marker="o", color="#D97706", linewidth=2, ax=ax)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Discount Rate (e.g. 0.2 = 20%)")
        ax.set_ylabel("Profit Margin (%)")
        st.pyplot(fig)
        
    with row3_col2:
        st.subheader("Top 5 Selling Products")
        top_prod = analyzer.get_top_products(analysis_df, metric="Sales", n=5)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Shorten names for plotting
        short_names = [name[:30] + "..." if len(name) > 30 else name for name in top_prod["Product Name"]]
        sns.barplot(x=top_prod["Sales"], y=short_names, palette="viridis", ax=ax)
        ax.set_xlabel("Total Sales ($)")
        st.pyplot(fig)

# =====================================================================
# PAGE 4: SALES FORECASTING PAGE
# =====================================================================
elif page == "🔮 Sales Forecasting":
    st.markdown("<div class='main-header'>Sales Forecasting Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Monthly forecasting model with Linear Regression and cyclical seasonal indicators.</div>", unsafe_allow_html=True)
    
    # 1. Prepare Monthly Series
    monthly_series = forecasting.prepare_time_series(df)
    
    # 2. Train Model and Get Predictions & Metrics
    # Split: 80% train, 20% test chronologically.
    model, train_df, test_df, metrics = forecasting.train_forecast_model(monthly_series)
    
    # Model details card
    st.info("""
    **💡 Forecasting Pipeline Explanation:**
    1. **Monthly Aggregation**: Individual transactions are summed into a single monthly sales series.
    2. **Features**:
       - **Time Index**: A sequential number (1, 2, 3...) to capture the long-term upward or downward trend.
       - **cyclical seasonality**: `Month Sin` and `Month Cos` derived from `sin(2*pi*month/12)` and `cos(2*pi*month/12)` to model annual seasonal cycles without overfitting.
    3. **Chronological Splitting**: Shuffling time-series data creates target leakage. Instead, the model is trained on the first **80%** of calendar months and evaluated on the latest **20%** chronological months.
    """)
    
    st.subheader("1. Model Performance (Test Set Evaluation)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Mean Absolute Error (MAE)",
        value=f"${metrics['MAE']:,.2f}",
        help="The average absolute difference between the predicted and actual sales. Lower is better."
    )
    col2.metric(
        label="Root Mean Squared Error (RMSE)",
        value=f"${metrics['RMSE']:,.2f}",
        help="Penalizes larger forecast errors more heavily. Lower is better."
    )
    col3.metric(
        label="R² Score (Coefficient of Determination)",
        value=f"{metrics['R2']:.4f}",
        help="Proportion of the monthly sales variance explained by the model trend and seasonal components."
    )
    
    st.write("---")
    
    # Model Evaluation Plot
    st.subheader("2. Model Predictions vs Historical Actuals")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df["Order Date"], train_df["Sales"], label="Actual Sales (Train)", color="#1F2937", marker="o", linewidth=2)
    ax.plot(train_df["Order Date"], train_df["Predicted_Sales"], label="Predicted Sales (Train)", color="#636EFA", linestyle="--")
    
    ax.plot(test_df["Order Date"], test_df["Sales"], label="Actual Sales (Test)", color="#10B981", marker="o", linewidth=2)
    ax.plot(test_df["Order Date"], test_df["Predicted_Sales"], label="Predicted Sales (Test)", color="#EF4444", linestyle="--")
    
    ax.set_ylabel("Monthly Sales ($)")
    ax.set_title("Training and Testing Fit - Actual vs Predicted Sales")
    ax.legend()
    st.pyplot(fig)
    
    # Comparison table
    st.subheader("Test Predictions Comparison Table")
    eval_table = test_df[["Order Date", "Sales", "Predicted_Sales"]].copy()
    eval_table["Sales"] = eval_table["Sales"].map(format_currency)
    eval_table["Predicted_Sales"] = eval_table["Predicted_Sales"].map(format_currency)
    eval_table["Error"] = (test_df["Sales"] - test_df["Predicted_Sales"]).map(format_currency)
    eval_table = eval_table.rename(columns={"Sales": "Actual Sales", "Predicted_Sales": "Predicted Sales"})
    st.dataframe(eval_table, use_container_width=True)
    
    st.write("---")
    
    # 3. Future Forecasting Section
    st.subheader("3. Future Sales Forecast")
    
    # Horizon slider
    horizon = st.slider("Select Forecast Horizon (Months):", min_value=3, max_value=12, value=6)
    
    # Generate future forecast
    last_row = monthly_series.iloc[-1]
    last_date = last_row["Order Date"]
    # Re-engineer to find final Time Index
    all_engineered = forecasting.engineer_features(monthly_series)
    last_time_index = all_engineered.iloc[-1]["Time Index"]
    
    future_forecast = forecasting.generate_future_forecast(model, last_date, last_time_index, horizon)
    
    # Future Plot (Combined view)
    st.write(f"Plotting {horizon}-Month Out-of-Sample Forecast:")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_series["Order Date"], monthly_series["Sales"], label="Historical Sales", color="#1F2937", marker="o", linewidth=2)
    ax.plot(future_forecast["Order Date"], future_forecast["Sales"], label="Future Forecast", color="#F59E0B", marker="s", linestyle="--", linewidth=2)
    ax.fill_between(future_forecast["Order Date"], future_forecast["Sales"] * 0.85, future_forecast["Sales"] * 1.15, color="#F59E0B", alpha=0.15, label="Estimate Boundary (±15%)")
    
    ax.set_ylabel("Monthly Sales ($)")
    ax.set_title("Historical Monthly Sales and Future Predictions")
    ax.legend()
    st.pyplot(fig)
    
    # Forecast Table
    st.write("Forecast Predictions Table:")
    display_forecast = future_forecast[["Order Date", "Sales"]].copy()
    display_forecast["Forecast Sales"] = display_forecast["Sales"].map(format_currency)
    display_forecast = display_forecast.drop(columns=["Sales"])
    st.dataframe(display_forecast, use_container_width=True)
    
    st.caption("ℹ️ *Forecasts are model estimates based on historical patterns and are not guaranteed future results.*")

# =====================================================================
# PAGE 5: BUSINESS INSIGHTS PAGE
# =====================================================================
elif page == "💡 Business Insights":
    st.markdown("<div class='main-header'>Business Insights & Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Dynamic business analysis and recommendations generated from the actual stores sales dataset.</div>", unsafe_allow_html=True)
    
    # Dynamically compute insight variables
    # 1. Years
    yr_df = analyzer.get_yearly_sales(df)
    best_yr = yr_df.loc[yr_df["Sales"].idxmax()]
    worst_yr = yr_df.loc[yr_df["Sales"].idxmin()]
    
    # 2. Months
    monthly_trend = analyzer.get_monthly_sales_trend(df)
    best_month = monthly_trend.loc[monthly_trend["Sales"].idxmax()]
    worst_month = monthly_trend.loc[monthly_trend["Sales"].idxmin()]
    
    # 3. Regions & Segments
    reg_perf = analyzer.get_regional_performance(df)
    best_reg = reg_perf.iloc[0]
    best_reg_profit = reg_perf.loc[reg_perf["Profit"].idxmax()]
    
    seg_perf = analyzer.get_segment_performance(df)
    best_seg = seg_perf.iloc[0]
    
    # 4. Sub-Categories
    subcat_perf = analyzer.get_subcategory_performance(df)
    best_sub = subcat_perf.iloc[0]
    profitable_sub = subcat_perf.sort_values("Profit", ascending=False).iloc[0]
    loss_making = subcat_perf[subcat_perf["Profit"] < 0]
    
    # 5. Future Trend Direction
    # Train forecasting model to calculate direction
    monthly_series = forecasting.prepare_time_series(df)
    model, _, _, _ = forecasting.train_forecast_model(monthly_series)
    all_engineered = forecasting.engineer_features(monthly_series)
    last_row = all_engineered.iloc[-1]
    future_forecast = forecasting.generate_future_forecast(model, last_row["Order Date"], last_row["Time Index"], horizon=6)
    
    avg_recent = monthly_series["Sales"].iloc[-12:].mean()
    avg_forecast = future_forecast["Sales"].mean()
    
    if avg_forecast > avg_recent:
        forecast_direction = "Expected Upward Trend"
        forecast_desc = "The model forecasts that average monthly sales will increase compared to the recent 12-month historical average."
    else:
        forecast_direction = "Expected Softening / Downward Trend"
        forecast_desc = "The model forecasts a decline in average monthly sales compared to the recent 12-month historical average, indicating potential softening demand."

    # Render Insights
    st.subheader("📊 Dynamic Data Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### 📅 Time-Based Insights
        * **Top Performing Year**: Year **{best_yr['Year']}** generated the highest total sales of **{format_currency(best_yr['Sales'])}**.
        * **Weakest Performing Year**: Year **{worst_yr['Year']}** saw the lowest sales of **{format_currency(worst_yr['Sales'])}**.
        * **Peak Monthly Transaction**: The highest sales month was **{best_month['Year-Month']}** with **{format_currency(best_month['Sales'])}** in sales.
        * **Lowest Monthly Transaction**: The lowest sales month was **{worst_month['Year-Month']}** with **{format_currency(worst_month['Sales'])}** in sales.
        """)
        
    with col2:
        st.markdown(f"""
        ### 🏢 Dimensional Performance
        * **Top Region by Sales**: The **{best_reg['Region']}** region generates the highest sales volume (**{format_currency(best_reg['Sales'])}**).
        * **Top Region by Profitability**: The **{best_reg_profit['Region']}** region yields the highest total profits (**{format_currency(best_reg_profit['Profit'])}**).
        * **Top Customer Segment**: The **{best_seg['Segment']}** segment leads furniture demand (**{format_currency(best_seg['Sales'])}**).
        * **Top Sub-Category**: **{best_sub['Sub-Category']}** leads in total sales (**{format_currency(best_sub['Sales'])}**), while **{profitable_sub['Sub-Category']}** leads in net profit (**{format_currency(profitable_sub['Profit'])}**).
        """)
        
    st.write("---")
    
    st.subheader("⚠️ Profitability Risks Alert")
    if not loss_making.empty:
        for idx, row in loss_making.iterrows():
            st.warning(f"🚨 **Loss-Making Product Sub-Category**: **{row['Sub-Category']}** generated **{format_currency(row['Sales'])}** in sales, but resulted in a net profit loss of **{format_currency(row['Profit'])}**.")
    else:
        st.success("✅ All sub-categories showed net positive profits over the historical range.")
        
    st.write("---")
    
    st.subheader("🔮 6-Month Forecasting Outlook")
    st.metric("Forecast Outlook Direction", forecast_direction, help=forecast_desc)
    st.write(forecast_desc)
    
    st.write("---")
    
    st.subheader("📋 Practical Business Recommendations")
    st.markdown(f"""
    Based on the exploratory and predictive analysis of your database, here are five primary recommendations:
    
    1. **Rectify Loss-Making Sub-Categories**:
       - Review pricing, supply chain costs, and promotional policies for **Tables** (and any other loss-making sub-categories). High sales volume coupled with negative profit margins suggests unsustainable discounting or high delivery/returns overheads.
    2. **Refine Discount Structures**:
       - Our analysis shows a sharp decline in profit margins as discount levels exceed **20%**. Establish corporate guidelines to restrict discounts on furniture to under 20% unless clearing stagnant inventory.
    3. **Leverage Q4 Seasonality**:
       - Plan supply chain logistics and inventory levels to peak in late Q3 to satisfy high demand in November and December. Use the forecast model to determine optimal quantities.
    4. **Double Down on West & East Regions**:
       - The West and East regions are the growth engine for profit margins. Target regional marketing campaigns to capture more corporate/home-office market share in these areas.
    5. **Implement Predictive Inventory Control**:
       - Replace flat-rate historical stocking sheets with the 3-12 month forecast predictions generated on the **Sales Forecasting** page to minimize warehousing overheads while avoiding stock-outs.
    """)

# =====================================================================
# PAGE 6: ABOUT PROJECT PAGE
# =====================================================================
elif page == "ℹ️ About Project":
    st.markdown("<div class='main-header'>About Project</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Project context, metadata, and intern information.</div>", unsafe_allow_html=True)
    
    st.subheader("Project Context")
    st.write("""
    This **Sales Forecasting System** is built as a Level 2 Intermediate Data Science and Data Analytics Internship Project.
    It demonstrates end-to-end data analysis workflows from extraction and preprocessing, through to modeling, evaluation, and dashboard deployment.
    """)
    
    st.subheader("Technical Workflow Summary")
    st.markdown("""
    * **Data Extraction**: Loaded transactional records with `latin1` fallback to avoid byte parsing decode failures.
    * **Preprocessing**: Cleaned timestamps, calculated durations, sorted order chronologically, and removed negative or empty sales.
    * **Exploratory Data Analysis**: Evaluated trends, segments, regions, sub-categories, top products, and impact of discount rates.
    * **Forecasting**: Aggregated sales into monthly observations, engineered cyclical trigonometric month values (Sin/Cos) and trained a Scikit-Learn `LinearRegression` model.
    * **Evaluation**: Assessed out-of-sample monthly predictions with MAE, RMSE, and R² scores.
    * **Deployment**: Wrapped in a Streamlit application suitable for local running and Streamlit Community Cloud hosting.
    """)
    
    st.subheader("Intern Information")
    st.markdown("""
    * **Name**: Vinit Sahani
    * **Role**: Data Science Intern
    * **Project Scope**: Sales Analysis & Time-Series Demand Forecasting Dashboard
    * **Code Repository**: GitHub-Ready
    * **Status**: Complete & Functional
    """)
