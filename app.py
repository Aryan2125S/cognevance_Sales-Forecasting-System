import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    /* Global Styles */
    :root {
        --primary-color: #A78BFA;
        --secondary-color: #34D399;
        --bg-color: #F9FAFB;
        --text-color: #1F2937;
        --card-bg: #FFFFFF;
        --border-color: #E5E7EB;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #8B5CF6;
            --secondary-color: #10B981;
            --bg-color: #111827;
            --text-color: #F9FAFB;
            --card-bg: #1F2937;
            --border-color: #374151;
        }
    }

    /* Styling headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
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
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid var(--border-color);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.02);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-color);
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Ensure markdown text inherits correct colors in dark mode */
    @media (prefers-color-scheme: dark) {
        .sub-header, .kpi-label {
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
            <div class="kpi-value" style="color: #A78BFA;">{format_currency(kpi_dict['total_sales'])}</div>
            <div class="kpi-label">Total Sales</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #60A5FA;">{format_currency(kpi_dict['total_profit'])}</div>
            <div class="kpi-label">Total Profit</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #34D399;">{kpi_dict['total_orders']:,}</div>
            <div class="kpi-label">Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #FBBF24;">{kpi_dict['total_quantity']:,}</div>
            <div class="kpi-label">Total Items Sold</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        color = "#34D399" if kpi_dict['profit_margin'] >= 0 else "#F87171"
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
        
        st.info(f"**Top Performing Category Segment:** Chairs (Sales: **{format_currency(best_sub['Sales'])}**, Profit: **{format_currency(best_sub['Profit'])}**)")
        st.warning("**Low Profitability Alert:** Bookcases and Tables have high sales volume but consistently low profits (Tables show net loss).")
        
    with c_right:
        st.success("**Seasonal Trends:** Peak sales occur in November and December due to Q4 holidays.")
        st.success("**Regional Strength:** The **West** region generates the highest sales and profit margin, followed closely by the **East**.")

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
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #A78BFA;">{format_currency(kpi_dict["total_sales"])}</div><div class="kpi-label">Filtered Sales</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #60A5FA;">{format_currency(kpi_dict["total_profit"])}</div><div class="kpi-label">Filtered Profit</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #34D399;">{kpi_dict["total_orders"]:,}</div><div class="kpi-label">Filtered Orders</div></div>', unsafe_allow_html=True)
    with col4:
        color = "#34D399" if kpi_dict['profit_margin'] >= 0 else "#F87171"
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: {color};">{kpi_dict["profit_margin"]:.2f}%</div><div class="kpi-label">Profit Margin</div></div>', unsafe_allow_html=True)
    
    st.write("---")
    
    # Layout with columns for charts
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Monthly Sales & Profit Trend")
        monthly_trend = analyzer.get_monthly_sales_trend(analysis_df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_trend["Year-Month"], y=monthly_trend["Sales"], mode='lines+markers', name='Sales', line=dict(color='#A78BFA', width=3)))
        fig.add_trace(go.Scatter(x=monthly_trend["Year-Month"], y=monthly_trend["Profit"], mode='lines+markers', name='Profit', line=dict(color='#34D399', width=3)))
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with row1_col2:
        st.subheader("Performance by Sub-Category")
        subcat = analyzer.get_subcategory_performance(analysis_df)
        fig = go.Figure(data=[
            go.Bar(name='Sales', x=subcat["Sub-Category"], y=subcat["Sales"], marker_color='#60A5FA'),
            go.Bar(name='Profit', x=subcat["Sub-Category"], y=subcat["Profit"], marker_color='#F472B6')
        ])
        fig.update_layout(barmode='group', margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("Sales and Profit by Region")
        reg_perf = analyzer.get_regional_performance(analysis_df)
        fig = px.bar(reg_perf, x="Region", y="Sales", color="Region", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.add_trace(go.Scatter(x=reg_perf["Region"], y=reg_perf["Profit"], mode='lines+markers', name='Profit', line=dict(color='#F87171', width=3)))
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with row2_col2:
        st.subheader("Sales by Customer Segment")
        seg_perf = analyzer.get_segment_performance(analysis_df)
        fig = px.pie(seg_perf, values='Sales', names='Segment', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    st.write("---")
    
    # Row 3: Discount Analysis and Top Products
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.subheader("Impact of Discounts on Profit Margin")
        disc_df = analyzer.get_discount_impact(analysis_df)
        fig = px.line(disc_df, x="Discount", y="Profit Margin (%)", markers=True)
        fig.update_traces(line=dict(color="#FBBF24", width=3))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with row3_col2:
        st.subheader("Top 5 Selling Products")
        top_prod = analyzer.get_top_products(analysis_df, metric="Sales", n=5)
        # Shorten names for plotting
        top_prod["Product Short Name"] = [name[:30] + "..." if len(name) > 30 else name for name in top_prod["Product Name"]]
        fig = px.bar(top_prod, x="Sales", y="Product Short Name", orientation='h', color="Sales", color_continuous_scale="Purp")
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PAGE 4: SALES FORECASTING PAGE
# =====================================================================
elif page == "🔮 Sales Forecasting":
    st.markdown("<div class='main-header'>Sales Forecasting Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Monthly forecasting model with Linear Regression and cyclical seasonal indicators.</div>", unsafe_allow_html=True)
    
    # 1. Prepare Monthly Series
    monthly_series = forecasting.prepare_time_series(df)
    
    # 2. Train Model and Get Predictions & Metrics
    model, train_df, test_df, metrics = forecasting.train_forecast_model(monthly_series)
    
    # Model details card
    st.info("""
    **💡 Forecasting Pipeline Explanation:**
    1. **Monthly Aggregation**: Individual transactions are summed into a single monthly sales series.
    2. **Features**:
       - **Time Index**: A sequential number (1, 2, 3...) to capture the long-term upward or downward trend.
       - **Cyclical Seasonality**: `Month Sin` and `Month Cos` derived from `sin(2*pi*month/12)` and `cos(2*pi*month/12)` to model annual seasonal cycles without overfitting.
    3. **Chronological Splitting**: Shuffling time-series data creates target leakage. Instead, the model is trained on the first **80%** of calendar months and evaluated on the latest **20%** chronological months.
    """)
    
    st.subheader("1. Model Performance (Test Set Evaluation)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #F87171;">${metrics["MAE"]:,.2f}</div><div class="kpi-label">Mean Absolute Error</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #FBBF24;">${metrics["RMSE"]:,.2f}</div><div class="kpi-label">Root Mean Squared Error</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #34D399;">{metrics["R2"]:.4f}</div><div class="kpi-label">R² Score</div></div>', unsafe_allow_html=True)
    
    st.write("---")
    
    # Model Evaluation Plot
    st.subheader("2. Model Predictions vs Historical Actuals")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df["Order Date"], y=train_df["Sales"], mode='lines+markers', name='Actual Sales (Train)', line=dict(color='#9CA3AF')))
    fig.add_trace(go.Scatter(x=train_df["Order Date"], y=train_df["Predicted_Sales"], mode='lines', name='Predicted Sales (Train)', line=dict(color='#A78BFA', dash='dash')))
    fig.add_trace(go.Scatter(x=test_df["Order Date"], y=test_df["Sales"], mode='lines+markers', name='Actual Sales (Test)', line=dict(color='#34D399')))
    fig.add_trace(go.Scatter(x=test_df["Order Date"], y=test_df["Predicted_Sales"], mode='lines', name='Predicted Sales (Test)', line=dict(color='#F87171', dash='dash')))
    
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison table
    st.subheader("Test Predictions Comparison Table")
    eval_table = test_df[["Order Date", "Sales", "Predicted_Sales"]].copy()
    eval_table["Error"] = eval_table["Sales"] - eval_table["Predicted_Sales"]
    
    # Formatting for display
    display_eval_table = eval_table.copy()
    display_eval_table["Sales"] = display_eval_table["Sales"].map(format_currency)
    display_eval_table["Predicted_Sales"] = display_eval_table["Predicted_Sales"].map(format_currency)
    display_eval_table["Error"] = display_eval_table["Error"].map(format_currency)
    display_eval_table = display_eval_table.rename(columns={"Sales": "Actual Sales", "Predicted_Sales": "Predicted Sales"})
    st.dataframe(display_eval_table, use_container_width=True)
    
    st.write("---")
    
    # 3. Future Forecasting Section
    st.subheader("3. Future Sales Forecast")
    
    horizon = st.slider("Select Forecast Horizon (Months):", min_value=3, max_value=12, value=6)
    
    last_row = monthly_series.iloc[-1]
    last_date = last_row["Order Date"]
    all_engineered = forecasting.engineer_features(monthly_series)
    last_time_index = all_engineered.iloc[-1]["Time Index"]
    
    future_forecast = forecasting.generate_future_forecast(model, last_date, last_time_index, horizon)
    
    st.write(f"Plotting {horizon}-Month Out-of-Sample Forecast:")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=monthly_series["Order Date"], y=monthly_series["Sales"], mode='lines+markers', name='Historical Sales', line=dict(color='#9CA3AF')))
    fig2.add_trace(go.Scatter(x=future_forecast["Order Date"], y=future_forecast["Sales"], mode='lines+markers', name='Future Forecast', line=dict(color='#FBBF24', dash='dash')))
    
    # Add confidence bounds visualization
    fig2.add_trace(go.Scatter(
        name='Upper Bound',
        x=future_forecast["Order Date"],
        y=future_forecast["Sales"] * 1.15,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ))
    fig2.add_trace(go.Scatter(
        name='Estimate Boundary (±15%)',
        x=future_forecast["Order Date"],
        y=future_forecast["Sales"] * 0.85,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(251, 191, 36, 0.2)',
        fill='tonexty'
    ))
    
    fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    
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
    yr_df = analyzer.get_yearly_sales(df)
    best_yr = yr_df.loc[yr_df["Sales"].idxmax()]
    worst_yr = yr_df.loc[yr_df["Sales"].idxmin()]
    
    monthly_trend = analyzer.get_monthly_sales_trend(df)
    best_month = monthly_trend.loc[monthly_trend["Sales"].idxmax()]
    worst_month = monthly_trend.loc[monthly_trend["Sales"].idxmin()]
    
    reg_perf = analyzer.get_regional_performance(df)
    best_reg = reg_perf.iloc[0]
    best_reg_profit = reg_perf.loc[reg_perf["Profit"].idxmax()]
    
    seg_perf = analyzer.get_segment_performance(df)
    best_seg = seg_perf.iloc[0]
    
    subcat_perf = analyzer.get_subcategory_performance(df)
    best_sub = subcat_perf.iloc[0]
    profitable_sub = subcat_perf.sort_values("Profit", ascending=False).iloc[0]
    loss_making = subcat_perf[subcat_perf["Profit"] < 0]
    
    monthly_series = forecasting.prepare_time_series(df)
    model, _, _, _ = forecasting.train_forecast_model(monthly_series)
    all_engineered = forecasting.engineer_features(monthly_series)
    last_row = all_engineered.iloc[-1]
    future_forecast = forecasting.generate_future_forecast(model, last_row["Order Date"], last_row["Time Index"], horizon=6)
    
    avg_recent = monthly_series["Sales"].iloc[-12:].mean()
    avg_forecast = future_forecast["Sales"].mean()
    
    if avg_forecast > avg_recent:
        forecast_direction = "Expected Upward Trend 📈"
        forecast_desc = "The model forecasts that average monthly sales will increase compared to the recent 12-month historical average."
    else:
        forecast_direction = "Expected Softening / Downward Trend 📉"
        forecast_desc = "The model forecasts a decline in average monthly sales compared to the recent 12-month historical average, indicating potential softening demand."

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
            st.error(f"🚨 **Loss-Making Product Sub-Category**: **{row['Sub-Category']}** generated **{format_currency(row['Sales'])}** in sales, but resulted in a net profit loss of **{format_currency(row['Profit'])}**.")
    else:
        st.success("✅ All sub-categories showed net positive profits over the historical range.")
        
    st.write("---")
    
    st.subheader("🔮 6-Month Forecasting Outlook")
    st.markdown(f"**{forecast_direction}**")
    st.info(forecast_desc)
    
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
    st.info("""
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
    * **Name**: Aryan Sagar
    * **Role**: Data Science Intern
    * **Project Scope**: Sales Analysis & Time-Series Demand Forecasting Dashboard
    * **Code Repository**: GitHub-Ready
    * **Status**: Complete & Functional
    """)
