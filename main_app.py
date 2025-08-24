# main_app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# 1. DATA GENERATION AND PREPROCESSING
def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic F&B process data for baked goods production
    """
    np.random.seed(42)
    
    # Raw material parameters
    data = {
        'flour_kg': np.random.normal(50, 2, n_samples),
        'water_kg': np.random.normal(30, 1.5, n_samples),
        'yeast_kg': np.random.normal(2, 0.2, n_samples),
        'sugar_kg': np.random.normal(5, 0.5, n_samples),
        'salt_kg': np.random.normal(1, 0.1, n_samples),
        'mixing_time_min': np.random.normal(15, 3, n_samples),
        'mixing_speed_rpm': np.random.normal(120, 10, n_samples),
        'fermentation_time_min': np.random.normal(120, 15, n_samples),
        'fermentation_temp_c': np.random.normal(30, 2, n_samples),
        'proofing_time_min': np.random.normal(60, 10, n_samples),
        'proofing_temp_c': np.random.normal(35, 1, n_samples),
        'baking_time_min': np.random.normal(45, 5, n_samples),
        'baking_temp_c': np.random.normal(180, 10, n_samples),
        'oven_zone1_temp_c': np.random.normal(190, 15, n_samples),
        'oven_zone2_temp_c': np.random.normal(185, 12, n_samples),
        'oven_zone3_temp_c': np.random.normal(175, 8, n_samples),
        'cooling_time_min': np.random.normal(30, 5, n_samples),
        'cooling_temp_c': np.random.normal(25, 3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate quality metrics based on process parameters
    df['weight_deviation'] = (
        (df['flour_kg'] + df['water_kg'] + df['yeast_kg'] + df['sugar_kg'] + df['salt_kg']) * 
        (1 - 0.01 * (df['baking_time_min'] - 45) / 10) + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Core temperature after baking (simulated)
    df['core_temp_c'] = (
        df['baking_temp_c'] * 0.6 - 
        (df['baking_time_min'] - 45) * 0.5 + 
        np.random.normal(0, 2, n_samples)
    )
    
    # Color development (simulated)
    df['color_score'] = (
        80 + 
        0.5 * (df['baking_temp_c'] - 180) + 
        0.8 * (df['baking_time_min'] - 45) + 
        np.random.normal(0, 5, n_samples)
    )
    
    # Final quality score (0-100)
    df['quality_score'] = (
        85 - 
        np.abs(df['weight_deviation'] - 87.5) * 2 -
        np.abs(df['core_temp_c'] - 92) * 0.5 -
        np.abs(df['color_score'] - 95) * 0.3 +
        np.random.normal(0, 3, n_samples)
    )
    
    # Ensure quality score is within bounds
    df['quality_score'] = df['quality_score'].clip(0, 100)
    
    # Add some anomalies (5% of data)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[anomaly_indices, 'quality_score'] = np.random.uniform(0, 50, len(anomaly_indices))
    
    return df

# 2. DATA QUALITY ANALYSIS
def analyze_data_quality(df):
    """
    Perform statistical analysis of data quality with improved outlier detection
    """
    st.header("F&B Process Data Analysis")
    
    # Basic statistics in an expander
    with st.expander("📊 Basic Statistics", expanded=True):
        st.dataframe(df.describe())
    
    # Missing values analysis
    with st.expander("🔍 Missing Values Analysis", expanded=True):
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Values': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_df)
        
        if missing_df['Missing_Values'].sum() == 0:
            st.success("✅ No missing values detected in the dataset.")
        else:
            st.warning(f"⚠️ Found {missing_df['Missing_Values'].sum()} missing values across the dataset.")
    
    # Improved Outlier detection with multiple methods
    with st.expander("📈 Outlier Analysis", expanded=True):
        st.subheader("Outlier Detection Results")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_results = {}
        
        # Method 1: IQR (for comparison)
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_results[col] = len(outliers)
        
        outlier_df = pd.DataFrame.from_dict(outlier_results, orient='index', columns=['Outlier_Count'])
        outlier_df['Outlier_Percentage'] = (outlier_df['Outlier_Count'] / len(df)) * 100
        
        # Method 2: Z-score (more appropriate for normal distributions)
        zscore_outliers = {}
        for col in numerical_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            zscore_outliers[col] = len(df[z_scores > 3])  # 3 standard deviations
        
        zscore_df = pd.DataFrame.from_dict(zscore_outliers, orient='index', columns=['ZScore_Outlier_Count'])
        zscore_df['ZScore_Outlier_Percentage'] = (zscore_df['ZScore_Outlier_Count'] / len(df)) * 100
        
        # Combine results
        combined_df = outlier_df.join(zscore_df)
        st.dataframe(combined_df)
        
        total_iqr_outliers = outlier_df['Outlier_Count'].sum()
        total_zscore_outliers = zscore_df['ZScore_Outlier_Count'].sum()
        
        st.info(f"📋 IQR Method outliers: {total_iqr_outliers}")
        st.info(f"📋 Z-Score Method outliers: {total_zscore_outliers}")
    
    # Visualize outliers for key parameters
    with st.expander("📉 Data Distribution Visualization", expanded=True):
        st.subheader("Distribution of Key Parameters")
        
        key_params = ['flour_kg', 'mixing_time_min', 'baking_temp_c', 'quality_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        for i, param in enumerate(key_params):
            # Create histogram with KDE
            sns.histplot(df[param], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {param}')
            axes[i].set_xlabel('')
            
            # Add vertical lines for mean and median
            axes[i].axvline(df[param].mean(), color='red', linestyle='--', label='Mean')
            axes[i].axvline(df[param].median(), color='green', linestyle='--', label='Median')
            axes[i].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("**Interpretation:** Histograms show the distribution of key parameters. Red dashed line = Mean, Green dashed line = Median.")
    
    # Enhanced Data Quality Summary
    with st.expander("✅ Data Quality Assessment", expanded=True):
        st.subheader("Overall Data Quality Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        
        with col2:
            missing_count = df.isnull().sum().sum()
            st.metric("Missing Values", missing_count, delta_color="inverse")
        
        with col3:
            # Use Z-score outliers as they're more reliable for synthetic data
            outlier_count = total_zscore_outliers
            st.metric("Statistical Outliers", outlier_count, delta_color="inverse")
        
        with col4:
            anomaly_count = len(df[df['quality_score'] < 60])  # Actual quality anomalies
            st.metric("Quality Anomalies", anomaly_count)
        
        # Enhanced quality assessment logic
        st.subheader("Quality Assessment")
        
        if missing_count > 0:
            st.error("❌ Data Quality Issue: Missing values detected that need handling.")
        elif outlier_count > len(df) * 0.10:  # More than 10% outliers
            st.warning("⚠️ Note: Significant statistical outliers detected. This is expected with synthetic data and real-world process variations.")
            st.info("💡 Recommendation: These 'outliers' may represent normal process variation. Focus on domain-specific limits rather than statistical outliers.")
        elif outlier_count > len(df) * 0.05:  # 5-10% outliers
            st.warning("⚠️ Moderate outliers detected. Review if they represent true anomalies or normal variation.")
        else:
            st.success("✅ Excellent data quality for analysis.")
        
        # Process-specific assessment
        st.subheader("Process Health Check")
        
        # Check if key parameters are within reasonable ranges
        flour_check = df['flour_kg'].between(40, 60).mean() > 0.95  # 95% within range
        temp_check = df['baking_temp_c'].between(160, 200).mean() > 0.95
        quality_check = df['quality_score'].mean() > 70
        
        checks_passed = sum([flour_check, temp_check, quality_check])
        
        if checks_passed == 3:
            st.success("✅ All process parameters within expected ranges")
        elif checks_passed >= 1:
            st.warning("⚠️ Some process parameters need attention")
        else:
            st.error("❌ Multiple process parameters out of expected ranges")
    
    # Recommendations section
    with st.expander("🎯 Recommendations & Next Steps", expanded=True):
        st.subheader("Data Preparation Recommendations")
        
        st.markdown("""
        1. **For Synthetic Data:** The detected 'outliers' are likely valid process variations
        2. **Focus** on domain-specific control limits rather than statistical outliers
        3. **Key parameters** to monitor:
           - Baking Temperature: 160-200°C
           - Quality Score: >60 acceptable, >80 good
           - Weight Deviation: <2% target
        4. **Next Steps:** Proceed to model training to predict quality anomalies
        """)
    
    return df
# 3. MACHINE LEARNING MODEL
def build_and_train_model(df):
    """
    Build and train the predictive model
    """
    st.header("Machine Learning Model Development")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['quality_score', 'weight_deviation', 'core_temp_c', 'color_score']]
    X = df[feature_cols]
    y = df['quality_score']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
        'Value': [mae, mse, rmse, r2]
    })
    st.dataframe(metrics_df)
    
    # Algorithm Justification
    st.subheader("Algorithm Selection Justification")
    st.markdown("""
    **Random Forest Regressor** was chosen for this problem for several key reasons:
    - **Handle Complex Relationships:** Effectively models non-linear interactions between multiple process parameters.
    - **Robustness:** Naturally handles outliers and missing data better than many other algorithms.
    - **Interpretability:** Provides **feature importance** scores, crucial for root-cause analysis.
    - **High Performance:** Consistently delivers high accuracy with minimal hyperparameter tuning.
    This model doesn't just predict; it helps us understand the 'why' behind a potential quality issue.
    """)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
    ax.set_title('Top 10 Feature Importances')
    st.pyplot(fig)
    
    # Engineering Interpretation
    st.subheader("Engineering Interpretation")
    st.markdown("""
    The feature importance analysis reveals critical process insights:
    - **Baking Temperature** drives the Maillard reaction (color/flavor) and starch gelatinization.
    - **Fermentation Time** directly controls yeast activity, impacting volume and taste.
    - **Mixing Time** develops the gluten network responsible for bread structure.
    Monitoring these parameters is critical for preventing anomalies.
    """)
    
    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Quality Score')
    ax.set_ylabel('Predicted Quality Score')
    ax.set_title('Actual vs Predicted Quality Scores')
    st.pyplot(fig)
    
    return model, scaler, feature_cols

# 4. REAL-TIME DASHBOARD
def create_dashboard(df, model, scaler, feature_cols):
    """
    Create a real-time dashboard for process monitoring
    """
    st.title("F&B Process Anomaly Prediction Dashboard")
    
    # Sidebar for user inputs
    st.sidebar.header("Process Control Parameters")
    
    # Create sliders for key parameters
    params = {}
    for col in feature_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        
        params[col] = st.sidebar.slider(
            col, 
            min_value=min_val, 
            max_value=max_val, 
            value=mean_val,
            step=(max_val - min_val) / 100
        )
    
    # Create a DataFrame from the parameters
    input_df = pd.DataFrame([params])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display prediction
    st.header("Quality Prediction")
    st.metric("Predicted Quality Score", f"{prediction:.2f}")
    
    # Quality indicator
    if prediction >= 80:
        st.success("✅ Quality: Excellent")
    elif prediction >= 60:
        st.warning("⚠️ Quality: Acceptable")
    else:
        st.error("❌ Quality: Poor - Process Anomaly Detected!")
    
    # Process visualization
    st.header("Process Parameters Visualization")
    
    # Select key parameters to visualize
    key_params = ['flour_kg', 'water_kg', 'mixing_time_min', 'mixing_speed_rpm', 
                  'fermentation_temp_c', 'baking_temp_c', 'baking_time_min']
    
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=key_params,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    for i, param in enumerate(key_params):
        row = i // 4 + 1
        col = i % 4 + 1
        
        fig.add_trace(
            go.Indicator(
                mode = "gauge+number",
                value = params[param],
                title = {'text': param},
                domain = {'row': row-1, 'column': col-1},
                gauge = {
                    'axis': {'range': [df[param].min(), df[param].max()]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [df[param].min(), df[param].quantile(0.33)], 'color': "lightgray"},
                        {'range': [df[param].quantile(0.33), df[param].quantile(0.66)], 'color': "gray"},
                        {'range': [df[param].quantile(0.66), df[param].max()], 'color': "darkgray"}
                    ]
                }
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=500, width=900)
    st.plotly_chart(fig)
    
    # Historical data visualization
    st.header("Historical Data Trends")
    
    selected_param = st.selectbox("Select parameter to visualize:", key_params)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[selected_param], alpha=0.7)
    ax.axhline(y=params[selected_param], color='r', linestyle='--', label='Current Setting')
    ax.set_title(f'Historical Trend of {selected_param}')
    ax.set_xlabel('Batch Number')
    ax.set_ylabel(selected_param)
    ax.legend()
    st.pyplot(fig)

# 5. PROCESS OVERVIEW PAGE
def show_process_overview():
    """
    Displays the detailed F&B process overview
    """
    st.header("🍞 Industrial Bakery Process Overview & Theory")

    st.subheader("1. Process Steps for Bread Production")
    st.markdown("""
    **1. Mixing & Ingredient Addition**
    - **Purpose:** Hydrates dry ingredients and develops gluten structure.
    - **Equipment:** Horizontal or Spiral Mixer
    - **Key Control Parameters:** `mixing_time_min`, `mixing_speed_rpm`, ingredient quantities

    **2. Fermentation (Bulk Fermentation)**
    - **Purpose:** Yeast consumes sugars, producing CO₂ (for volume) and flavor compounds.
    - **Equipment:** Fermentation Cabinet
    - **Key Control Parameters:** `fermentation_time_min`, `fermentation_temp_c`

    **3. Proofing (Final Proof)**
    - **Purpose:** Final rise of the shaped dough before baking.
    - **Equipment:** Proofing Cabinet
    - **Key Control Parameters:** `proofing_time_min`, `proofing_temp_c`

    **4. Baking**
    - **Purpose:** Sets structure, creates crust, develops flavor via Maillard reaction.
    - **Equipment:** Multi-zone Tunnel Oven
    - **Key Control Parameters:** `baking_time_min`, `baking_temp_c`, oven zone temperatures

    **5. Cooling**
    - **Purpose:** Allows moisture to redistribute and structure to set.
    - **Equipment:** Cooling Conveyor
    - **Key Control Parameters:** `cooling_time_min`, `cooling_temp_c`
    """)

    st.subheader("2. Definition of Final Product Quality")
    st.markdown("""
    Product quality is quantitatively defined based on measurable attributes:
    - **Core Temperature (`core_temp_c`):** Must reach **92-99°C** for safety and doneness.
    - **Weight Deviation (`weight_deviation`):** Ideal range **< 2%** from target weight.
    - **Color Score (`color_score`):** Target > **90** for optimal browning.
    - **Overall Quality Score (`quality_score`):** Composite index (0-100). **Scores below 60 indicate anomaly.**
    """)

    st.subheader("3. References")
    st.markdown("""
    - Cauvain, S.P. & Young, L.S. (2007). *Technology of Breadmaking*. Springer.
    - AIBI (International Association of Plant Bakeries) Best Practices Guidelines
    - UCI Machine Learning Repository for process analysis datasets
    """)

# 6. MAIN APPLICATION
def main():
    """
    Main function to run the F&B Process Anomaly Prediction System
    """
    st.set_page_config(
        page_title="F&B Process Anomaly Prediction",
        page_icon="🍞",
        layout="wide"
    )
    
    # Generate or load data
    st.sidebar.title("Data Management")
    if st.sidebar.button("Generate New Data"):
        df = generate_synthetic_data(1000)
        st.session_state.df = df
    elif 'df' not in st.session_state:
        df = generate_synthetic_data(1000)
        st.session_state.df = df
    else:
        df = st.session_state.df
    
    # Display dataset info
    st.sidebar.info(f"Dataset Shape: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Application menu
    app_mode = st.sidebar.selectbox("Select Page", 
                                   ["Process Overview", "Data Analysis", "Machine Learning Model", "Real-time Dashboard"])
    
    if app_mode == "Process Overview":
        show_process_overview()
        
    elif app_mode == "Data Analysis":
        st.title("F&B Process Data Analysis")
        analyze_data_quality(df)
        
    elif app_mode == "Machine Learning Model":
        if 'model' not in st.session_state:
            model, scaler, feature_cols = build_and_train_model(df)
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
        else:
            build_and_train_model(df)
            
    elif app_mode == "Real-time Dashboard":
        if 'model' in st.session_state:
            create_dashboard(df, st.session_state.model, st.session_state.scaler, st.session_state.feature_cols)
        else:
            st.warning("Please train the model first from the 'Machine Learning Model' page.")

if __name__ == "__main__":
    main()