
import sys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
# from prophet import Prophet

def perform_hypothesis_testing(data, numerical_columns, cat_columns):
    """Performs hypothesis testing based on user input."""
    st.subheader("📊 Hypothesis Testing")
    test_type = st.selectbox('Select a hypothesis test:', ['T-Test', 'Chi-Square Test', 'ANOVA'])

    if test_type == 'T-Test':
        col1, col2 = st.columns(2)
        with col1:
            group1_col = st.selectbox('Select column for group 1:', options=numerical_columns)
        with col2:
            group2_col = st.selectbox('Select column for group 2:', options=numerical_columns)

        if st.button('Perform T-Test'):
            t_stat, p_val = stats.ttest_ind(data[group1_col].dropna(), data[group2_col].dropna())
            st.write(f"T-Test Results: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
            if p_val < 0.05:
                st.success("The difference between the groups is statistically significant.")
            else:
                st.info("No statistically significant difference between the groups.")

    elif test_type == 'Chi-Square Test':
        col1, col2 = st.columns(2)
        with col1:
            cat1 = st.selectbox('Select first categorical column:', options=cat_columns)
        with col2:
            cat2 = st.selectbox('Select second categorical column:', options=cat_columns)

        if st.button('Perform Chi-Square Test'):
            contingency_table = pd.crosstab(data[cat1], data[cat2])
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-Square Test Results: chi2-statistic = {chi2_stat:.4f}, p-value = {p_val:.4f}")
            if p_val < 0.05:
                st.success("There is a statistically significant association between the variables.")
            else:
                st.info("No statistically significant association between the variables.")

    elif test_type == 'ANOVA':
        col1, col2 = st.columns(2)
        with col1:
            cat_col = st.selectbox('Select categorical column:', options=cat_columns)
        with col2:
            num_col = st.selectbox('Select numerical column:', options=numerical_columns)

        if st.button('Perform ANOVA'):
            anova_result = stats.f_oneway(*(data[data[cat_col] == cat][num_col].dropna() for cat in data[cat_col].unique()))
            st.write(f"ANOVA Results: F-statistic = {anova_result.statistic:.4f}, p-value = {anova_result.pvalue:.4f}")
            if anova_result.pvalue < 0.05:
                st.success("There is a statistically significant difference between the groups.")
            else:
                st.info("No statistically significant difference between the groups.")

def perform_regression(data, numerical_columns, categorical_columns):
    """Performs regression analysis based on user input."""
    st.subheader("📊 Regression Analysis")
    regression_type = st.selectbox('Select a regression type:', ['Linear Regression', 'Logistic Regression'])

    if regression_type == 'Linear Regression':
        features = st.multiselect('Select features (independent variables):', options=numerical_columns)
        target = st.selectbox('Select target variable (dependent variable):', options=numerical_columns)

        if st.button('Perform Linear Regression'):
            X = data[features].dropna()
            y = data[target].dropna()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"Model Coefficients: {model.coef_}")
            st.write(f"Intercept: {model.intercept_}")

    elif regression_type == 'Logistic Regression':
        features = st.multiselect('Select features (independent variables):', options=numerical_columns)
        target = st.selectbox('Select target variable (dependent variable):', options=categorical_columns)

        if st.button('Perform Logistic Regression'):
            X = data[features].dropna()
            y = data[target].dropna()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Model Coefficients: {model.coef_}")
            st.write(f"Intercept: {model.intercept_}")



# def forecast_with_prophet(data, periods=30):
#     """
#     Forecasts future values using Facebook Prophet.

#     Parameters:
#     - data: The DataFrame containing the time series data.
#     - periods: The number of periods (days) to forecast into the future.

#     Returns:
#     - forecast: The DataFrame containing the forecasted values.
#     """
#     if 'date' not in data.columns:
#         st.error("The dataset does not contain a 'date' column.")
#         return None

#     # Prepare data for Prophet
#     df = data.reset_index().rename(columns={'date': 'ds', data.columns[0]: 'y'})  # Rename for Prophet

#     # Initialize and fit the Prophet model
#     model = Prophet()
#     model.fit(df)

#     # Create a DataFrame with future dates for prediction
#     future = model.make_future_dataframe(periods=periods)
#     forecast = model.predict(future)

#     # Display the forecasted values
#     st.write(f"Forecast for the next {periods} days:")
#     st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
#     # Plot the forecast
#     fig = model.plot(forecast)
#     st.write(fig)
    
#     return forecast


from statsmodels.tsa.seasonal import seasonal_decompose

def seasonal_decomposition(data, model='multiplicative'):
    """
    Performs seasonal decomposition on the time series data.

    Parameters:
    - data: The DataFrame containing the time series data.
    - model: The type of seasonal decomposition ('additive' or 'multiplicative').

    Returns:
    - decomposition: The object containing the decomposed components.
    """
    if 'date' not in data.columns:
        st.error("The dataset does not contain a 'date' column.")
        return None

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data, model=model)

    # Display the decomposed components
    st.subheader("Seasonal Decomposition")
    st.write("Trend Component")
    st.line_chart(decomposition.trend)

    st.write("Seasonal Component")
    st.line_chart(decomposition.seasonal)

    st.write("Residual Component")
    st.line_chart(decomposition.resid)

    return decomposition


def perform_kmeans_clustering(data, num_clusters):
    st.subheader("K-Means Clustering")

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        st.error("No numeric data available for clustering.")
        return data

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Perform K-Means Clustering
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(scaled_data)

    # Add the cluster labels to the original DataFrame
    data['Cluster'] = clusters

    # Display the cluster centers
    st.write("Cluster Centers:")
    st.write(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=numeric_data.columns))

    # Display the clustered data
    st.write("Clustered Data:")
    st.dataframe(data)

    # Visualize the clusters
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=data['Cluster'], palette='viridis', ax=ax)
    ax.set_title("K-Means Clustering")
    st.pyplot(fig)

    return data

def perform_dbscan_clustering(data, eps, min_samples):
    st.subheader("DBSCAN Clustering")

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        st.error("No numeric data available for clustering.")
        return data

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Perform DBSCAN Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)

    # Add the cluster labels to the original DataFrame
    data['Cluster'] = clusters

    # Display the clustered data
    st.write("Clustered Data:")
    st.dataframe(data)

    # Visualize the clusters
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=data['Cluster'], palette='viridis', ax=ax)
    ax.set_title("DBSCAN Clustering")
    st.pyplot(fig)

    return data
