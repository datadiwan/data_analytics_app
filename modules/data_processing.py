import streamlit as st
def handle_duplicates(data):
    st.subheader("⚠️ Duplicated Detection",divider='gray')
    # Check for duplicates
    duplicates = data.duplicated()
    num_duplicates = duplicates.sum()

    if num_duplicates > 0:
        st.subheader(f"Detected {num_duplicates} Duplicate Rows")
        st.write("Here are the duplicate rows:")
        st.dataframe(data[duplicates])

        # Option to remove duplicates
        if st.button('Remove Duplicates'):
            data = data.drop_duplicates()
            st.success(f"Removed {num_duplicates} duplicate rows.")
            st.write("Updated DataFrame:")
            st.dataframe(data)
    else:
        st.success("No duplicate rows detected.")
    return data

def handle_outliers(data):
    st.subheader("📊 Outlier Detection",divider='gray')

    numerical_columns = data.select_dtypes(include='number').columns
    if not numerical_columns.any():
        st.warning("No numerical columns available for outlier detection.")
        return data

    # Select column for outlier detection
    column = st.selectbox("Select column for outlier detection:", options=numerical_columns)

    # Choose method for outlier detection
    method = st.selectbox("Select method to detect outliers:", options=["IQR", "Z-score"])

    if st.button("Detect Outliers"):
        if method == "IQR":
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        elif method == "Z-score":
            from scipy import stats
            z_scores = stats.zscore(data[column])
            outliers = data[(z_scores < -3) | (z_scores > 3)]

        if not outliers.empty:
            st.write(f"Detected {len(outliers)} outliers in column '{column}':")
            st.dataframe(outliers)

            if st.button("Remove Outliers"):
                data = data.drop(outliers.index)
                st.success(f"Removed {len(outliers)} outliers from column '{column}'.")
                st.write("Updated DataFrame:")
                st.dataframe(data)
        else:
            st.success(f"No outliers detected in column '{column}'.")

    return data

def identify_filterable_columns(data):
    """Identify columns suitable for filtering."""
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    return categorical_columns, numerical_columns

def filter_data(data):
    """Create UI elements for filtering the dataset."""
    categorical_columns, numerical_columns = identify_filterable_columns(data)

    st.sidebar.header("Data Filtering")

    # Create filters for categorical columns
    for col in categorical_columns:
        unique_values = data[col].unique()
        selected_values = st.sidebar.multiselect(f"Filter {col}", options=unique_values, default=unique_values)
        if selected_values:
            data = data[data[col].isin(selected_values)]

    # Create filters for numerical columns
    for col in numerical_columns:
        min_value, max_value = data[col].min(), data[col].max()
        selected_range = st.sidebar.slider(f"Filter {col}", min_value=min_value, max_value=max_value, value=(min_value, max_value))
        data = data[(data[col] >= selected_range[0]) & (data[col] <= selected_range[1])]

    return data


def handle_missing_data(data):
    st.subheader("🛑 Missing Detection",divider='gray')
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()

    if total_missing > 0:
        st.subheader(f" Missing Data Detected: {total_missing} missing values found")
        st.write("Missing values by column:")
        st.write(missing_values[missing_values > 0])

        if st.checkbox('Handle Missing Data'):
            st.info(f"Total missing values: {total_missing}")
            
            # Select the method to handle missing data
            missing_option = st.selectbox('Select a method to handle missing data', 
                                        ['Drop Rows', 'Fill with Mean', 'Fill with Median', 'Fill with Mode', 
                                        'Fill with Custom Value', 'Forward Fill', 'Backward Fill'])

            # Choose specific columns to apply the method (optional)
            columns_to_apply = st.multiselect('Choose specific columns to apply this method (leave empty for all columns)', 
                                            options=data.columns[data.isnull().sum() > 0])

            # If no columns are selected, apply to all columns with missing data
            if not columns_to_apply:
                columns_to_apply = data.columns[data.isnull().sum() > 0]

            # Filter out non-numeric columns if the method requires numeric data
            if missing_option in ['Fill with Mean', 'Fill with Median', 'Fill with Mode']:
                numeric_columns = data[columns_to_apply].select_dtypes(include=['number']).columns
                non_numeric_columns = set(columns_to_apply) - set(numeric_columns)

                if non_numeric_columns:
                    st.warning(f"Cannot apply {missing_option} to non-numeric columns: {', '.join(non_numeric_columns)}")
                    columns_to_apply = list(numeric_columns)
                    
            # Handle custom value
            custom_value = None
            if missing_option == 'Fill with Custom Value':
                custom_value_input = st.text_input('Enter the custom value to fill missing data')
                try:
                    custom_value = float(custom_value_input) if custom_value_input else None
                except ValueError:
                    st.error("Please enter a valid numeric value.")

            # Apply the selected method
            if missing_option == 'Drop Rows':
                data = data.dropna(subset=columns_to_apply)
            elif missing_option == 'Fill with Mean':
                data[columns_to_apply] = data[columns_to_apply].fillna(data[columns_to_apply].mean())
            elif missing_option == 'Fill with Median':
                data[columns_to_apply] = data[columns_to_apply].fillna(data[columns_to_apply].median())
            elif missing_option == 'Fill with Mode':
                for col in columns_to_apply:
                    mode_value = data[col].mode()
                    if not mode_value.empty:
                        data[col] = data[col].fillna(mode_value.iloc[0])
                    else:
                        st.warning(f"Cannot fill with mode for column '{col}' because the mode is undefined.")
            elif missing_option == 'Fill with Custom Value' and custom_value is not None:
                data[columns_to_apply] = data[columns_to_apply].fillna(custom_value)
            elif missing_option == 'Forward Fill':
                data[columns_to_apply] = data[columns_to_apply].fillna(method='ffill')
            elif missing_option == 'Backward Fill':
                data[columns_to_apply] = data[columns_to_apply].fillna(method='bfill')

            st.success(f"Missing data handled using {missing_option}.")
            st.write("Updated DataFrame:")
            st.dataframe(data)

    else:
        st.success("No missing data detected.")
    return data


def group_by_time(data, time_period):
    """
    Groups data by a specified time period.

    Parameters:
    - data: The DataFrame containing the time series data.
    - time_period: The period by which to group the data ('D' for daily, 'W' for weekly, etc.).

    Returns:
    - grouped_data: The DataFrame grouped by the specified time period.
    """
    if 'date' not in data.columns:
        st.error("The dataset does not contain a 'date' column.")
        return None

    grouped_data = data.resample(time_period).mean()  # Resample by time period and calculate mean
    st.write(f"Data grouped by {time_period}:")
    st.dataframe(grouped_data)
    return grouped_data