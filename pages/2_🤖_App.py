# Import libraries
import base64
import pandas as pd 
import plotly.express as px
import streamlit as st   
import os


# Set the page configuration
st.set_page_config(
    page_title='Data Diwan Analytics App',
    page_icon='📊'
)
# Path to your logo image
current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, 'logo.jpeg')

# Read the image file
with open(logo_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Inject CSS to style the image as circular and center it
st.markdown(
    f"""
    <style>
    .logo-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px; /* Adjust this if needed */
    }}
    .logo-img {{
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #ddd; /* Optional: Add a border around the circle */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo in the app
st.markdown(
    f"""
    <div class="logo-container">
        <img src="data:image/jpeg;base64,{encoded_image}" class="logo-img" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
# st.title(':orange[Data Scientist App]:robot_face:')
st.subheader(':gray[Discover data effortlessly.]', divider='orange')
st.sidebar.title("The Data scientist App📊📈")
def plot_value_counts(data, column, num_rows):
    """Plots the value counts for a given column."""
    result = data[column].value_counts().reset_index().head(num_rows)
    result.columns = ['index', 'count']  # Rename columns for clarity

    st.dataframe(result)

    st.subheader('Visualization', divider='gray')

    # Create the bar chart
    fig = px.bar(data_frame=result, x='index', y='count', text='count', template='plotly_white')
    st.plotly_chart(fig)

    # Create the line chart
    fig = px.line(data_frame=result, x='index', y='count', text='count', template='plotly_white')
    st.plotly_chart(fig)

    # Create the pie chart
    fig = px.pie(data_frame=result, names='index', values='count')
    st.plotly_chart(fig)


data = None  # Initialize the data variable to None

# Load a dataset from a CSV or Excel file
files = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'], accept_multiple_files=True)

if files:
    datasets = {}  # Dictionary to store the datasets
    for file in files:
        if file.name.endswith('csv'):
            data = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            st.error(f"Unsupported file format: {file.name}")
            continue  # Skip to the next file if unsupported format
        
        datasets[file.name] = data  # Store each dataset in the dictionary
        st.text(f'{file.name} is successfully Uploaded✅')

    if datasets:
        dataset_to_discover = st.selectbox("***Select a dataset to discover:***", options=list(datasets.keys()))
        data = datasets[dataset_to_discover]  # Fetch the selected dataset
        st.dataframe(data)
        
        st.subheader(f":orange[🛠️ Analyzing Dataset:] {dataset_to_discover}")

        # Tabs for Data Analysis
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔝 Top/Bottom", "📈 Summary", 
            "📑 Columns", "🧹Cleaning", "🔢 Value Counts", "🔠 Group By"
        ])

        with tab1:
            st.subheader(":orange[Rows Header]")
            top_n = st.slider("Select number of top rows to view:", 1, min(100, data.shape[0]), 5, key='top_n_slider')
            st.dataframe(data.head(top_n))

            st.subheader(":orange[Rows Tail]")
            bottom_n = st.slider("Select number of bottom rows to view:", 1, min(100, data.shape[0]), 5, key='bottom_n_slider')
            st.dataframe(data.tail(bottom_n))

        with tab2:
            st.subheader(":orange[Descriptive Statistics]")
            st.dataframe(data.describe())

            st.write(f":orange[Shape of the Dataset:] {data.shape[0]} rows and {data.shape[1]} columns")
            
            st.subheader(":orange[Data Types]")
            data_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
            st.dataframe(data_types)

        with tab3:
            st.subheader(":orange[Columns in Dataset]")
            st.write(list(data.columns))

        with tab4:
            st.subheader(":orange[⚠️Handle Duplicates]")
            # Check for duplicates
            duplicates = data.duplicated()
            num_duplicates = duplicates.sum()

            if num_duplicates > 0:
                st.text(f" Detected {num_duplicates} Duplicate Rows")
                
                # Show duplicate rows
                st.write("Here are the duplicate rows:")
                st.dataframe(data[duplicates])

                # Option to remove duplicates
                if st.button('Remove Duplicates'):
                    data = data.drop_duplicates()
                    datasets[dataset_to_discover] = data  # Update the dataset in the dictionary
                    st.success(f"Removed {num_duplicates} duplicate rows.")
                    st.write("Updated DataFrame:")
                    st.dataframe(data)
            else:
                st.success("No duplicate rows detected.")

            st.subheader(":orange[🛑Handle Missing Data]")
            # Check for missing values
            missing_values = data.isnull().sum()
            total_missing = missing_values.sum()

            if total_missing > 0:
                st.markdown(f"**There are :orange[{total_missing}] missing values found**")
                st.write(f"Missing values by column in {dataset_to_discover}:")
                st.write(missing_values[missing_values > 0])

                # Visualize missing values
                st.bar_chart(missing_values[missing_values > 0])

                if st.checkbox('Handle Missing Data'):
                    st.info(f"Total missing values in {dataset_to_discover}: {total_missing}")
                    
                    # Select the method to handle missing data
                    missing_option = st.selectbox('Select a method to handle missing data', 
                                                ['Drop Rows', 'Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Fill with Custom Value'])
                    
                    # Choose specific columns to apply the method (optional)
                    columns_to_apply = st.multiselect('Choose specific columns to apply this method (leave empty for all columns)', 
                                                    options=data.columns[data.isnull().sum() > 0])
                    
                    # If no columns are selected, apply to all columns with missing data
                    if not columns_to_apply:
                        columns_to_apply = data.columns[data.isnull().sum() > 0]
                    
                    # Separate numeric and non-numeric columns
                    numeric_columns = data[columns_to_apply].select_dtypes(include=['number']).columns
                    non_numeric_columns = data[columns_to_apply].select_dtypes(exclude=['number']).columns

                    # Handle custom value
                    custom_value = None
                    if missing_option == 'Fill with Custom Value':
                        custom_value_input = st.text_input('Enter the custom value to fill missing data')
                        try:
                            custom_value = float(custom_value_input) if custom_value_input else None
                        except ValueError:
                            st.error("Please enter a valid numeric value.")
                    
                    # Preview the changes before applying
                    preview = data.copy()
                    if missing_option == 'Drop Rows':
                        preview = preview.dropna(subset=columns_to_apply)
                        st.write("Preview after dropping rows with missing values:")

                    elif missing_option == 'Fill with Mean' and not numeric_columns.empty:
                        preview[numeric_columns] = preview[numeric_columns].fillna(preview[numeric_columns].mean())
                        st.write("Preview after filling with mean:")

                    elif missing_option == 'Fill with Median' and not numeric_columns.empty:
                        preview[numeric_columns] = preview[numeric_columns].fillna(preview[numeric_columns].median())
                        st.write("Preview after filling with median:")

                    elif missing_option == 'Fill with Mode':
                        preview[columns_to_apply] = preview[columns_to_apply].fillna(preview[columns_to_apply].mode().iloc[0])
                        st.write("Preview after filling with mode:")

                    elif missing_option == 'Fill with Custom Value' and custom_value is not None:
                        preview[numeric_columns] = preview[numeric_columns].fillna(custom_value)
                        st.write(f"Preview after filling numeric columns with custom value: {custom_value}")

                    st.dataframe(preview)

                    # Provide an explicit confirmation option before applying changes
                    apply_changes = st.button(f"Apply to {dataset_to_discover}")

                    if apply_changes:
                        confirmation = st.radio(
                            "Are you sure you want to apply these changes? This action cannot be undone.",
                            ("No", "Yes")
                        )

                        if confirmation == "Yes":
                            datasets[dataset_to_discover] = preview
                            st.success(f'Missing data in {dataset_to_discover} handled using {missing_option}.')
                            st.write(f"Updated data for {dataset_to_discover}:")
                            st.dataframe(datasets[dataset_to_discover])
                        else:
                            st.warning("Changes were not applied.")

            else:
                st.success(f"No missing data detected in {dataset_to_discover}.")
                
            st.subheader(":orange[📊Outlier Detection]", divider='gray')

            numerical_columns = data.select_dtypes(include='number').columns
            if not numerical_columns.any():
                st.warning("No numerical columns available for outlier detection.")
            else:
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

                st.success("Data Cleaning Operations Completed!")

        with tab5:
            # Column value counts and visualization
            st.subheader(f":orange[Column Values To Count]", divider='orange')
            with st.expander(":gray[Column Values To Count]"):
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox('Column Values To Count', options=list(data.columns))
                with col2:
                    toprows = st.number_input('Top Rows', min_value=1, step=1)

                count = st.button('Count')
                if count:
                    plot_value_counts(data, column, toprows)

        with tab6:
            st.subheader(f":orange[Groupby: Simplify your data analysis]", divider='orange')
            st.write('The groupby lets you summarize data by specific categories and groups')
            with st.expander('Groupby'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    groupby_cols = st.multiselect('Choose your column to groupby', options=list(data.columns))
                with col2:
                    operation_col = st.selectbox('Choose column for operation', options=list(data.columns))
                with col3:
                    operation = st.selectbox('Operation', options=['sum', 'max', 'min', 'mean', 'median', 'count'])

                if groupby_cols:
                    result = data.groupby(groupby_cols).agg(
                        newcol=(operation_col, operation)
                    ).reset_index()

                    st.dataframe(result)

                    st.subheader(f":gray Visualization", divider='gray')
                    graphs = st.selectbox('Graphs', options=['line', 'bar', 'scatter', 'pie', 'sunburst'])
                    if graphs == 'line':
                        x_axis = st.selectbox('choose_x', options=list(result.columns))
                        y_axis = st.selectbox('choose_y', options=list(result.columns))
                        color = st.selectbox('color_info', options=[None] + list(result.columns))
                        fig = px.line(data_frame=result, x=x_axis, y=y_axis, color=color, markers='o')
                        st.plotly_chart(fig)
                    elif graphs == 'bar':
                        x_axis = st.selectbox('choose_x', options=list(result.columns))
                        y_axis = st.selectbox('choose_y', options=list(result.columns))
                        color = st.selectbox('color_info', options=[None] + list(result.columns))
                        facet_col = st.selectbox('facet_col', options=[None] + list(result.columns))
                        fig = px.bar(data_frame=result, x=x_axis, y=y_axis, color=color, facet_col=facet_col, barmode='group')
                        st.plotly_chart(fig)
                    elif graphs == 'scatter':
                        x_axis = st.selectbox('choose_x', options=list(result.columns))
                        y_axis = st.selectbox('choose_y', options=list(result.columns))
                        color = st.selectbox('color_info', options=[None] + list(result.columns))
                        size = st.selectbox('size_column', options=[None] + list(result.columns))
                        fig = px.scatter(data_frame=result, x=x_axis, y=y_axis, color=color, size=size)
                        st.plotly_chart(fig)
                    elif graphs == 'pie':
                        values = st.selectbox('numerical_values', options=list(result.columns))
                        names = st.selectbox('labels', options=list(result.columns))
                        fig = px.pie(data_frame=result, values=values, names=names)
                        st.plotly_chart(fig)
                    elif graphs == 'sunburst':
                        path = st.multiselect('Path', options=list(result.columns))
                        fig = px.sunburst(data_frame=result, path=path, values='newcol')
                        st.plotly_chart(fig)

        # Option to combine datasets
        if len(datasets) > 1:
            st.header(":orange[🧬 Combine Datasets]" ,divider='orange')

            # Select datasets to combine
            datasets_to_combine = st.multiselect(
                "Select datasets to combine:",
                options=list(datasets.keys()),
                default=list(datasets.keys())
            )

            if len(datasets_to_combine) >= 2:
                # Select merge type
                merge_type = st.selectbox(
                    "Select how to combine datasets:",
                    options=["Vertical (Append)", "Horizontal (Merge)"]
                )

                if st.button("Combine Datasets"):
                    try:
                        if merge_type == "Vertical (Append)":
                            # Ensure all datasets have the same columns
                            columns_set = set(datasets[datasets_to_combine[0]].columns)
                            if all(set(datasets[name].columns) == columns_set for name in datasets_to_combine):
                                combined_data = pd.concat([datasets[name] for name in datasets_to_combine], ignore_index=True)
                                st.success("Datasets combined successfully!")
                                st.write(f"Combined dataset shape: {combined_data.shape}")
                                st.write(combined_data.head())
                                
                                # Convert the combined DataFrame to CSV
                                csv_data = combined_data.to_csv(index=False).encode('utf-8')

                                # Add a download button
                                st.download_button(
                                    label="Download Combined Dataset",
                                    data=csv_data,
                                    file_name="combined_dataset.csv",
                                    mime="text/csv"
                                )

                            else:
                                st.error("All selected datasets must have the same columns for vertical append.")
                        else:
                            # Horizontal merge on a common key
                            common_columns = set(datasets[datasets_to_combine[0]].columns)
                            for name in datasets_to_combine[1:]:
                                common_columns = common_columns.intersection(set(datasets[name].columns))
                            
                            if common_columns:
                                merge_key = st.selectbox("Select the key column to merge on:", options=list(common_columns))
                                combined_data = datasets[datasets_to_combine[0]]
                                for name in datasets_to_combine[1:]:
                                    combined_data = pd.merge(combined_data, datasets[name], on=merge_key, how='outer')
                                st.success("Datasets merged successfully!")
                                st.write(f"Combined dataset shape: {combined_data.shape}")
                                st.write(combined_data.head())
                                
                                # Convert the combined DataFrame to CSV
                                csv_data = combined_data.to_csv(index=False).encode('utf-8')

                                # Add a download button
                                st.download_button(
                                    label="Download Combined Dataset",
                                    data=csv_data,
                                    file_name="combined_dataset.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No common columns found to perform horizontal merge.")
                    except Exception as e:
                        st.error(f"Error combining datasets: {e}")


# Sidebar: Add logo
st.sidebar.image('./media/second_logo.jpeg', use_column_width=True)

st.sidebar.write("")

# Signature
st.sidebar.write("")


st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <a href="https://datadiwan.com">
            <img src="https://upload.wikimedia.org/wikipedia/commons/d/d1/Favicon.ico.png" alt="Favicon" width="40">
        </a>
        &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/datadiwan">
            <img src="https://upload.wikimedia.org/wikipedia/commons/2/24/Github_logo_svg.svg" width="40">
        </a>
        &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="mailto:sawsan.abdulbari@gmail.com">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" width="40">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css(r"style/style.css")
# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    color: white;
    text-align: center;
}
</style>
<div class="footer">
<p>Developed with <span style='color:green;'>❤</span> by <a href="https://www.datadiwan.com/" target="_blank">Data Diwan</a> </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.sidebar.markdown("""

<div>
<p>Developed with <span style='color:green;'>❤</span> by <a href="https://www.datadiwan.com/" target="_blank">Data diwan</a> </p>
</div>
""", unsafe_allow_html=True)