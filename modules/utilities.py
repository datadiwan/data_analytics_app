# import pandas as pd
# import streamlit as st

# def load_dataset(file):
#     if file.name.endswith('.csv'):
#         return pd.read_csv(file)
#     elif file.name.endswith(('.xls', '.xlsx')):
#         return pd.read_excel(file)
#     else:
#         st.error(f"Unsupported file format: {file.name}")
#         return None
import pandas as pd
import streamlit as st

def load_dataset(file, parse_dates=False, date_column=None):
    """
    Loads a dataset from a file. Supports CSV and Excel formats.
    
    Parameters:
    - file: The file to load.
    - parse_dates (bool): Whether to parse dates or not.
    - date_column (str): The column to parse as dates and set as index if parse_dates is True.
    
    Returns:
    - pd.DataFrame: The loaded dataset.
    """
    if file.name.endswith('.csv'):
        if parse_dates and date_column:
            return pd.read_csv(file, parse_dates=[date_column], index_col=date_column)
        else:
            return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        if parse_dates and date_column:
            return pd.read_excel(file, parse_dates=[date_column], index_col=date_column)
        else:
            return pd.read_excel(file)
    else:
        st.error(f"Unsupported file format: {file.name}")
        return None

# Example usage in Streamlit
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xls', 'xlsx'])

if uploaded_file is not None:
    parse_dates = st.sidebar.checkbox("Parse dates?")
    date_column = None
    if parse_dates:
        date_column = st.sidebar.text_input("Enter the name of the date column")

    data = load_dataset(uploaded_file, parse_dates=parse_dates, date_column=date_column)
    st.write("Dataset Loaded:")
    st.dataframe(data)
