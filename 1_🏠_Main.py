import base64
import os
import streamlit as st

# Title and subtitle
st.title(':orange[Data Scientist App]:robot_face:')
st.sidebar.title("The Data scientist App📊📈")

# Sidebar: Add logo
st.sidebar.image('./media/logo.jpeg', use_column_width=True)

st.sidebar.write("")

# Signature
st.sidebar.write("")

# Insert the image at the top of the page
logo_path="./media/logo1.png"
# Add logo at the top of the page
# Read the image file
with open(logo_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Documentation Section
st.markdown("## 📚 Documentation")

# Introduction section
st.info("""
This app empowers you to interactively analyze your datasets. 
Upload your data and utilize a wide range of features such as summary statistics, 
visualizations, grouping operations, and data cleaning tools.
""")

# Step-by-step instructions
st.markdown("### 🔍 How to Use the App")
st.markdown("""
1. **Upload Data**: Start by uploading a CSV or Excel file using the file uploader on the main page.
2. **Explore Basic Information**: View the summary statistics, top and bottom rows, data types, and column names.
3. **Clean Your Data**: Address missing values, remove duplicates, and detect outliers using the provided tools.
4. **Generate Visualizations**: Create various plots like bar charts, scatter plots, line charts, pie charts, and more to visualize your data.
5. **Group and Analyze**: Aggregate your data by specific columns and perform operations like sum, mean, median, and more.
6. **Combine Datasets**: If you have multiple datasets, use the "Combine Datasets" feature to merge or append them together.
7. **Download Results**: After processing, you can download the combined or cleaned dataset for further use.
""")

# Example datasets
st.markdown("### 📂 Example Datasets")
st.markdown("""
If you don't have your own dataset, you can download and use one of the following example datasets:
- [Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data): A classic dataset for classification, contains 150 samples of iris flowers. Note: This will open as a text file, so you might need to save it manually as a `.csv`.
- [Titanic Dataset](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv): Information about the passengers on the Titanic, used for survival prediction.
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv): Contains data on red and white variants of Portuguese "Vinho Verde" wine.
""")

# Additional resources and links
st.markdown("### 📖 Additional Resources")
st.markdown("""
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Streamlit Documentation](https://docs.streamlit.io/)
""")

# Feedback section
# Feedback section header
st.markdown("### 💬 Feedback")
# Info message below the form
st.info("Have suggestions or found a bug? Let us know by sending feedback through this form.")

# Contact form for feedback
contact_form = """
<form action="https://formsubmit.co/104e90140f4e1b95b596d226cbacefa9" method="POST">
    <input type="hidden" name="_captcha" value="true">
    <input type="text" name="name" placeholder="Your name" required>
    <input type="email" name="email" placeholder="Your email" required>
    <textarea name="message" placeholder="Your message here" required></textarea>
    <button type="submit">Send Feedback</button>
</form>
"""
# Render the contact form in the Streamlit app
st.markdown(contact_form, unsafe_allow_html=True)


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
        <a href="mailto:infodatadiwan@gmail.com">
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
<p>Developed with <span style='color:green;'>❤</span> by <a href="https://www.datadiwan.com/" target="_blank">Data diwan</a> </p>
</div>
""", unsafe_allow_html=True)


# Footer
st.sidebar.markdown("""

<div>
<p>Developed with <span style='color:green;'>❤</span> by <a href="https://www.datadiwan.com/" target="_blank">Data diwan</a> </p>
</div>
""", unsafe_allow_html=True)
