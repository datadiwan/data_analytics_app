import streamlit as st
import plotly.express as px



def plot_value_counts(data, column, num_rows, txt):
    """Plots the value counts for a given column."""
    result = data[column].value_counts().reset_index().head(num_rows)
    result.columns = ['index', 'count']  # Rename columns for clarity

    st.dataframe(result)

    st.subheader(txt['visualization'], divider='gray')

    # Create the bar chart
    fig = px.bar(data_frame=result, x='index', y='count', text='count', template='plotly_white')
    st.plotly_chart(fig)

    # Create the line chart
    fig = px.line(data_frame=result, x='index', y='count', text='count', template='plotly_white')
    st.plotly_chart(fig)

    # Create the pie chart
    fig = px.pie(data_frame=result, names='index', values='count')
    st.plotly_chart(fig)

def plot_grouped_data(data, groupby_cols, operation_col, operation, txt):
    """Performs a groupby operation and visualizes the results."""
    if groupby_cols:
        result = data.groupby(groupby_cols).agg(
            newcol=(operation_col, operation)
        ).reset_index()

        st.dataframe(result)

        st.subheader(txt['visualization'], divider='gray')

        # Choose the type of plot to generate
        plot_type = st.selectbox(txt['graphs'], options=['line', 'bar', 'scatter', 'pie', 'sunburst'])
        if plot_type == 'line':
            fig = px.line(result, x=groupby_cols[0], y='newcol', template='plotly_white')
        elif plot_type == 'bar':
            fig = px.bar(result, x=groupby_cols[0], y='newcol', template='plotly_white')
        elif plot_type == 'scatter':
            fig = px.scatter(result, x=groupby_cols[0], y='newcol', template='plotly_white')
        elif plot_type == 'pie':
            fig = px.pie(result, names=groupby_cols[0], values='newcol')
        elif plot_type == 'sunburst':
            fig = px.sunburst(result, path=groupby_cols, values='newcol')

        st.plotly_chart(fig)
    else:
        st.warning(txt['groupby_explain'])
def plot_correlation_heatmap(data, selected_columns, method, color_scale):
    """Plots a correlation heatmap for selected columns."""
    corr_matrix = data[selected_columns].corr(method=method.lower())
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale=color_scale)
    st.plotly_chart(fig)


def plot_treemap(data, hierarchy_columns, size_column, color_column=None, color_scale='Viridis'):
    """Creates a treemap based on the selected hierarchy and size columns."""
    if hierarchy_columns and size_column:
        fig = px.treemap(
            data, 
            path=hierarchy_columns, 
            values=size_column, 
            color=color_column, 
            color_continuous_scale=color_scale if color_column else None
        )

        st.plotly_chart(fig)
    else:
        st.error("Please select at least one hierarchy column and a size column.")
        
def plot_animated_line_chart(data, time_variable, value_variable, category_variable=None):
    """Creates an animated line chart based on the given variables."""
    if time_variable and value_variable:
        fig = px.line(
            data, 
            x=time_variable, 
            y=value_variable, 
            color=category_variable if category_variable else None,
            animation_frame=time_variable, 
            animation_group=category_variable if category_variable else None,
            template='plotly_white'
        )

        fig.update_layout(
            title=f"Animated Plot of {value_variable} over {time_variable}",
            xaxis_title=time_variable,
            yaxis_title=value_variable
        )

        st.plotly_chart(fig)
    else:
        st.error("Please select both a time variable and a value variable.")