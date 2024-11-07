import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Note: Replace this with your own path or loading method as required.
data = pd.read_csv("data.csv", delimiter=";")

# Define app layout and title
st.title("Student Outcome Analysis")
st.write("This app allows you to explore, classify, and compare data on student outcomes.")

# Notice
st.sidebar.markdown("### Important Notice")
st.sidebar.markdown("""
This application developed for Business Analytics class on RTU by Hasan Can, Burak Caka and Vepa Tuliyev. Used dataset is not contains any sensitive information, we are using publicly available dataset!
""")

# Sidebar options for dataset preview
st.sidebar.header("Dataset Options")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(data)

# Sidebar options for filtering and classification
st.sidebar.header("Filter Options")
columns = data.columns.tolist()

# Select columns for classification
selected_columns = st.sidebar.multiselect("Select columns to filter", columns)
filters = {}
for col in selected_columns:
    unique_values = data[col].unique()
    selected_values = st.sidebar.multiselect(f"Filter {col}", unique_values)
    if selected_values:
        filters[col] = selected_values

# Apply filters to data
if filters:
    filtered_data = data
    for col, values in filters.items():
        filtered_data = filtered_data[filtered_data[col].isin(values)]
    st.write("Filtered Data", filtered_data)
else:
    filtered_data = data

# Visualization options
st.sidebar.header("Visualization Options")
plot_type = st.sidebar.selectbox("Choose plot type", ["Histogram", "Scatter Plot", "Box Plot"])

# Visualization based on plot type
if plot_type == "Histogram":
    column = st.sidebar.selectbox("Choose column for Histogram", columns)
    st.write(f"Histogram of {column}")
    fig, ax = plt.subplots()
    filtered_data[column].astype(float).hist(bins=20, ax=ax, edgecolor="black")
    st.pyplot(fig)

elif plot_type == "Scatter Plot":
    x_col = st.sidebar.selectbox("Choose X-axis for Scatter Plot", columns)
    y_col = st.sidebar.selectbox("Choose Y-axis for Scatter Plot", columns)
    st.write(f"Scatter Plot of {x_col} vs {y_col}")
    fig, ax = plt.subplots()
    ax.scatter(filtered_data[x_col].astype(float), filtered_data[y_col].astype(float), alpha=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

elif plot_type == "Box Plot":
    column = st.sidebar.selectbox("Choose column for Box Plot", columns)
    st.write(f"Box Plot of {column}")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_data[column].astype(float), ax=ax)
    st.pyplot(fig)

# Custom comparison tool
st.sidebar.header("Comparison Tool")
comparison_col = st.sidebar.selectbox("Choose column to compare by", columns)
comparison_metric = st.sidebar.selectbox("Choose comparison metric", ["mean", "median", "count"])

if st.sidebar.button("Run Comparison"):
    comparison_data = filtered_data.groupby(comparison_col).agg({comparison_col: comparison_metric})
    st.write(f"{comparison_metric.capitalize()} comparison for {comparison_col}", comparison_data)

# Instructions
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Select columns to filter data.
2. Choose a visualization type and configure it in the sidebar.
3. Use the comparison tool to compare different values.
""")
