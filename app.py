import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os

# Load the dataset
data = pd.read_csv("data.csv", delimiter=";")

# Define the OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Ensure this environment variable is set

# Function to generate AI summary using OpenRouter's API
def generate_summary(prompt):
    if not OPENROUTER_API_KEY:
        st.error("API key for OpenRouter is missing. Please set it as an environment variable.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Updated payload with the specified model and refined prompt
    payload = {
        "model": "liquid/lfm-40b:free",
        "messages": [{"role": "user", "content": f"{prompt}. Please provide only summary insights in plain text without any code or examples."}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        # Extract the response text and clean any unwanted sections if necessary
        summary_text = response.json()["choices"][0]["message"]["content"].strip()
        
        # Post-process to remove any code if present (optional)
        import re
        summary_text = "\n".join([line for line in summary_text.splitlines() if not re.match(r"^\s*(import|#|plt|sns|df|data\.)", line)])
        
        return summary_text

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with OpenRouter API: {e}")
        return None

# Define app layout and title
st.title("Student Outcome Analysis")
st.write("This app allows you to explore, classify, and compare data on student outcomes.")

# Notice
st.sidebar.markdown("### Important Notice")
st.sidebar.markdown("""
This application developed for Business Analytics class at RTU by Hasan Can, Burak Caka and Vepa Tuliyev. Used dataset is not contains any sensitive information, we are using publicly available dataset!
""")

# Instructions
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Select columns to filter data.
2. Choose a visualization type and configure it in the sidebar.
3. Use the comparison tool to compare different values.
4. Click on 'Generate AI Summary' buttons to get a summary of the displayed data.
""")

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

    # AI Summary button
    if st.button("Generate AI Summary for Histogram"):
        prompt = f"Provide a summary analysis for the histogram of the column '{column}' in the dataset."
        summary = generate_summary(prompt)
        if summary:
            st.write("AI Summary:", summary)

elif plot_type == "Scatter Plot":
    x_col = st.sidebar.selectbox("Choose X-axis for Scatter Plot", columns)
    y_col = st.sidebar.selectbox("Choose Y-axis for Scatter Plot", columns)
    st.write(f"Scatter Plot of {x_col} vs {y_col}")
    fig, ax = plt.subplots()
    ax.scatter(filtered_data[x_col].astype(float), filtered_data[y_col].astype(float), alpha=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

    # AI Summary button
    if st.button("Generate AI Summary for Scatter Plot"):
        prompt = f"Provide a summary analysis for the scatter plot between '{x_col}' and '{y_col}' in the dataset."
        summary = generate_summary(prompt)
        if summary:
            st.write("AI Summary:", summary)

elif plot_type == "Box Plot":
    column = st.sidebar.selectbox("Choose column for Box Plot", columns)
    st.write(f"Box Plot of {column}")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_data[column].astype(float), ax=ax)
    st.pyplot(fig)

    # AI Summary button
    if st.button("Generate AI Summary for Box Plot"):
        prompt = f"Provide a summary analysis for the box plot of the column '{column}' in the dataset."
        summary = generate_summary(prompt)
        if summary:
            st.write("AI Summary:", summary)

# Custom comparison tool
st.sidebar.header("Comparison Tool")
comparison_col = st.sidebar.selectbox("Choose column to compare by", columns)
comparison_metric = st.sidebar.selectbox("Choose comparison metric", ["mean", "median", "count"])

if st.sidebar.button("Run Comparison"):
    comparison_data = filtered_data.groupby(comparison_col).agg({comparison_col: comparison_metric})
    st.write(f"{comparison_metric.capitalize()} comparison for {comparison_col}", comparison_data)

    # AI Summary button for comparison
    if st.button("Generate AI Summary for Comparison"):
        prompt = f"Provide a summary analysis for the {comparison_metric} comparison based on the column '{comparison_col}' in the dataset."
        summary = generate_summary(prompt)
        if summary:
            st.write("AI Summary:", summary)

# Sidebar options for dataset preview
st.sidebar.header("Dataset Options")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(data)
