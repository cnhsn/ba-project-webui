# Student Outcome Analysis App

**Student Outcome Analysis App** is a project for the **Business Analytics** class at **Riga Technical University (RTU)**. This application allows users to explore, classify, and compare student outcome data, with additional AI-powered summaries for deeper insights. 

## Project Information

- **Class**: Business Analytics, Riga Technical University
- **Project Owners**: Hasan Can, Burak Caka, Vepa Tuliyev
- **Future Development**: No further development planned beyond the initial submission

## Overview

This Streamlit-based application offers several core functionalities:
1. **Dataset Filtering**: Users can filter data by selected columns and values.
2. **Data Visualization**: Options for visualizing data as histograms, scatter plots, or box plots.
3. **Data Comparison**: Group data by a chosen column and apply aggregate functions like mean, median, or count.
4. **AI-Powered Summary**: Generate textual summaries for visualized data using OpenRouter.ai’s API.

## Features

- **Visualization Options**: Choose from histogram, scatter plot, or box plot.
- **Comparison Tool**: Group data and apply aggregate functions.
- **AI Summary Generation**: Get text-based summaries of visualizations powered by OpenRouter.ai.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/student-outcome-analysis.git
   cd student-outcome-analysis
   ```

2. **Install Required Packages**:
   Install the necessary libraries via `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up OpenRouter API Key**:
   - Obtain an API key from [OpenRouter.ai](https://openrouter.ai/).
   - Store the API key as an environment variable:
     ```bash
     export OPENROUTER_API_KEY="your_api_key"
     ```
     Alternatively, add this line to a `.env` file for local development:
     ```plaintext
     OPENROUTER_API_KEY=your_api_key
     ```

4. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open your browser and go to `http://localhost:8501` to view the app.

## Usage

1. **Upload a Dataset**: Upload a CSV file from your local machine.
2. **Filter Data**: Select columns to filter and apply specific values to refine the dataset view.
3. **Choose Visualization**: Select a chart type (Histogram, Scatter Plot, Box Plot) and customize columns for visualization.
4. **Use the Comparison Tool**: Group data by a chosen column and apply an aggregate function (mean, median, count).
5. **Generate AI Summary**: Click on the “Generate AI Summary” button under each visualization or comparison to get a brief AI-generated summary.

## Requirements

- **Python 3.8+**
- **Libraries**:
  - `streamlit`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `requests`

## License

This project is developed for educational purposes only and may not be reused commercially or modified beyond the requirements of the RTU Business Analytics class.
