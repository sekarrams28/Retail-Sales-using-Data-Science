# Retail Sales Analysis and Prediction

## Project Overview

This is a real-world data science project based on retail sales.

The project performs:

- Data loading
- Data cleaning
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Business analysis
- Data visualization
- Correlation analysis
- Machine learning
- Revenue prediction

## Dataset

The project includes a generated retail dataset with 1,000 realistic sales records.

Columns:

- `Order_ID`
- `Order_Date`
- `Product`
- `Category`
- `Quantity`
- `Price`
- `Discount`
- `Customer_ID`
- `Region`
- `Payment_Mode`
- `Revenue`

Revenue is calculated using:

`Revenue = Quantity × Price × (1 - Discount)`

## Machine Learning

Three regression algorithms are included:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The target variable is `Revenue`.

Features used:

- Quantity
- Price
- Discount

Evaluation metrics:

- MAE
- RMSE
- R² Score

## Visualizations

The project generates:

- Monthly revenue
- Revenue by category
- Top 10 products
- Revenue by region
- Discount vs revenue
- Correlation matrix
- Actual vs predicted revenue

All graphs are saved in the `visualizations` folder.

## Project Structure

```text
Retail-Sales-Data-Science/
│
├── dataset/
│   └── retail_sales.csv
│
├── src/
│   ├── retail_analysis.py
│   └── model_results.csv
│
├── visualizations/
│   ├── monthly_revenue.png
│   ├── revenue_by_category.png
│   ├── top_10_products.png
│   ├── revenue_by_region.png
│   ├── discount_vs_revenue.png
│   ├── correlation_matrix.png
│   └── actual_vs_predicted.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Install Python 3.10 or newer.

Open Command Prompt or PowerShell inside the project folder.

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python src/retail_analysis.py
```

The program will:

1. Load the CSV
2. Display data quality information
3. Perform business analysis
4. Generate visualizations
5. Train three ML models
6. Display evaluation metrics
7. Ask for quantity, price and discount
8. Predict revenue using Random Forest

## Example Prediction Input

```text
Enter quantity: 5
Enter price: 1200
Enter discount as decimal (example: 0.10): 0.10
```

The program will display the predicted revenue.

## Important Note

The included CSV is a synthetic educational dataset created for this project. It is designed to demonstrate an end-to-end real-world data science workflow and should not be treated as actual company sales data.

## Learning Outcomes

After completing this project, you will understand:

- How to work with CSV datasets
- Data cleaning with Pandas
- Exploratory Data Analysis
- Data visualization with Matplotlib
- Correlation analysis
- Regression models
- Train/test splitting
- Model evaluation
- Interactive prediction
- Organizing a data science project for GitHub

## Author

Built by Sekar Ram
