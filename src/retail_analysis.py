"""
Retail Sales Analysis and Prediction
Run:
    python src/retail_analysis.py
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "retail_sales.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n========== RETAIL SALES DATASET ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nMissing values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# Basic statistics
print("\n========== BUSINESS SUMMARY ==========")
print(f"Total Revenue: {df['Revenue'].sum():,.2f}")
print(f"Average Order Revenue: {df['Revenue'].mean():,.2f}")
print(f"Total Quantity Sold: {df['Quantity'].sum():,}")
print(f"Unique Customers: {df['Customer_ID'].nunique():,}")

print("\nRevenue by Category:")
print(df.groupby("Category")["Revenue"].sum().sort_values(ascending=False))

print("\nRevenue by Region:")
print(df.groupby("Region")["Revenue"].sum().sort_values(ascending=False))

print("\nTop 10 Products:")
print(df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10))

# EDA graphs
monthly = df.groupby("Month")["Revenue"].sum()
plt.figure(figsize=(10, 5))
monthly.plot(kind="line", marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_revenue.png"), dpi=150)
plt.close()

category = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
category.plot(kind="bar")
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "revenue_by_category.png"), dpi=150)
plt.close()

top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
top_products.plot(kind="bar")
plt.title("Top 10 Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_10_products.png"), dpi=150)
plt.close()

region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
region.plot(kind="bar")
plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "revenue_by_region.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["Discount"] * 100, df["Revenue"], alpha=0.45)
plt.title("Discount vs Revenue")
plt.xlabel("Discount (%)")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "discount_vs_revenue.png"), dpi=150)
plt.close()

# ML
X = df[["Quantity", "Price", "Discount"]]
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
    )
}

trained = {}

print("\n========== MODEL RESULTS ==========")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    trained[name] = model
    print(f"\n{name}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

# Interactive prediction
print("\n========== SALES PREDICTION ==========")
try:
    quantity = float(input("Enter quantity: "))
    price = float(input("Enter price: "))
    discount = float(input("Enter discount as decimal (example: 0.10): "))

    prediction = trained["Random Forest"].predict(
        [[quantity, price, discount]]
    )[0]

    print(f"\nPredicted Revenue: {prediction:,.2f}")
except ValueError:
    print("Invalid input. Please enter numeric values.")
