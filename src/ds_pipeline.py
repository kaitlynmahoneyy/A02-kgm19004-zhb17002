# Step 1: Read the data
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# you can save the boxplot...
plt.figure(figsize=(12,8))
df["MedHouseVal"].plot.box()
plt.title("Median House Value Boxplot")
plt.ylabel("Median House Value")

plt.tight_layout()
plt.savefig("figures/MedHouseVal_boxplot.png") # saved boxplot image
plt.close()

# Step 2: Splitting the data into test and training sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = housing.frame.drop(columns=["MedHouseVal"])
y = housing.frame["MedHouseVal"]

# train
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# test and val
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

#Quick check
print(X_train_scaled)
