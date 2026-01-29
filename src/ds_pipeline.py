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

#Step 3: Add the model with Early stopping
from sklearn.neural_network import MLPRegressor

mlp=MLPRegressor(random_state=42,
                 hidden_layer_sizes=(10,5), # Added hidden layer sizes as it is a requirement
                 max_iter=200,
                 batch_size=1000, # Added a batch size
                 early_stopping=True)

mlp.fit(X_train_scaled, y_train)

# Step 4: Add train, validation and test predictions and plot
import numpy as np

y_pred_train = mlp.predict(X_train_scaled)
y_pred_val   = mlp.predict(X_val_scaled)
y_pred_test  = mlp.predict(X_test_scaled)

# Plot
# Scatterplots: predicted vs actual 
def scatter_with_reference(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1, color='red')  # reference line
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.tight_layout()

#Plot for train parition
scatter_with_reference(y_train, y_pred_train, "Predicted vs Actual — Train")
plt.savefig("figures/train_actual_vs_predicted.png") # saved train scatterplot image
plt.show()
plt.close()

#Plot for validation partition
scatter_with_reference(y_val, y_pred_val, "Predicted vs Actual — Validation")
plt.savefig("figures/Validation_actual_vs_predicted.png") # saved validation scatterplot image
plt.show()
plt.close()

#Plot for test partition
scatter_with_reference(y_test, y_pred_test, "Predicted vs Actual — Test")
plt.savefig("figures/Test_actual_vs_predicted.png") # saved test scatterplot image
plt.show()
plt.close()

