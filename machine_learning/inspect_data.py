
import pandas as pd

# Load the dataset
dataset_path = "machine_learning/data/ai4i2020.csv"
df = pd.read_csv(dataset_path)

# 1. Dataset dimensions
print("\n========== DATASET DIMENSIONS ==========")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("Dataset shape:", df.shape)

# 2. First five records
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# 3. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 4. Dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 5. Descriptive statistics
print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

# 6. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 7. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Number of duplicates:", df.duplicated().sum())

# 8. Target distribution
print("\n========== MACHINE FAILURE DISTRIBUTION ==========")
print(df["Machine failure"].value_counts())

print("\n========== INSPECTION COMPLETED ==========")
