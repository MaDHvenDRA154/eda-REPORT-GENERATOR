"""
Generate a realistic sample dataset for testing the EDA report generator.
Run: python sample_data/generate_sample.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 1500

age        = np.random.normal(38, 12, n).clip(18, 75).astype(int)
income     = (np.random.lognormal(10.5, 0.6, n)).astype(int)
experience = (age - 18 - np.random.exponential(3, n)).clip(0, 40).astype(int)
credit     = (600 + 0.8 * income / 1000 + 2 * experience
              + np.random.normal(0, 40, n)).clip(300, 850).astype(int)

dept_map = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
dept_weights = [0.30, 0.18, 0.20, 0.10, 0.12, 0.10]
department = np.random.choice(dept_map, n, p=dept_weights)

edu_map = ["High School", "Bachelor's", "Master's", "PhD"]
education = np.random.choice(edu_map, n, p=[0.15, 0.50, 0.28, 0.07])

# Inject ~8% missing values in a few columns
loan_amount = np.random.exponential(25000, n).clip(1000, 250000).astype(float)
loan_amount[np.random.choice(n, int(n * 0.08), replace=False)] = np.nan

satisfaction = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.25, 0.38, 0.22])
satisfaction = satisfaction.astype(float)
satisfaction[np.random.choice(n, int(n * 0.05), replace=False)] = np.nan

churn = ((income < 40000) & (satisfaction < 3) |
         (np.random.rand(n) < 0.07)).astype(int)

df = pd.DataFrame({
    "employee_id":   range(10001, 10001 + n),
    "age":           age,
    "department":    department,
    "education":     education,
    "years_exp":     experience,
    "annual_income": income,
    "credit_score":  credit,
    "loan_amount":   loan_amount,
    "satisfaction":  satisfaction,
    "churned":       churn,
})

out = Path(__file__).parent / "employees.csv"
df.to_csv(out, index=False)
print(f"Sample dataset saved → {out}  ({len(df)} rows)")
