"""
Example: how to use the EDA generator from Python code.
"""

import os
import sys
sys.path.insert(0, "src")

from eda_generator import generate_report
import pandas as pd

# ── Example 1: from a CSV file ────────────────────────────────────────────────
generate_report(
    source="sample_data/employees.csv",
    output_path="reports/employees_eda.html",
    dataset_name="Employee Dataset",
    # api_key="sk-ant-..."   # or set ANTHROPIC_API_KEY env var
)

# ── Example 2: from a DataFrame ──────────────────────────────────────────────
# df = pd.read_csv("your_data.csv")
# generate_report(df, "reports/my_report.html", dataset_name="My Data")
