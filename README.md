# 📊 Automated EDA Report Generator

Generates rich, dark-themed HTML reports from any CSV or DataFrame — with **AI-powered natural language insights** via Google Gemini.

---

## Features

| Feature | Details |
|---|---|
| 📈 Charts | Histograms, boxplots, correlation heatmap, categorical bar charts, missing value map |
| 🤖 AI Insights | Gemini analyzes your data and writes a plain-English narrative (skewness, outliers, correlations, data quality warnings, suggested next steps) |
| 📋 Column Stats | Per-column: dtype, missingness, unique count, mean/median/std/skewness/outliers |
| 🔗 Correlation Table | Top 10 strongest pairs ranked by Pearson r |
| 💾 Single-file output | Everything embedded in one self-contained `.html` file |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key

```bash
export GEMINI_API_KEY="AIza..."
```

Alternatively, create a local `.env` file at the project root with:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

A `.env.example` file is included; copy it to `.env` and fill in your key.
The repository's `.gitignore` excludes `.env` and the virtualenv.

> Get a free key at https://aistudio.google.com/app/apikey  
> The report still works without a key — AI insights section will be skipped.

### 3. Generate a sample dataset (optional)

```bash
python sample_data/generate_sample.py
```

### 4a. Run from CLI

```bash
python src/eda_generator.py sample_data/employees.csv -o reports/report.html -n "Employees"
```

### 4b. Run from Python

```python
from src.eda_generator import generate_report

generate_report(
    source="your_data.csv",
    output_path="reports/my_report.html",
    dataset_name="My Dataset",
)
```

---

## Project Structure

```
eda_generator/
├── src/
│   └── eda_generator.py      # Core engine
├── sample_data/
│   └── generate_sample.py    # Generates a test CSV
├── reports/                  # Output reports saved here
├── example_usage.py          # Python usage example
├── requirements.txt
└── README.md
```

---

## CLI Options

```
python src/eda_generator.py <csv> [-o OUTPUT] [-n NAME] [-k API_KEY]

positional:
  csv             Path to input CSV file

optional:
  -o, --output    Output HTML path (default: eda_report.html)
  -n, --name      Dataset display name
  -k, --api-key   Gemini API key (or set GEMINI_API_KEY env var)
```

---

## Extending the Project

- **Multi-file comparison** — diff two versions of the same dataset
- **Time-series detection** — auto-detect datetime columns and plot trends
- **Target-aware mode** — pass a target column and get feature importance, class balance, leakage warnings
- **PDF export** — use `weasyprint` to convert the HTML report to PDF
- **Streamlit UI** — drag-and-drop CSV, instant report in browser
- **Schema drift alerts** — compare current stats against a saved baseline

---

## Output Preview

The report is a single `.html` file with:
- Dark theme, monospace + DM Sans typography
- Embedded base64 charts (no external dependencies)
- AI narrative with column-level observations (powered by Gemini)
- Color-coded correlation table (strong = pink, medium = cyan)
