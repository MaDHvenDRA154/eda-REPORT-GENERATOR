"""
Automated EDA Report Generator
Generates rich HTML reports with natural language insights using Google Gemini AI.
"""

import os
import json
import base64
import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')


def _json_serial(obj):
  """JSON serializer for NumPy and pandas types."""
  if isinstance(obj, (np.integer,)):
    return int(obj)
  if isinstance(obj, (np.floating,)):
    return float(obj)
  if isinstance(obj, (np.ndarray,)):
    return obj.tolist()
  if isinstance(obj, (np.bool_,)):
    return bool(obj)
  return str(obj)

# ── Color palette ──────────────────────────────────────────────────────────────
PALETTE = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7", "#3a0ca3",
           "#480ca8", "#560bad", "#b5179e", "#f72585", "#4361ee"]
sns.set_theme(style="darkgrid", palette=PALETTE)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none", dpi=130)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def compute_overview(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime"]).columns.tolist()

    return {
        "rows": len(df),
        "cols": len(df.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dt_cols": dt_cols,
        "missing_total": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }


def compute_column_stats(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        info = {
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isnull().sum()),
            "missing_pct": round(s.isnull().mean() * 100, 2),
            "unique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update({
                "mean": round(s.mean(), 4),
                "median": round(s.median(), 4),
                "std": round(s.std(), 4),
                "min": round(s.min(), 4),
                "max": round(s.max(), 4),
                "skewness": round(s.skew(), 4),
                "kurtosis": round(s.kurtosis(), 4),
            })
            # Outlier count via IQR
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            info["outliers_iqr"] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        else:
            info["top_values"] = s.value_counts().head(5).to_dict()
        rows.append(info)
    return rows


def compute_correlations(df: pd.DataFrame) -> dict:
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return {}
    corr = num_df.corr()
    # Top 10 strongest pairs (excluding self)
    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                 .stack()
                 .reset_index()
                 .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "r"}))
    pairs["abs_r"] = pairs["r"].abs()
    top = pairs.nlargest(10, "abs_r")[["col_a", "col_b", "r"]].to_dict("records")
    return {"matrix": corr.round(3).to_dict(), "top_pairs": top}


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHART GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

_BG = "#0f0f1a"
_GRID = "#1e1e3a"
_TEXT = "#e0e0ff"


def _style_ax(ax, title=""):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.title.set_color(_TEXT)
    ax.title.set_fontsize(11)
    if title:
        ax.set_title(title)
    if ax.get_xlabel():
        ax.xaxis.label.set_color(_TEXT)
    if ax.get_ylabel():
        ax.yaxis.label.set_color(_TEXT)


def chart_missing(df: pd.DataFrame) -> str:
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, max(3, len(miss) * 0.4)), facecolor=_BG)
    miss.plot.barh(ax=ax, color=PALETTE[1])
    ax.set_xlabel("Missing Fraction")
    _style_ax(ax, "Missing Values per Column")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_distributions(df: pd.DataFrame, max_cols: int = 6) -> str:
    num_cols = df.select_dtypes(include=np.number).columns[:max_cols].tolist()
    if not num_cols:
        return ""
    n = len(num_cols)
    cols_per_row = min(3, n)
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row,
                              figsize=(5 * cols_per_row, 3.5 * rows),
                              facecolor=_BG)
    axes = np.array(axes).flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        ax.set_facecolor(_BG)
        data = df[col].dropna()
        ax.hist(data, bins=30, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor="none")
        ax.axvline(data.mean(), color="#f72585", lw=1.5, linestyle="--", label="mean")
        ax.axvline(data.median(), color="#4cc9f0", lw=1.5, linestyle=":", label="median")
        _style_ax(ax, col)
        ax.legend(fontsize=7, facecolor=_GRID, labelcolor=_TEXT, framealpha=0.7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout(pad=1.5)
    return _fig_to_b64(fig)


def chart_correlation(df: pd.DataFrame) -> str:
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return ""
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8),
                                    max(5, len(corr) * 0.7)), facecolor=_BG)
    cmap = sns.diverging_palette(240, 350, s=90, l=40, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.4, linecolor=_GRID,
                annot_kws={"size": 8, "color": _TEXT},
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT, labelsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_categorical(df: pd.DataFrame, max_cols: int = 4) -> str:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns[:max_cols].tolist()
    if not cat_cols:
        return ""
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), facecolor=_BG)
    if n == 1:
        axes = [axes]
    for i, col in enumerate(cat_cols):
        ax = axes[i]
        ax.set_facecolor(_BG)
        vc = df[col].value_counts().head(10)
        vc.plot.bar(ax=ax, color=PALETTE[i % len(PALETTE)], edgecolor="none")
        _style_ax(ax, col)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(pad=1.5)
    return _fig_to_b64(fig)


def chart_boxplots(df: pd.DataFrame, max_cols: int = 6) -> str:
    num_cols = df.select_dtypes(include=np.number).columns[:max_cols].tolist()
    if not num_cols:
        return ""
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 1.2), 5), facecolor=_BG)
    ax.set_facecolor(_BG)
    data_to_plot = [df[c].dropna().values for c in num_cols]
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                    medianprops=dict(color="#f72585", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "fliers", "caps"]:
        for item in bp[element]:
            item.set_color(_TEXT)
    ax.set_xticks(range(1, len(num_cols) + 1))
    ax.set_xticklabels(num_cols, rotation=30, ha="right")
    _style_ax(ax, "Boxplots (Numerical Columns)")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LLM INSIGHT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(overview: dict, col_stats: list[dict],
                      correlations: dict, api_key: str | None = None) -> str:
    """Call Gemini to produce natural-language insights."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return "<p><em>No GEMINI_API_KEY found — skipping AI insights.</em></p>"

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    payload = {
      "overview": overview,
      "column_stats": col_stats[:20],
      "top_correlations": correlations.get("top_pairs", []),
    }
    payload_json = json.dumps(payload, indent=2, default=_json_serial)
    prompt = f"""You are a senior data scientist reviewing a dataset.
Given the following statistical summary in JSON, write a concise but insightful
EDA narrative (4-6 paragraphs) in plain HTML (use <p>, <strong>, <ul>, <li> tags only).
Cover:
1. Dataset shape and quality (missingness, duplicates, memory).
2. Most interesting distributions (skewness, outliers, unusual spreads).
3. Key correlations and what they might imply.
4. Potential data quality issues or warnings.
5. Suggested next steps for analysis or modelling.
Be specific — reference actual column names and numbers.
Return only the HTML — no markdown fences, no preamble.

JSON:
{payload_json}
"""
    response = model.generate_content(prompt)
    return response.text


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTML REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _img_tag(b64: str, alt: str = "") -> str:
    if not b64:
        return ""
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" class="chart-img">'


def build_html_report(df: pd.DataFrame,
                      dataset_name: str = "Dataset",
                      api_key: str | None = None) -> str:

    print("  → Computing statistics...")
    overview   = compute_overview(df)
    col_stats  = compute_column_stats(df)
    corr_data  = compute_correlations(df)

    print("  → Generating charts...")
    img_missing  = chart_missing(df)
    img_dist     = chart_distributions(df)
    img_corr     = chart_correlation(df)
    img_cat      = chart_categorical(df)
    img_box      = chart_boxplots(df)

    print("  → Calling Claude for insights...")
    insights_html = generate_insights(overview, col_stats, corr_data, api_key)

    # ── Column stats table rows ───────────────────────────────────────────────
    def stat_rows():
        rows_html = ""
        for s in col_stats:
            if pd.api.types.is_numeric_dtype(df[s["name"]]):
                extra = (f"mean={s.get('mean','—')} | median={s.get('median','—')} | "
                         f"std={s.get('std','—')} | skew={s.get('skewness','—')} | "
                         f"outliers={s.get('outliers_iqr','—')}")
            else:
                tv = s.get("top_values", {})
                extra = ", ".join(f"{k}: {v}" for k, v in list(tv.items())[:3])
            miss_cls = "warn" if s["missing_pct"] > 5 else ""
            rows_html += f"""
            <tr>
              <td><strong>{s['name']}</strong></td>
              <td>{s['dtype']}</td>
              <td class="{miss_cls}">{s['missing']} ({s['missing_pct']}%)</td>
              <td>{s['unique']}</td>
              <td class="extra-col">{extra}</td>
            </tr>"""
        return rows_html

    # ── Correlation table ─────────────────────────────────────────────────────
    def corr_rows():
        pairs = corr_data.get("top_pairs", [])
        if not pairs:
            return "<tr><td colspan='3'>No numerical columns to correlate.</td></tr>"
        html = ""
        for p in pairs:
            strength = abs(p["r"])
            cls = "strong-corr" if strength > 0.7 else ("mid-corr" if strength > 0.4 else "")
            html += f"<tr class='{cls}'><td>{p['col_a']}</td><td>{p['col_b']}</td><td>{p['r']:.4f}</td></tr>"
        return html

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>EDA Report — {dataset_name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:       #0a0a14;
    --surface:  #0f0f1e;
    --card:     #141428;
    --border:   #1e1e3a;
    --accent1:  #4361ee;
    --accent2:  #f72585;
    --accent3:  #4cc9f0;
    --text:     #d8d8f0;
    --muted:    #7070a0;
    --warn:     #ff9f43;
    --mono:     'Space Mono', monospace;
    --sans:     'DM Sans', sans-serif;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.7;
  }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, #0d0d22 0%, #0a0a1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 3rem;
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(67,97,238,.12) 0%, transparent 70%);
    pointer-events: none;
  }}
  header h1 {{
    font-family: var(--mono);
    font-size: 1.9rem;
    color: #fff;
    letter-spacing: -0.5px;
  }}
  header h1 span {{ color: var(--accent2); }}
  header p {{ color: var(--muted); margin-top: .3rem; font-size: 13px; }}

  /* ── Layout ── */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem 2rem 4rem; }}

  /* ── Section ── */
  section {{ margin-bottom: 2.8rem; }}
  .section-title {{
    font-family: var(--mono);
    font-size: .7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent1);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: .6rem;
  }}
  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* ── Overview cards ── */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 1rem;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color .2s;
  }}
  .stat-card:hover {{ border-color: var(--accent1); }}
  .stat-card .val {{
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent3);
  }}
  .stat-card .label {{ color: var(--muted); font-size: 12px; margin-top: .25rem; }}

  /* ── Insights box ── */
  .insights-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    line-height: 1.8;
  }}
  .insights-box p {{ margin-bottom: .9rem; }}
  .insights-box strong {{ color: var(--accent3); }}
  .insights-box ul {{ padding-left: 1.2rem; margin-bottom: .9rem; }}
  .insights-box li {{ margin-bottom: .3rem; }}

  /* ── Table ── */
  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 10px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    background: #0d0d22;
    color: var(--accent1);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: .7rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: .6rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,.02); }}
  .warn {{ color: var(--warn); }}
  .extra-col {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}
  .strong-corr td:last-child {{ color: var(--accent2); font-weight: 700; }}
  .mid-corr td:last-child {{ color: var(--accent3); }}

  /* ── Charts ── */
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 1.2rem;
  }}
  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    overflow: hidden;
  }}
  .chart-card h3 {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .8rem;
  }}
  .chart-img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}

  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding: 2rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <h1>EDA Report <span>//</span> {dataset_name}</h1>
  <p>Automated Exploratory Data Analysis — generated by eda_generator.py</p>
</header>

<div class="container">

  <!-- Overview -->
  <section>
    <p class="section-title">01 · Overview</p>
    <div class="stat-grid">
      <div class="stat-card"><div class="val">{overview['rows']:,}</div><div class="label">Rows</div></div>
      <div class="stat-card"><div class="val">{overview['cols']}</div><div class="label">Columns</div></div>
      <div class="stat-card"><div class="val">{len(overview['num_cols'])}</div><div class="label">Numerical</div></div>
      <div class="stat-card"><div class="val">{len(overview['cat_cols'])}</div><div class="label">Categorical</div></div>
      <div class="stat-card"><div class="val">{overview['missing_pct']}%</div><div class="label">Missing Values</div></div>
      <div class="stat-card"><div class="val">{overview['duplicate_rows']}</div><div class="label">Duplicate Rows</div></div>
      <div class="stat-card"><div class="val">{overview['memory_mb']}</div><div class="label">Memory (MB)</div></div>
    </div>
  </section>

  <!-- AI Insights -->
  <section>
    <p class="section-title">02 · AI-Generated Insights</p>
    <div class="insights-box">
      {insights_html}
    </div>
  </section>

  <!-- Column Stats -->
  <section>
    <p class="section-title">03 · Column Statistics</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Stats / Top Values</th>
          </tr>
        </thead>
        <tbody>{stat_rows()}</tbody>
      </table>
    </div>
  </section>

  <!-- Distributions -->
  <section>
    <p class="section-title">04 · Distributions</p>
    <div class="chart-grid">
      {f'<div class="chart-card"><h3>Histograms</h3>{_img_tag(img_dist, "distributions")}</div>' if img_dist else ''}
      {f'<div class="chart-card"><h3>Boxplots</h3>{_img_tag(img_box, "boxplots")}</div>' if img_box else ''}
      {f'<div class="chart-card"><h3>Categorical Counts</h3>{_img_tag(img_cat, "categories")}</div>' if img_cat else ''}
      {f'<div class="chart-card"><h3>Missing Values</h3>{_img_tag(img_missing, "missing")}</div>' if img_missing else '<div class="chart-card"><h3>Missing Values</h3><p style="color:var(--muted);padding:1rem">✓ No missing values detected.</p></div>'}
    </div>
  </section>

  <!-- Correlation -->
  <section>
    <p class="section-title">05 · Correlation Analysis</p>
    <div class="chart-grid">
      {f'<div class="chart-card" style="grid-column:1/-1"><h3>Correlation Heatmap</h3>{_img_tag(img_corr, "correlation heatmap")}</div>' if img_corr else ''}
    </div>
    <br>
    <div class="table-wrap" style="max-width:500px">
      <table>
        <thead><tr><th>Column A</th><th>Column B</th><th>Pearson r</th></tr></thead>
        <tbody>{corr_rows()}</tbody>
      </table>
    </div>
  </section>

</div>

<footer>Generated by <strong>eda_generator.py</strong> · Powered by Claude AI</footer>
</body>
</html>"""

    return html


# ══════════════════════════════════════════════════════════════════════════════
# 5. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(source,
                    output_path: str = "eda_report.html",
                    dataset_name: str | None = None,
                    api_key: str | None = None) -> str:
    """
    Generate an EDA report from a CSV path or a pandas DataFrame.

    Parameters
    ----------
    source       : str path to CSV or pd.DataFrame
    output_path  : where to save the HTML report
    dataset_name : display name (defaults to filename or 'Dataset')
    api_key      : Gemini API key (falls back to GEMINI_API_KEY env var)

    Returns
    -------
    str : absolute path of the saved report
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        df = pd.read_csv(path)
        name = dataset_name or path.stem
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
        name = dataset_name or "Dataset"
    else:
        raise TypeError("source must be a file path or pd.DataFrame")

    print(f"\n📊 Generating EDA report for '{name}' ({len(df):,} rows × {len(df.columns)} cols)...")
    html = build_html_report(df, dataset_name=name, api_key=api_key)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"✅ Report saved → {out.resolve()}")
    return str(out.resolve())


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated EDA Report Generator")
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output", default="eda_report.html",
                        help="Output HTML file path (default: eda_report.html)")
    parser.add_argument("-n", "--name", default=None, help="Dataset display name")
    parser.add_argument("-k", "--api-key", default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    generate_report(args.csv, args.output, args.name, args.api_key)
