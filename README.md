# Real-Time Web Analytics Visualization Engine

A full-stack Flask analytics engine for ingesting, storing, analyzing, and visualizing marketing spend versus user signup conversion telemetry in real time.

## Features

- **Executive Overview Dashboard (`/`)**:
  - Live KPI cards: Total Marketing Spend, Total User Signups, Average Cost per Signup (CAC), Pearson Correlation ($r$) with strength classification, and OLS Coefficient of Determination ($R^2$).
  - Interactive **Chart.js** visualizations supporting **Scatter + OLS Trendline**, **Chronological Dual-Axis Line**, and **Signups Bar Chart** modes.
  - **Real-Time Live Stream**: Simulate incoming conversion events (`POST /api/simulate`) manually or via a 5-second auto-stream toggle.
  - **Latest Telemetry Table**: Displays recent UTC-timestamped observations and per-event acquisition cost.
- **Plotly Deep Dive Explorer (`/chart`)**:
  - Interactive Plotly visualizations (`scatter`, `line`, `bar`, `histogram`, `box`) with server-computed OLS trendlines (`scipy.stats.linregress`) and regression statistics (`slope`, `R²`, `p-value`).
- **CSV Ingestion & Data Portability**:
  - Upload custom `.csv` datasets with automatic schema normalization, numeric/non-negative validation, and error feedback.
  - Download a sample CSV template (`/api/sample-csv`), export the active dataset (`/api/export`), or reset to the baseline dataset (`/api/reset`).

## Tech Stack

- **Backend**: Python, Flask, SQLAlchemy (SQLite by default; configurable via `DATABASE_URL`)
- **Data & Statistics**: Pandas, NumPy, SciPy
- **Visualization**: Chart.js 4, Plotly
- **Testing & Serving**: Pytest, Gunicorn

## Project Structure

```text
├── app.py                  # Flask application factory, SQLAlchemy model, analytics & API routes
├── data.csv                # Baseline sample marketing spend & user signups dataset
├── requirements.txt        # Python dependencies
├── static/
│   └── style.css           # Dark analytics dashboard stylesheet
├── templates/
│   ├── index.html          # Main overview dashboard & live stream UI
│   └── chart.html          # Plotly bivariate & distribution explorer
└── tests/
    └── test_app.py         # Automated pytest suite for routes, CSV validation, and analytics
```

## Getting Started

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

For production serving with Gunicorn:

```bash
gunicorn app:app --bind 0.0.0.0:8000
```

### 3. Run the test suite

```bash
pytest -v
```

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET`, `POST` | `/` | Dashboard UI and CSV dataset upload |
| `GET` | `/chart?type=scatter` | Plotly explorer (`scatter`, `line`, `bar`, `histogram`, `box`) |
| `GET` | `/api/data` | JSON array of stored telemetry records |
| `GET` | `/api/analytics` | Full payload with records, KPIs, and OLS regression metrics |
| `POST` | `/api/simulate` | Appends a simulated real-time observation and returns updated KPIs |
| `POST` | `/api/reset` | Resets database records to the baseline `data.csv` dataset |
| `GET` | `/api/export` | Downloads the active database records as CSV |
| `GET` | `/api/sample-csv` | Downloads the sample baseline CSV template |
| `GET` | `/health` | Service health check and current record count |

## CSV Schema

Uploaded CSV files must include `marketing_spend` and `user_signups` columns (case-insensitive) and may optionally include `timestamp`:

```csv
timestamp,marketing_spend,user_signups
2026-03-01,1450.00,98
2026-03-02,1820.50,124
```
