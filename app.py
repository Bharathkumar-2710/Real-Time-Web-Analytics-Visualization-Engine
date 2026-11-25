from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, Response, jsonify, render_template, request
from plotly.utils import PlotlyJSONEncoder
from scipy import stats
from sqlalchemy import Column, DateTime, Float, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "analytics.db")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data.csv")
ALLOWED_EXTENSIONS = {"csv"}
VALID_PLOTLY_TYPES = {"scatter", "line", "bar", "histogram", "box"}

Base = declarative_base()


class Record(Base):
    """Database model storing marketing spend and user signup metrics."""

    __tablename__ = "records"

    id = Column(Integer, primary_key=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    marketing_spend = Column(Float, nullable=False)
    user_signups = Column(Float, nullable=False)


def _build_engine_and_session(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, echo=False, connect_args=connect_args)
    session_factory = sessionmaker(bind=engine)
    return engine, session_factory


DEFAULT_DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")
engine, Session = _build_engine_and_session(DEFAULT_DATABASE_URL)


def allowed_file(filename: str) -> bool:
    """Return True if the uploaded filename has an allowed CSV extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_sample_data() -> pd.DataFrame:
    """Load baseline analytics data from data.csv or generate a synthetic fallback."""
    if os.path.exists(SAMPLE_CSV_PATH) and os.path.getsize(SAMPLE_CSV_PATH) > 0:
        try:
            df = pd.read_csv(SAMPLE_CSV_PATH)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if {"marketing_spend", "user_signups"}.issubset(df.columns):
                df["marketing_spend"] = pd.to_numeric(df["marketing_spend"], errors="coerce")
                df["user_signups"] = pd.to_numeric(df["user_signups"], errors="coerce")
                df = df.dropna(subset=["marketing_spend", "user_signups"]).copy()
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                else:
                    df["timestamp"] = pd.date_range(end=datetime.now(timezone.utc).date(), periods=len(df), freq="D")
                df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now("UTC").tz_localize(None))
                if not df.empty:
                    return df[["timestamp", "marketing_spend", "user_signups"]]
        except Exception:
            pass

    rng = np.random.default_rng(42)
    n_points = 30
    spend = np.linspace(1500, 9500, n_points) + rng.normal(0, 350, n_points)
    spend = np.clip(spend, 500, None)
    signups = np.round(spend * 0.065 + rng.normal(0, 12, n_points)).clip(min=10)
    timestamps = pd.date_range(end=datetime.now(timezone.utc).date(), periods=n_points, freq="D")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "marketing_spend": np.round(spend, 2),
            "user_signups": signups,
        }
    )


def _normalize_timestamp(value: Any, fallback: datetime) -> datetime:
    if value is None or pd.isna(value):
        return fallback
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return fallback
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(None)
    return parsed.to_pydatetime()


def store_df(df: pd.DataFrame, session_factory=None) -> int:
    """Replace all records in the database with rows from the provided DataFrame."""
    factory = session_factory or Session
    session = factory()
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        session.query(Record).delete()
        records = []
        for idx, row in df.iterrows():
            fallback_ts = now_utc - timedelta(days=max(len(df) - 1 - int(idx) if isinstance(idx, (int, np.integer)) else 0, 0))
            ts = _normalize_timestamp(row.get("timestamp"), fallback_ts)
            records.append(
                Record(
                    timestamp=ts,
                    marketing_spend=float(row["marketing_spend"]),
                    user_signups=float(row["user_signups"]),
                )
            )
        session.add_all(records)
        session.commit()
        return len(records)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def fetch_data(session_factory=None) -> pd.DataFrame:
    """Fetch all analytics records sorted chronologically."""
    factory = session_factory or Session
    session = factory()
    try:
        rows = (
            session.query(
                Record.id,
                Record.timestamp,
                Record.marketing_spend,
                Record.user_signups,
            )
            .order_by(Record.timestamp.asc(), Record.id.asc())
            .all()
        )
        return pd.DataFrame(
            rows, columns=["id", "timestamp", "marketing_spend", "user_signups"]
        )
    finally:
        session.close()


def init_db(db_engine=None, session_factory=None) -> None:
    """Create database tables and seed baseline sample data if empty."""
    target_engine = db_engine or engine
    factory = session_factory or Session
    Base.metadata.create_all(target_engine)
    session = factory()
    try:
        if session.query(Record).count() == 0:
            store_df(load_sample_data(), session_factory=factory)
    finally:
        session.close()


def parse_csv_upload(file_storage) -> tuple[pd.DataFrame | None, str | None]:
    """Validate and parse an uploaded CSV file into a clean DataFrame."""
    if not file_storage or file_storage.filename == "":
        return None, "Please select a CSV file to upload."
    if not allowed_file(file_storage.filename):
        return None, "Invalid file format. Only .csv files are supported."

    try:
        raw_bytes = file_storage.read()
        if not raw_bytes.strip():
            return None, "The uploaded CSV file is empty."
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except UnicodeDecodeError:
        return None, "Could not decode file. Please upload a UTF-8 encoded CSV."
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return None, f"Malformed CSV file: {exc}"

    df.columns = [str(col).strip().lower() for col in df.columns]
    required_cols = ["marketing_spend", "user_signups"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return None, f"CSV is missing required column(s): {', '.join(missing)}."

    if df.empty:
        return None, "CSV contains headers but no data rows."

    df["marketing_spend"] = pd.to_numeric(df["marketing_spend"], errors="coerce")
    df["user_signups"] = pd.to_numeric(df["user_signups"], errors="coerce")

    if not np.isfinite(df["marketing_spend"]).all() or not np.isfinite(df["user_signups"]).all():
        return None, "CSV contains missing or non-numeric values in required columns."

    if (df["marketing_spend"] < 0).any() or (df["user_signups"] < 0).any():
        return None, "Marketing spend and user signups must be non-negative numbers."

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(
            end=datetime.now(timezone.utc).replace(tzinfo=None),
            periods=len(df),
            freq="D",
        )
    else:
        parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed_ts.isna().all():
            df["timestamp"] = pd.date_range(
                end=datetime.now(timezone.utc).replace(tzinfo=None),
                periods=len(df),
                freq="D",
            )
        else:
            df["timestamp"] = parsed_ts.ffill().bfill()

    df = df.reset_index(drop=True)
    return df[["timestamp", "marketing_spend", "user_signups"]], None


def compute_analytics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate summary KPIs, Pearson correlation, and OLS regression metrics."""
    if df.empty:
        return {
            "record_count": 0,
            "total_spend": 0.0,
            "total_signups": 0.0,
            "avg_spend": 0.0,
            "avg_signups": 0.0,
            "cost_per_signup": None,
            "correlation": None,
            "correlation_display": "N/A",
            "correlation_strength": "Insufficient Data",
            "r_squared": None,
            "slope": None,
            "intercept": None,
            "p_value": None,
            "trendline": [],
        }

    x = df["marketing_spend"].to_numpy(dtype=float)
    y = df["user_signups"].to_numpy(dtype=float)
    total_spend = float(np.sum(x))
    total_signups = float(np.sum(y))
    avg_spend = float(np.mean(x))
    avg_signups = float(np.mean(y))
    cac = round(total_spend / total_signups, 2) if total_signups > 0 else None

    correlation = None
    r_squared = None
    slope = None
    intercept = None
    p_value = None
    trendline: list[dict[str, float]] = []

    if len(df) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        reg = stats.linregress(x, y)
        correlation = round(float(reg.rvalue), 4)
        r_squared = round(float(reg.rvalue ** 2), 4)
        slope = round(float(reg.slope), 6)
        intercept = round(float(reg.intercept), 4)
        p_value = round(float(reg.pvalue), 6)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        trendline = [
            {"x": round(x_min, 2), "y": round(float(reg.slope * x_min + reg.intercept), 2)},
            {"x": round(x_max, 2), "y": round(float(reg.slope * x_max + reg.intercept), 2)},
        ]

    if correlation is None:
        strength = "Insufficient Variance"
    elif correlation >= 0.7:
        strength = "Strong Positive"
    elif correlation >= 0.4:
        strength = "Moderate Positive"
    elif correlation > -0.4:
        strength = "Weak / Neutral"
    elif correlation > -0.7:
        strength = "Moderate Negative"
    else:
        strength = "Strong Negative"

    return {
        "record_count": int(len(df)),
        "total_spend": round(total_spend, 2),
        "total_signups": round(total_signups, 2),
        "avg_spend": round(avg_spend, 2),
        "avg_signups": round(avg_signups, 2),
        "cost_per_signup": cac,
        "correlation": correlation,
        "correlation_display": f"{correlation:.3f}" if correlation is not None else "N/A",
        "correlation_strength": strength,
        "r_squared": r_squared,
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "trendline": trendline,
    }


def serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize DataFrame records into JSON-friendly dictionaries."""
    if df.empty:
        return []
    records = []
    for row in df.itertuples(index=False):
        ts = row.timestamp
        if isinstance(ts, pd.Timestamp):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)
        records.append(
            {
                "id": int(row.id) if row.id is not None else None,
                "timestamp": ts_str,
                "marketing_spend": round(float(row.marketing_spend), 2),
                "user_signups": round(float(row.user_signups), 2),
            }
        )
    return records


def build_plotly_figure(df: pd.DataFrame, chart_type: str, analytics: dict[str, Any]) -> go.Figure:
    """Create a themed Plotly figure without requiring statsmodels."""
    chart_type = chart_type if chart_type in VALID_PLOTLY_TYPES else "scatter"

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Analytics Data Available")
        return fig

    if chart_type == "scatter":
        fig = px.scatter(
            df,
            x="marketing_spend",
            y="user_signups",
            hover_data=["timestamp"],
            title="Marketing Spend vs. User Signups (OLS Regression)",
            labels={"marketing_spend": "Marketing Spend ($)", "user_signups": "User Signups"},
        )
        fig.update_traces(
            marker=dict(size=10, color="#38bdf8", line=dict(width=1, color="#0284c7"), opacity=0.85),
            name="Observed Campaigns",
        )
        if analytics.get("trendline"):
            tl = analytics["trendline"]
            fig.add_trace(
                go.Scatter(
                    x=[tl[0]["x"], tl[1]["x"]],
                    y=[tl[0]["y"], tl[1]["y"]],
                    mode="lines",
                    name=f"OLS Trendline (R² = {analytics['r_squared']})",
                    line=dict(color="#f59e0b", width=3, dash="solid"),
                )
            )
    elif chart_type == "line":
        ordered = df.sort_values("timestamp")
        fig = px.line(
            ordered,
            x="timestamp",
            y=["marketing_spend", "user_signups"],
            title="Chronological Spend & Signups Trend",
            labels={"value": "Metric Value", "timestamp": "Timestamp", "variable": "Metric"},
        )
    elif chart_type == "bar":
        ordered = df.sort_values("timestamp")
        fig = px.bar(
            ordered,
            x="timestamp",
            y="user_signups",
            color="marketing_spend",
            color_continuous_scale="Tealgrn",
            title="User Signups Over Time (Colored by Marketing Spend)",
            labels={"timestamp": "Timestamp", "user_signups": "User Signups", "marketing_spend": "Spend ($)"},
        )
    elif chart_type == "histogram":
        fig = px.histogram(
            df,
            x="user_signups",
            nbins=15,
            title="Distribution of User Signups",
            labels={"user_signups": "User Signups"},
            color_discrete_sequence=["#38bdf8"],
        )
    else:  # box
        fig = px.box(
            df,
            y="user_signups",
            points="all",
            title="Statistical Spread of User Signups",
            labels={"user_signups": "User Signups"},
            color_discrete_sequence=["#38bdf8"],
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#e2e8f0"),
        margin=dict(l=55, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.15)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.15)")
    return fig


def create_app(database_url: str | None = None) -> Flask:
    """Application factory supporting custom database URLs for testing and deployment."""
    flask_app = Flask(__name__)
    flask_app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB CSV upload limit

    if database_url:
        app_engine, app_session = _build_engine_and_session(database_url)
    else:
        app_engine, app_session = engine, Session

    init_db(db_engine=app_engine, session_factory=app_session)

    @flask_app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        success = None

        if request.method == "POST":
            df_uploaded, parse_err = parse_csv_upload(request.files.get("file"))
            if parse_err:
                error = parse_err
            elif df_uploaded is not None:
                inserted = store_df(df_uploaded, session_factory=app_session)
                success = f"Successfully imported {inserted} records into the analytics engine."

        df = fetch_data(session_factory=app_session)
        analytics = compute_analytics(df)
        recent_records = serialize_records(df.tail(10).iloc[::-1])
        return render_template(
            "index.html",
            analytics=analytics,
            correlation=analytics["correlation_display"],
            n=analytics["record_count"],
            recent_records=recent_records,
            error=error,
            success=success,
        )

    @flask_app.route("/api/data")
    def api_data():
        df = fetch_data(session_factory=app_session)
        return jsonify(serialize_records(df))

    @flask_app.route("/api/analytics")
    def api_analytics():
        df = fetch_data(session_factory=app_session)
        analytics = compute_analytics(df)
        return jsonify(
            {
                "records": serialize_records(df),
                "analytics": analytics,
            }
        )

    @flask_app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        df = fetch_data(session_factory=app_session)
        analytics = compute_analytics(df)

        if not df.empty:
            min_spend = max(500.0, float(df["marketing_spend"].min()))
            max_spend = max(min_spend + 500.0, float(df["marketing_spend"].max()))
            new_spend = round(float(np.random.uniform(min_spend, max_spend)), 2)
            if analytics["slope"] is not None and analytics["intercept"] is not None:
                baseline_signups = analytics["slope"] * new_spend + analytics["intercept"]
                noise_scale = max(5.0, abs(baseline_signups) * 0.06)
                new_signups = max(1.0, round(float(baseline_signups + np.random.normal(0, noise_scale))))
            else:
                new_signups = max(1.0, round(new_spend * 0.065 + float(np.random.normal(0, 12))))
        else:
            new_spend = round(float(np.random.uniform(1500, 8500)), 2)
            new_signups = round(new_spend * 0.065)

        now_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        session = app_session()
        try:
            rec = Record(
                timestamp=now_ts,
                marketing_spend=new_spend,
                user_signups=float(new_signups),
            )
            session.add(rec)
            session.commit()
            session.refresh(rec)
            created = {
                "id": rec.id,
                "timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "marketing_spend": round(rec.marketing_spend, 2),
                "user_signups": round(rec.user_signups, 2),
            }
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        updated_df = fetch_data(session_factory=app_session)
        return (
            jsonify(
                {
                    "record": created,
                    "analytics": compute_analytics(updated_df),
                }
            ),
            201,
        )

    @flask_app.route("/api/reset", methods=["POST"])
    def api_reset():
        sample_df = load_sample_data()
        count = store_df(sample_df, session_factory=app_session)
        updated_df = fetch_data(session_factory=app_session)
        return jsonify(
            {
                "message": f"Reset dataset to {count} baseline sample records.",
                "analytics": compute_analytics(updated_df),
            }
        )

    @flask_app.route("/api/export")
    def api_export():
        df = fetch_data(session_factory=app_session)
        export_df = df[["timestamp", "marketing_spend", "user_signups"]] if not df.empty else df
        csv_content = export_df.to_csv(index=False)
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=analytics_export.csv"},
        )

    @flask_app.route("/api/sample-csv")
    def api_sample_csv():
        sample_df = load_sample_data()
        csv_content = sample_df.to_csv(index=False)
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=sample_analytics_data.csv"},
        )

    @flask_app.route("/chart")
    def bivariate_chart():
        chart_type = request.args.get("type", "scatter").lower()
        if chart_type not in VALID_PLOTLY_TYPES:
            chart_type = "scatter"

        df = fetch_data(session_factory=app_session)
        analytics = compute_analytics(df)
        fig = build_plotly_figure(df, chart_type, analytics)
        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        return render_template(
            "chart.html",
            chart_json=chart_json,
            chart_type=chart_type,
            analytics=analytics,
        )

    @flask_app.route("/health")
    def health():
        df = fetch_data(session_factory=app_session)
        return jsonify({"status": "ok", "records": int(len(df))})

    return flask_app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="127.0.0.1", port=port, debug=debug)
