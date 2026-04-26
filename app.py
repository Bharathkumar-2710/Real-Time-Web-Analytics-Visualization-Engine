from flask import Flask, render_template, jsonify, request
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SQLite setup (you can change to PostgreSQL/MySQL later)
engine = create_engine("sqlite:///analytics.db", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Example table: marketing_spend vs user_signups
class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    marketing_spend = Column(Float, nullable=False)
    user_signups = Column(Float, nullable=False)

Base.metadata.create_all(engine)


def load_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="D"),
            "marketing_spend": np.random.rand(100) * 10000,
            "user_signups": np.random.rand(100) * 500 + np.random.rand(100) * 100,
        }
    )


def store_data():
    data = load_data()
    session = Session()
    try:
        session.query(Record).delete()
        for _, row in data.iterrows():
            rec = Record(
                timestamp=row["timestamp"],
                marketing_spend=float(row["marketing_spend"]),
                user_signups=float(row["user_signups"]),
            )
            session.add(rec)
        session.commit()
    finally:
        session.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def store_df(df):
    session = Session()
    try:
        session.query(Record).delete()
        for _, row in df.iterrows():
            timestamp = row.get("timestamp")
            if pd.isna(timestamp) or timestamp is None:
                timestamp = datetime.utcnow()
            else:
                timestamp = pd.to_datetime(timestamp, errors="coerce")
                if pd.isna(timestamp):
                    timestamp = datetime.utcnow()

            rec = Record(
                timestamp=timestamp,
                marketing_spend=float(row["marketing_spend"]),
                user_signups=float(row["user_signups"]),
            )
            session.add(rec)
        session.commit()
    finally:
        session.close()


def fetch_data():
    session = Session()
    try:
        rows = session.query(
            Record.timestamp,
            Record.marketing_spend,
            Record.user_signups,
        ).all()
        return pd.DataFrame(rows, columns=["timestamp", "marketing_spend", "user_signups"])
    finally:
        session.close()


def init_db():
    Base.metadata.create_all(engine)
    session = Session()
    try:
        if session.query(Record).count() == 0:
            store_data()
    finally:
        session.close()


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    correlation = "N/A"
    n = 0

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part in request"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No selected file"
            elif not allowed_file(file.filename):
                error = "Allowed file type is CSV only"
            else:
                df = pd.read_csv(file.stream)
                required_cols = ["marketing_spend", "user_signups"]
                if not all(col in df.columns for col in required_cols):
                    error = "CSV must include columns: marketing_spend, user_signups"
                else:
                    df["marketing_spend"] = pd.to_numeric(df["marketing_spend"], errors="coerce")
                    df["user_signups"] = pd.to_numeric(df["user_signups"], errors="coerce")
                    if df[required_cols].isna().any().any():
                        error = "CSV contains invalid numeric values"
                    else:
                        store_df(df)
                        corr = df["marketing_spend"].corr(df["user_signups"])
                        correlation = round(corr, 3) if not pd.isna(corr) else "N/A"
                        n = len(df)
                        return render_template("index.html", correlation=correlation, n=n, error=error)

    df = fetch_data()
    if not df.empty:
        corr = df["marketing_spend"].corr(df["user_signups"])
        correlation = round(corr, 3) if not pd.isna(corr) else "N/A"
    n = len(df)
    return render_template("index.html", correlation=correlation, n=n, error=error)


@app.route("/api/data")
def api_data():
    df = fetch_data()
    return jsonify(df.to_dict(orient="records"))


@app.route("/chart")
def bivariate_chart():
    df = fetch_data()
    if df.empty or len(df) < 2:
        fig = px.scatter(x=[], y=[], title="No chart data available")
    else:
        fig = px.scatter(
            df,
            x="marketing_spend",
            y="user_signups",
            title="Marketing Spend vs User Signups",
        )
    chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template("chart.html", chart_json=chart_json)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
