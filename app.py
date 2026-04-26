from flask import Flask, render_template, jsonify
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json

app = Flask(__name__)

# SQLite setup (you can change to PostgreSQL/MySQL later)
engine = create_engine("sqlite:///analytics.db", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Example table: marketing_spend vs user_signups
class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    price = Column(Float)           # e.g., stock price
    marketing_spend = Column(Float) # X variable
    user_signups = Column(Float)    # Y variable

Base.metadata.create_all(engine)


def load_data():
    # In practice, this can be:
    # - CSV file: pd.read_csv("data.csv")
    # - API call to stock API
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="D"),
            "price": np.random.randn(100).cumsum() + 100,
            "marketing_spend": np.random.rand(100) * 10000,
            "user_signups": np.random.rand(100) * 500 + np.random.rand(100) * 100,
        }
    )
    return data


def store_data():
    """Save a cleaned dataset to DB once."""
    data = load_data()
    session = Session()
    try:
        session.query(Record).delete()
        for _, row in data.iterrows():
            rec = Record(
                price=row["price"],
                marketing_spend=row["marketing_spend"],
                user_signups=row["user_signups"],
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
            Record.price,
            Record.marketing_spend,
            Record.user_signups,
        ).all()
        df = pd.DataFrame(rows, columns=["timestamp", "price", "marketing_spend", "user_signups"])
        return df
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


@app.route("/")
def index():
    df = fetch_data()
    if len(df) < 2:
        corr_sp = 0.0
    else:
        corr_sp = df["marketing_spend"].corr(df["user_signups"])
    return render_template(
        "index.html",
        correlation=round(corr_sp or 0.0, 3),
        n=len(df),
    )


@app.route("/api/data")
def api_data():
    df = fetch_data()
    return jsonify(df.to_dict(orient="records"))


@app.route("/chart")
def bivariate_chart():
    df = fetch_data()
    fig = px.scatter(
        df,
        x="marketing_spend",
        y="user_signups",
        trendline="ols",
        title="Marketing Spend vs User Signups"
    )
    chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template("chart.html", chart_json=chart_json)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
