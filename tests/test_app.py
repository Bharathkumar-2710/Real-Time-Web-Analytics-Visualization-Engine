import io
import pytest

from app import create_app


@pytest.fixture()
def client(tmp_path):
    db_file = tmp_path / "test_analytics.db"
    flask_app = create_app(database_url=f"sqlite:///{db_file}")
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


def test_health_and_initial_seed(client):
    res = client.get("/health")
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["status"] == "ok"
    assert payload["records"] == 30


def test_dashboard_and_analytics_api(client):
    home = client.get("/")
    assert home.status_code == 200
    assert b"Real-Time Web Analytics Engine" in home.data

    data_res = client.get("/api/data")
    assert data_res.status_code == 200
    records = data_res.get_json()
    assert isinstance(records, list)
    assert len(records) == 30

    analytics_res = client.get("/api/analytics")
    assert analytics_res.status_code == 200
    body = analytics_res.get_json()
    assert body["analytics"]["record_count"] == 30
    assert body["analytics"]["correlation"] is not None
    assert len(body["analytics"]["trendline"]) == 2


@pytest.mark.parametrize("chart_type", ["scatter", "line", "bar", "histogram", "box", "invalid"])
def test_plotly_chart_modes(client, chart_type):
    res = client.get(f"/chart?type={chart_type}")
    assert res.status_code == 200
    assert b"Plotly Analytics Explorer" in res.data


def test_csv_upload_valid_and_invalid(client):
    valid_csv = b"timestamp,marketing_spend,user_signups\n2026-04-01,1000,60\n2026-04-02,2000,130\n"
    res = client.post(
        "/",
        data={"file": (io.BytesIO(valid_csv), "upload.csv")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    assert b"Successfully imported 2 records" in res.data

    # Missing column
    bad_cols = b"spend,signups\n1000,50\n"
    res_bad_cols = client.post(
        "/",
        data={"file": (io.BytesIO(bad_cols), "bad.csv")},
        content_type="multipart/form-data",
    )
    assert res_bad_cols.status_code == 200
    assert b"missing required column" in res_bad_cols.data

    # Negative values
    neg_csv = b"marketing_spend,user_signups\n-100,50\n"
    res_neg = client.post(
        "/",
        data={"file": (io.BytesIO(neg_csv), "neg.csv")},
        content_type="multipart/form-data",
    )
    assert res_neg.status_code == 200
    assert b"non-negative numbers" in res_neg.data

    # Wrong extension
    res_ext = client.post(
        "/",
        data={"file": (io.BytesIO(b"abc"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert res_ext.status_code == 200
    assert b"Only .csv files are supported" in res_ext.data


def test_simulate_reset_and_exports(client):
    sim = client.post("/api/simulate")
    assert sim.status_code == 201
    sim_body = sim.get_json()
    assert sim_body["analytics"]["record_count"] == 31

    reset = client.post("/api/reset")
    assert reset.status_code == 200
    assert reset.get_json()["analytics"]["record_count"] == 30

    export_res = client.get("/api/export")
    assert export_res.status_code == 200
    assert "text/csv" in export_res.content_type
    assert b"marketing_spend,user_signups" in export_res.data

    sample_res = client.get("/api/sample-csv")
    assert sample_res.status_code == 200
    assert "text/csv" in sample_res.content_type
