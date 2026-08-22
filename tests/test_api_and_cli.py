from fastapi.testclient import TestClient

from spam_detector.api import create_app
from spam_detector.cli import main

SPAM = "URGENT: verify your account now at http://secure-login-verify.example or it is closed"


def test_health_reports_model_availability(trained_model):
    model_path, _ = trained_model
    client = TestClient(create_app(model_path))
    assert client.get("/health").json() == {"status": "ok", "model_available": True}


def test_predict_endpoint_returns_scores(trained_model):
    model_path, _ = trained_model
    client = TestClient(create_app(model_path))
    response = client.post("/predict", json={"messages": [SPAM, "lunch at noon?"]})
    assert response.status_code == 200
    body = response.json()
    assert [p["label"] for p in body["predictions"]] == ["spam", "ham"]
    assert 0.0 <= body["predictions"][0]["spam_probability"] <= 1.0


def test_predict_endpoint_rejects_empty_payload(trained_model):
    model_path, _ = trained_model
    client = TestClient(create_app(model_path))
    assert client.post("/predict", json={"messages": []}).status_code == 422


def test_api_reports_missing_model(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.joblib"))
    assert client.post("/predict", json={"messages": ["hi"]}).status_code == 503


def test_cli_predict_outputs_json(trained_model, capsys):
    model_path, _ = trained_model
    exit_code = main(["predict", "--model-path", str(model_path), "--json", SPAM])
    assert exit_code == 0
    assert '"label": "spam"' in capsys.readouterr().out
