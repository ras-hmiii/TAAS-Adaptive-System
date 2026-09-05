from fastapi.testclient import TestClient

from src.api.main import app


def test_health_reports_model_state():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_rejects_empty_features():
    response = TestClient(app).post("/predictions", json={"features": {}})
    assert response.status_code == 422


def test_artifact_endpoints_return_empty_collections_without_results():
    client = TestClient(app)
    assert isinstance(client.get("/experiments").json()["experiments"], list)
    assert isinstance(client.get("/drift").json()["events"], list)
    assert isinstance(client.get("/adaptation").json()["events"], list)


def test_explanation_requires_a_trained_model():
    response = TestClient(app).post("/explanations", json={"features": {"age": 30}})
    assert response.status_code in (422, 503)
