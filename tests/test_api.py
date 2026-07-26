"""
tests/test_api.py
───────────────────
Smoke tests for the Flask API in api/routes.py, using Flask's built-in
test client — no live server or network socket required. These exercise
the same code path `python3 server.py` runs, just in-process.
"""

import pytest

from server import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_meta_exposes_model_info(client):
    res = client.get("/api/meta")
    assert res.status_code == 200
    body = res.get_json()
    assert body["pathology_count"] > 0
    assert isinstance(body["pathologies"], list)


def test_chat_start_returns_session_and_greeting(client):
    res = client.post("/api/chat/start", json={})
    assert res.status_code == 200
    body = res.get_json()
    assert body["session_id"]
    assert "HealthBot" in body["reply"]


def test_chat_message_round_trip_includes_meta(client):
    start = client.post("/api/chat/start", json={}).get_json()
    session_id = start["session_id"]

    res = client.post(
        "/api/chat/message",
        json={
            "session_id": session_id,
            "message": "I am a 49 year old female with cough, nausea, heartburn and stomach pain",
        },
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["session_id"] == session_id
    assert body["reply"]
    assert "meta" not in body
    assert "severe" in body["reply"].lower()

    res = client.post("/api/chat/message", json={"session_id": session_id, "message": "moderate"})
    assert res.status_code == 200
    body = res.get_json()
    assert "Only say `chest`" in body["reply"]

    followups = ["stomach", "3 days", "none", "none"]
    for answer in followups:
        res = client.post("/api/chat/message", json={"session_id": session_id, "message": answer})
        assert res.status_code == 200
        body = res.get_json()

    assert "meta" in body
    assert "predictions" in body["meta"]
    assert "red_flag" in body["meta"]
    assert "Details used by model" in body["reply"]
    assert "duration: 3 days" in body["reply"]
    assert "previous disease: none" in body["reply"]


def test_chat_collects_required_details_before_prediction(client):
    start = client.post("/api/chat/start", json={}).get_json()
    session_id = start["session_id"]

    first = client.post(
        "/api/chat/message",
        json={"session_id": session_id, "message": "I have chest pain and shortness of breath"},
    ).get_json()
    assert "age" in first["reply"].lower()
    assert "meta" not in first

    answers = ["45", "male", "8/10", "chest", "2 days", "diabetes", "family heart disease"]
    body = first
    for answer in answers:
        res = client.post("/api/chat/message", json={"session_id": session_id, "message": answer})
        assert res.status_code == 200
        body = res.get_json()

    assert "meta" in body
    assert body["meta"]["predictions"]
    assert body["meta"]["red_flag"]["has_red_flag"] is True
    assert "Details used by model" in body["reply"]


def test_chat_message_without_text_is_rejected(client):
    res = client.post("/api/chat/message", json={"message": ""})
    assert res.status_code == 400


def test_chat_reset_requires_session_id(client):
    res = client.post("/api/chat/reset", json={})
    assert res.status_code == 400


def test_case_predict_with_recognizable_symptoms(client):
    res = client.post(
        "/api/case/predict",
        json={
            "age": 55,
            "sex": "M",
            "evidences": "shortness of breath, wheezing, cough",
            "initial_evidence": "shortness of breath",
            "top_n": 3,
        },
    )
    assert res.status_code == 200
    body = res.get_json()
    assert len(body["predictions"]) == 3
    assert "explanation" in body


def test_case_predict_rejects_unrecognizable_symptoms(client):
    res = client.post("/api/case/predict", json={"age": 30, "sex": "F", "evidences": "xyzxyz"})
    assert res.status_code == 400


def test_analyze_exposes_scope_warning_for_out_of_scope_symptoms(client):
    res = client.post(
        "/api/analyze",
        json={"text": "burning urination with lower back pain for 2 days female age 30", "top_n": 3},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["evidence_mode"] == "model_scope_limited"
    assert body["scope_warning"]["reason"] == "urinary_evidence_outside_model_scope"
    assert body["predictions"][0]["flag"] == "INFO"
