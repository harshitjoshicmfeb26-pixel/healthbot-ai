import chatbot.bot as bot


def _case_details():
    return {
        "age": 45,
        "gender": "male",
        "symptoms_text": "chest pain with shortness of breath",
        "duration": "2 days",
        "severity": "8/10",
        "pain_location": "chest",
        "previous_disease_or_history": "diabetes",
        "genetic_or_family_history": "family heart disease",
    }


def _structured_prediction(**kwargs):
    return [{
        "rank": 1,
        "disease": "Asthma",
        "pathology": "Asthma",
        "label_id": 1,
        "confidence": 0.8,
        "confidence_pct": "80.0%",
        "flag": "HIGH",
        "source": "structured clinical evidence",
    }]


def test_complete_case_always_uses_structured_engine(monkeypatch):
    calls = {"structured": 0, "simplified": 0}

    def structured(**kwargs):
        calls["structured"] += 1
        return _structured_prediction(**kwargs)

    def simplified(*args, **kwargs):
        calls["simplified"] += 1
        raise AssertionError("simplified classifier must not be called")

    monkeypatch.setattr(bot, "predict_case", structured)
    monkeypatch.setattr(bot, "semantic_search_case", lambda **kwargs: [])
    monkeypatch.setattr(bot, "predict_disease", simplified, raising=False)
    monkeypatch.setattr(bot, "disease_classifier_available", lambda: True, raising=False)

    result = bot.assess_symptoms(
        "I have chest pain and shortness of breath",
        case_details=_case_details(),
    )

    assert calls == {"structured": 1, "simplified": 0}
    assert result["evidence_bridge"]["mode"] == "human_symptoms_to_structured_evidence"
    assert result["predictions"][0]["source"] == "structured clinical evidence"


def test_conversation_slots_are_retained_after_routing_change(monkeypatch):
    monkeypatch.setattr(bot, "predict_case", _structured_prediction)
    monkeypatch.setattr(bot, "semantic_search_case", lambda **kwargs: [])
    result = bot.assess_symptoms(
        "I have chest pain and shortness of breath",
        case_details=_case_details(),
    )

    details = result["evidence_bridge"]["case_details"]
    assert details["duration"] == "2 days"
    assert details["severity"] == "8/10"
    assert details["previous_disease_or_history"] == "diabetes"
    assert details["genetic_or_family_history"] == "family heart disease"
    assert details["pain_location"] == "chest"


def test_missing_simplified_artifact_does_not_change_prediction(monkeypatch):
    calls = {"structured": 0}

    def structured(**kwargs):
        calls["structured"] += 1
        return _structured_prediction(**kwargs)

    monkeypatch.setattr(bot, "predict_case", structured)
    monkeypatch.setattr(bot, "semantic_search_case", lambda **kwargs: [])
    monkeypatch.setattr(bot, "disease_classifier_available", lambda: False, raising=False)
    monkeypatch.setattr(
        bot,
        "predict_disease",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        raising=False,
    )

    result = bot.assess_symptoms(
        "I have chest pain and shortness of breath",
        case_details=_case_details(),
    )

    assert calls["structured"] == 1
    assert result["predictions"][0]["source"] == "structured clinical evidence"


def test_red_flag_handling_is_independent_of_prediction(monkeypatch):
    monkeypatch.setattr(bot, "predict_case", _structured_prediction)
    monkeypatch.setattr(bot, "semantic_search_case", lambda **kwargs: [])
    result = bot.assess_symptoms("chest pain and shortness of breath")

    assert result["red_flag_result"]["has_red_flag"] is True
    assert result["red_flag_result"]["urgency_level"] in {"urgent", "emergency"}


def test_structured_explanation_remains_available(monkeypatch):
    monkeypatch.setattr(bot, "predict_case", _structured_prediction)
    monkeypatch.setattr(bot, "semantic_search_case", lambda **kwargs: [])
    monkeypatch.setattr(
        bot,
        "explain_case",
        lambda **kwargs: {"pathology": "Asthma", "contributions": [{"feature": "ev_E_66"}], "note": ""},
    )

    assessment = bot.assess_symptoms("cough and shortness of breath")
    explanation = bot.assessment_explanation(assessment)

    assert explanation["pathology"] == "Asthma"
    assert explanation["contributions"]
