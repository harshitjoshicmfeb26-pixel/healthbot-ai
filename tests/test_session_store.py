"""
tests/test_session_store.py
────────────────────────────
Tests for chatbot.session_store.SessionStore — the in-memory bridge that
lets the stateless Flask API hold a multi-turn ChatSession across
requests. See that module's docstring for scope/limitations.
"""

from chatbot.session_store import SessionStore


def test_create_returns_a_usable_session_id():
    store = SessionStore()
    session_id = store.create()
    assert isinstance(session_id, str) and session_id

    same_id, session = store.get(session_id)
    assert same_id == session_id
    assert session.state == "greeting"


def test_get_with_unknown_id_transparently_starts_a_new_session():
    store = SessionStore()
    new_id, session = store.get("this-id-does-not-exist")
    assert new_id != "this-id-does-not-exist"
    assert session.state == "greeting"


def test_get_with_none_starts_a_new_session():
    store = SessionStore()
    session_id, session = store.get(None)
    assert session_id
    assert session is not None


def test_session_state_persists_across_get_calls():
    store = SessionStore()
    session_id = store.create()
    _, session = store.get(session_id)
    session.reply("I have a cough and fever")

    _, session_again = store.get(session_id)
    assert session_again is session
    assert session_again.symptom_messages, "The symptom message should have been recorded on the stored session."


def test_reset_clears_session_state_but_keeps_the_same_id():
    store = SessionStore()
    session_id = store.create()
    _, session = store.get(session_id)
    session.reply("I have a cough and fever")
    assert session.symptom_messages

    store.reset(session_id)
    _, reset_session = store.get(session_id)
    assert reset_session.symptom_messages == []
    assert reset_session.state == "greeting"


def test_active_count_reflects_created_sessions():
    store = SessionStore()
    store.create()
    store.create()
    assert store.active_count() == 2
