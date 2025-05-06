import pytest
from app import generate_answer

# You should ideally mock the database here, but for simple tests:
def test_generate_answer_no_results(monkeypatch):
    class DummyDB:
        def similarity_search_with_relevance_scores(self, query, k):
            return []  # Simulate no results

    monkeypatch.setattr("app.db", DummyDB())
    answer, sources = generate_answer("What is the purpose of life?")
    assert answer == "No relevant results found."
    assert sources == []

def test_generate_answer_with_results(monkeypatch):
    class DummyDoc:
        page_content = "The mitochondria is the powerhouse of the cell."
        metadata = {"source": "biology101.md"}

    class DummyModel:
        def generate_content(self, prompt):
            return type("obj", (object,), {"text": "Here is a generated answer."})()

    class DummyDB:
        def similarity_search_with_relevance_scores(self, query, k):
            return [(DummyDoc(), 0.9)]

    monkeypatch.setattr("app.db", DummyDB())
    monkeypatch.setattr("app.genai.GenerativeModel", lambda model_name: DummyModel())

    answer, sources = generate_answer("What is the mitochondria?")
    assert "generated answer" in answer.lower()
    assert sources == ["biology101.md"]
