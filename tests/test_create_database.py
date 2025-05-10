from langchain.schema import Document

from create_database import count_tokens, load_documents, split_text


def test_split_text_returns_chunks():
    dummy_docs = [Document(page_content="This is a test document. " * 20)]
    chunks = split_text(dummy_docs, chunk_size=300)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)
    assert all(count_tokens(c.page_content) <= 300 for c in chunks)


def test_load_documents_returns_list():
    # This test assumes test files exist in data/books/
    docs = load_documents()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)


def test_save_to_chroma_creates_dir(tmp_path):
    from create_database import save_to_chroma
    dummy_doc = Document(page_content="Sample content")
    chroma_path = tmp_path / "chroma_test"

    class DummyEmbedding:
        def embed_documents(self, texts): return [[0.0] * 768 for _ in texts]
        def embed_query(self, text): return [0.0] * 768

    save_to_chroma([dummy_doc], persist_path=str(chroma_path), embedding_function=DummyEmbedding())
    assert chroma_path.exists()
