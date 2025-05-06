import os
import pytest
from unittest.mock import patch
from create_database import split_text, load_documents, save_to_chroma
from langchain.schema import Document

def test_split_text_returns_chunks():
    dummy_docs = [Document(page_content="This is a test document. " * 20)]
    chunks = split_text(dummy_docs)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)
    assert len(chunks[0].page_content) <= 300

def test_load_documents_returns_list():
    # This test assumes test files exist in data/books/
    docs = load_documents()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)

def test_save_to_chroma_creates_dir(tmp_path):
    dummy_doc = Document(page_content="Sample content")
    save_to_chroma([dummy_doc])
    assert os.path.exists("chroma")