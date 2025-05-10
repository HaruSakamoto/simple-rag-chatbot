import os

import tiktoken
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

api_key = os.getenv("GOOGLE_API_KEY")
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"

if USE_EMBEDDINGS:
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
else:
    embedding = None


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", use_multithreading=True)
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load() + pdf_loader.load()
    return documents


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def split_text(documents: list[Document], chunk_size=300, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[0]
    print(document.page_content)

    return chunks


def save_to_chroma(docs, persist_path="chroma", embedding_function=None):
    if embedding_function is None:
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=persist_path,
    )
    db.persist()


if __name__ == "__main__":
    main()
