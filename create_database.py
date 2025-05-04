from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

from dotenv import load_dotenv

import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
 
def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader  = DirectoryLoader(DATA_PATH, glob = "*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10]
    print(document.page_content)

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)

    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    main()