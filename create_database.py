from langchain.document_loaders import DirectoryLoader

DATA_PATH = "data/books"

def load_documents():
    loader  = DirectoryLoader(DATA_PATH, glob = "*.md")
    documents = loader.load()
    return documents