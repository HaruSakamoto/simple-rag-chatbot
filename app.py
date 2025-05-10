import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)


def generate_answer(query_text: str):
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if not results or results[0][1] < 0.6:
        return "No relevant results found.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = f"""
    Answer the question based only on the following context:

    {context_text}

    ---

    Answer the question based on the above context: {query_text}
    """

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content(prompt)
    sources = [doc.metadata.get("source", "N/A") for doc, _ in results]
    return response.text, sources


# Streamlit UI
st.title("RAG Chatbot")
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question...")
if query:
    answer, sources = generate_answer(query)
    st.session_state.history.append((query, answer, sources))

for q, a, s in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        st.caption("Sources: " + ", ".join(s))
