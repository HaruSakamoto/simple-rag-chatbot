import argparse
import json
import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Use Gemini embeddings for vector store
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Search similar documents
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.6:
        print("Unable to find matching results.")
        return

    # Build prompt from context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt Sent to Gemini:\n", prompt)

    # Generate response using Gemini
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content(prompt)
    response_text = response.text

    # Output response and sources
    sources = [doc.metadata.get("source", "unknown") for doc, _ in results]
    output = {
        "answer": response_text,
        "sources": sources
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
