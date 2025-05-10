# Simple-Rag-Chatbot
A lightweight Retrieval-Augmented Generation (RAG) chatbot using LangChain, Gemini APIs, and Chroma vector store. This project demonstrates core RAG components (document ingestion → embedding → vector search → LLM generation) with full dev pipeline support including Docker Compose, Streamlit UI, CI/CD, and testing.


### Features
- Gemini Embeddings + Gemini Text Generation API

- Chroma vector store (in-memory)

- LangChain-powered chunking and retrieval

- FastAPI or Streamlit-based interface

- Docker Compose setup (api, vector-db)

- GitHub Actions CI pipeline (lint + test)

- Pytest test coverage ≥ 50%

- CLI and Web UI for Q&A


## Tech Stack

### Component	|        Tech Used

Language	|        Python 3.10

Embeddings	|        Gemini Embeddings API

LLM	|        Gemini Text Generation API

Vector Store	|        Chroma (in-memory)

RAG Framework	|        LangChain

UI	|        CLI / Streamlit

Containerization	|        Docker Compose

CI/CD	        |        GitHub Actions

Testing	        |        pytest + coverage


## Project Structure
simple-rag-chatbot/

├── .github/workflows/         # GitHub Actions CI config

├── chroma/                    # Vector store data

├── data/books/                # Docs to embed (≤10 PDF/Markdown files)

├── tests/                     # Pytest unit tests

├── venv/                      # Python virtual environment

├── .env                       # API key and env variables

├── .gitignore

├── .coverage

├── app.py                     # Streamlit UI

├── create_database.py         # Embed docs into Chroma

├── docker-compose.yml         # Compose config (API + Chroma)

├── Dockerfile                 # Container definition

├── query_data.py              # Search & generation logic

├── requirements.txt

└── README.md
        

## Setup Instructions
### Local (Python)

### Create environment
```python3 -m venv .venv```

```source venv\Scripts\activate```

### Install dependencies
```pip install -r requirements.txt```

### Set your Gemini API key
```export GOOGLE_API_KEY="your-gemini-api-key"```

### Embed docs into vector store
```python create_database.py```

### Run CLI chatbot
```python query_data.py```

### Or launch Streamlit UI
```streamlit run app.py```

### Run Docker Compose
```docker compose up --build```


## CI Pipeline (GitHub Actions)
Configured to trigger on push and pull_request.


### Pipeline Steps:
- Checkout repository

- Set up Python 3.10

- Run flake8 and isort for lint checks

- Run pytest with coverage

### Pass Criteria:

- No lint errors

- All tests pass

- Coverage ≥ 50%

### Run unit tests with:

```pytest --cov=app tests/```

### Implementation Time: 

Estimated total: ~16 hours

