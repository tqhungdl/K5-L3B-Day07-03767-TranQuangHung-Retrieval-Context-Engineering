import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agentic import (
    DenseRetriever,
    OpenAIController,
    HybridRetriever,
    LexicalRetriever,
    run_agent,
)

DAY7 = Path(__file__).resolve().parents[1]
DAY6 = (
    DAY7.parent
    / "K5-L3B-Day06-2A202603767-DataIngestionEngineering"
)

sys.path.insert(0, str(DAY6 / "starter"))

from src.embeddings import MyEmbedder
from src.store import PgVectorStore

# Explicit environment values take precedence, followed by Day07 config.
load_dotenv(DAY7 / ".env")
load_dotenv(DAY7 / "starter" / ".env")
load_dotenv(DAY6 / "starter" / ".env")
if not os.environ.get("OPENAI_API_KEY", "").strip():
    raise ValueError("Set OPENAI_API_KEY in PowerShell or Day07 .env")
database_url = os.environ["DATABASE_URL"]
strategy = "B"

config = PgVectorStore(database_url, 1).get_config(strategy)
model = config["model"]

embedder = MyEmbedder(
    model_name=model["name"],
    revision=model["revision"],
    cache_folder=DAY6 / ".cache" / "models",
)

store = PgVectorStore(database_url, embedder.dimensions)

chunks = json.loads(
    (DAY7 / "starter/data/converted/chunks.json").read_text(
        encoding="utf-8"
    )
)

dense = DenseRetriever(embedder, store, strategy=strategy)
lexical = LexicalRetriever(chunks)
hybrid = HybridRetriever(dense, lexical)

# First inspect retrieval without making an OpenAI call.
question = "What four functions make up the AI RMF Core?"

for name, retriever in [
    ("tfidf", lexical),
    ("dense", dense),
    ("hybrid", hybrid),
]:
    print(f"\n{name}:")
    for chunk in retriever.search(question, filters=None, top_k=3):
        print(chunk["id"], chunk["page"], chunk["score"])

controller = OpenAIController()
try:
    result = run_agent(
        question,
        controller,
        hybrid,
        top_k=3,
        budget=800,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
finally:
    controller.close()
