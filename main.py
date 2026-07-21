from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Vector Search API")

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "dummy_products"

print("Initiating Embedding model")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Initiated")

@app.get("/")
def read_root():
    return {message: "Vector DB api is up and ready."}