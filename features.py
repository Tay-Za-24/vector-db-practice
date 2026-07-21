import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    GeoRadius,
    GeoPoint,
    PayloadSchemaType,
    RecommendQuery,
    RecommendInput
)

# 1. Define rich mock dataset of products
products = [
    {
        "id": 1,
        "title": "Wireless Noise-Canceling Headphones",
        "description": "High-quality wireless headphones with active noise cancellation and 30-hour battery life.",
        "category": "electronics",
        "price": 199.99,
        "rating": 4.8,
        "tags": ["audio", "wireless", "premium"],
        "location": {"lat": 51.5074, "lon": -0.1278} # London
    },
    {
        "id": 2,
        "title": "Budget Bluetooth Earbuds",
        "description": "Affordable wireless earbuds with decent bass and water resistance, perfect for workouts.",
        "category": "electronics",
        "price": 39.99,
        "rating": 4.1,
        "tags": ["audio", "wireless", "budget"],
        "location": {"lat": 51.5074, "lon": -0.1278} # London
    },
    {
        "id": 3,
        "title": "Ergonomic Mechanical Keyboard",
        "description": "Tactile mechanical keyboard with customizable RGB backlighting and wrist rest.",
        "category": "electronics",
        "price": 120.00,
        "rating": 4.6,
        "tags": ["office", "accessories", "rgb"],
        "location": {"lat": 48.8566, "lon": 2.3522} # Paris
    },
    {
        "id": 4,
        "title": "Minimalist Leather Journal",
        "description": "Refillable leather-bound notebook with premium cream paper, ideal for sketching and writing.",
        "category": "stationery",
        "price": 25.00,
        "rating": 4.9,
        "tags": ["leather", "gift", "writing"],
        "location": {"lat": 40.7128, "lon": -74.0060} # New York
    },
    {
        "id": 5,
        "title": "Stainless Steel Water Bottle",
        "description": "Double-walled vacuum insulated bottle that keeps drinks cold for 24 hours or hot for 12 hours.",
        "category": "outdoors",
        "price": 18.50,
        "rating": 4.5,
        "tags": ["eco-friendly", "hydration", "durable"],
        "location": {"lat": 34.0522, "lon": -118.2437} # Los Angeles
    },
    {
        "id": 6,
        "title": "Ultra-lightweight Running Shoes",
        "description": "Breathable mesh running shoes with responsive cushioning for maximum energy return.",
        "category": "apparel",
        "price": 89.95,
        "rating": 4.3,
        "tags": ["sports", "footwear", "running"],
        "location": {"lat": 51.5074, "lon": -0.1278} # London
    }
]

print("Initializing Qdrant Client...")
client = QdrantClient(url="http://localhost:6333")

collection_name = "dummy_products"

# 2. Load Local Embedding Model
print("Loading local sentence-transformers model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
vector_size = model.get_embedding_dimension()
print(f"Model loaded. Embeddings dimension: {vector_size}")

# 3. Create/Recreate Qdrant Collection
print(f"Creating/recreating collection '{collection_name}'...")
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

# 4. Generate Embeddings and Ingest Data
print("Generating embeddings for product descriptions...")
descriptions = [p["description"] for p in products]
embeddings = model.encode(descriptions).tolist()

points = []
for idx, product in enumerate(products):
    points.append({
        "id": product["id"],
        "vector": embeddings[idx],
        "payload": {
            "title": product["title"],
            "description": product["description"],
            "category": product["category"],
            "price": product["price"],
            "rating": product["rating"],
            "tags": product["tags"],
            "location": product["location"]
        }
    })

print("Uploading points to Qdrant...")
client.upsert(
    collection_name=collection_name,
    points=points
)
print("Data uploaded successfully!\n")

# Wait a brief moment to ensure index is refreshed
time.sleep(1)

# 5. Create payload indexes for faster queries (highly recommended for production)
print("Creating payload indexes...")
client.create_payload_index(collection_name, "category", PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name, "price", PayloadSchemaType.FLOAT)
client.create_payload_index(collection_name, "location", PayloadSchemaType.GEO)
