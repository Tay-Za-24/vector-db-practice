from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, GeoRadius, GeoPoint, RecommendQuery, RecommendInput

# 1. Connect to Qdrant (the database is already loaded and running!)
client = QdrantClient(url="http://localhost:6333")
collection_name = "dummy_products"

# 2. Load the embedding model (needed to translate your search term into numbers)
print("Loading local sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Perform a search
keyword = "running shoes or sportswear"

# priceFilter = Filter(
#     must=[
#         FieldCondition(key="category", match=MatchValue(value="electronics")),
#         FieldCondition(key="price", range=Range(gte=100.00, lte=200.00))
#     ]
# )

# geoCondition = FieldCondition(
#     key="location",
#     geo_radius=GeoRadius(
#         center=GeoPoint(lat=51.5074, lon=-0.1278),
#         radius=50000.0
#     )
# )

recommendation = RecommendQuery(
    recommend=RecommendInput(
        positive=[1],
        negative=[2]
    )
)


search_result = client.query_points(
    collection_name=collection_name,
    query=recommendation,
    limit=2,
).points

print("\n--- Search Results ---")
for item in search_result:
    print(f"ID: {item.id} | Score: {item.score:.3f} | Title: {item.payload['title']} | Location: {item.payload['location']}")