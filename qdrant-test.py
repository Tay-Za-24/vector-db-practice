from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

search_result = client.query_points(
    collection_name="test_collection",
    query=[0.2, 0.1, 0.9, 0.7],
    query_filter=Filter(
        must=[FieldCondition(key="city", match=MatchValue(value="London"))]
    ),
    with_payload=True,
    limit=3,
).points

print(search_result)