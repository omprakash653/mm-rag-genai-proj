from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://c72c6ec2-2cf0-45f4-b5f1-5674064e6ac2.eu-central-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NjQ1Zjk4N2MtODBiMS00MzM5LWE2OTItYzg0ZDdkYTA5ZTg1In0.MAlJkuSLiI4oIgjwXuHFkaGeSSIU9gKmbC8h8RaqGT0"
)

print(client.get_collections())