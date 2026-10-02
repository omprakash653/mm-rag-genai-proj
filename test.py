from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("QDRANT_Cluster_Endpoint")

print("QDRANT URL:")
print(repr(url))