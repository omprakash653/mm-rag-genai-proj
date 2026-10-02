from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)

vector = embeddings.embed_query(
    "What is artificial intelligence?"
)

print("Embedding dimension:", len(vector))
print("First 5 values:", vector[:5])