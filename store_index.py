from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extracted_data = load_pdf("Data/")

text_chunks = text_split(extracted_data)

print("Number of text chunks we got: ", len(text_chunks))

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chat-bot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

texts = [doc.page_content.strip() for doc in text_chunks if doc.page_content.strip()]

vector_embeddings = embeddings.embed_documents(texts)

index = pc.Index(index_name)

vectors = [
    {
        "id": f"doc-{i}",
        "values": embedding,
        "metadata": {"text": texts[i]}
    }
    for i, embedding in enumerate(vector_embeddings)
]

def upsert_in_batches(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"✅ Upserted batch {i}–{i + len(batch) - 1}")

upsert_in_batches(index, vectors, batch_size=100)
