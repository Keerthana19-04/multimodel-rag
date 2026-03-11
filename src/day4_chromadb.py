import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent client — saves to disk!
client = chromadb.PersistentClient(path="./chroma_db")

# get_or_create = won't crash if already exists
collection = client.get_or_create_collection(
    name="rag_chunks",
    metadata={"hnsw:space": "cosine"}
)

print("✅ Persistent ChromaDB created!")
print(f"Collection name: {collection.name}")
print(f"Total chunks: {collection.count()}")

# 20 realistic RAG project chunks
chunks = [
    "Python is a high-level programming language",
    "FastAPI is a modern web framework for Python",
    "LangChain helps build LLM applications easily",
    "ChromaDB is an open source vector database",
    "Embeddings convert text into numerical vectors",
    "RAG stands for Retrieval Augmented Generation",
    "Groq provides fast LLM inference API",
    "Qdrant is a vector database for production use",
    "CLIP model connects text and images together",
    "PyMuPDF extracts text and images from PDFs",
    "Next.js is a React framework for web apps",
    "Tailwind CSS is a utility first CSS framework",
    "Three.js creates 3D graphics in the browser",
    "GSAP is a professional animation library",
    "FastAPI supports async await for performance",
    "Vector search finds semantically similar text",
    "Cosine similarity measures angle between vectors",
    "Chunk size affects RAG retrieval quality",
    "RAGAS framework evaluates RAG system quality",
    "Multimodal RAG handles both text and images",
]

categories = [
    "python","backend","ai","database","ai",
    "ai","api","database","ai","pdf",
    "frontend","frontend","frontend","frontend","backend",
    "ai","math","ai","evaluation","ai"
]

# Generate embeddings
print("\nGenerating embeddings...")
embeddings = model.encode(chunks).tolist()

# Add with metadata
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{
        "category": categories[i],
        "chunk_index": i,
        "word_count": len(chunks[i].split()),
        "source": "day4_test"
    } for i in range(len(chunks))]
)

print(f"✅ Added chunks!")
print(f"Total in DB: {collection.count()}")

print("\n" + "="*50)
print("QUERYING CHROMADB")
print("="*50)

questions = [
    "What tools are used for vector databases?",
    "How do I build a frontend web application?",
    "What is used for AI and machine learning?"
]

for question in questions:
    query_embedding = model.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "distances", "metadatas"]
    )
    
    print(f"\n Question: {question}")
    print("Top 3 results:")
    for i, (doc, dist, meta) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        print(f"  {i+1}. [{meta['category']}] {doc}")
        print(f"     Distance: {dist:.3f}")



print("\n" + "="*50)
print("METADATA FILTERING")
print("="*50)

query_embedding = model.encode(
    ["What frameworks exist?"]
).tolist()

filtered_results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    where={"category": "frontend"},
    include=["documents", "metadatas"]
)

print("\n🔍 Searching ONLY in frontend category:")
for doc, meta in zip(
    filtered_results['documents'][0],
    filtered_results['metadatas'][0]
):
    print(f"  → [{meta['category']}] {doc}")


print("\n" + "="*10)
print("CRUD OPERATIONS")


# Count
print(f"Total chunks: {collection.count()}")

# Update
collection.update(
    ids=["chunk_0"],
    documents=["Python is the most popular AI language"],
    embeddings=model.encode(
        ["Python is the most popular AI language"]
    ).tolist(),
    metadatas=[{
        "category": "python",
        "chunk_index": 0,
        "word_count": 8,
        "source": "updated"
    }]
)
print("✅ Updated chunk_0!")

# Verify update
updated = collection.get(ids=["chunk_0"])
print(f"Updated text: {updated['documents'][0]}")

# Delete
collection.delete(ids=["chunk_19"])
print("✅ Deleted chunk_19!")

# Count after delete
print(f"Total after delete: {collection.count()}")

# Peek at first 3
peek = collection.peek(limit=3)
print("\nFirst 3 items in collection:")
for doc in peek['documents']:
    print(f"  → {doc}")
