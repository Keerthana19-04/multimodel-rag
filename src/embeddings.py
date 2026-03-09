from sentence_transformers import SentenceTransformer
import numpy as np

print("loading embedding model..")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded")

sentences = [
    "the cat sat on the mat",
    "A dog lay on the floor",
    "Machine learning is amazing",
    "Deep learning uses neural networks",
    "I love playing cricket",
    "Football is a popular sport",
    "Python is a programming language",
    "Java is used for backend development",
    "The sky is blue today",
    "It is a sunny and bright day",
]

print("\\nGenerating embedding..")
embeddings = model.encode(sentences)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Each sentnce = {embeddings.shape[1]}numbers")
print(f"\nFirst sentence vector (first 10 numbers):")
print(embeddings[0][:10])

from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between ALL sentence pairs
print("\n" + "="*50)
print("SIMILARITY SEARCH RESULTS")
print("="*50)

# Pick a query sentence
query = "The cat sat on the mat"
query_idx = 0

# Compare query against all other sentences
similarities = cosine_similarity([embeddings[query_idx]], embeddings)[0]

# Sort by similarity score
results = sorted(zip(sentences, similarities), 
                 key=lambda x: x[1], reverse=True)

print(f"\nQuery: '{query}'")
print("\nMost similar sentences:")
for sentence, score in results:
    bar = "█" * int(score * 20)
    print(f"{score:.3f} {bar} → {sentence}")

import chromadb

# Create ChromaDB client
print("\n" + "="*50)
print("STORING IN CHROMADB")
print("="*50)

client = chromadb.Client()

# Create a collection (like a table in SQL)
collection = client.create_collection(name="day2_sentences")

# Add all sentences with their embeddings
collection.add(
    documents=sentences,
    embeddings=embeddings.tolist(),
    ids=[f"sentence_{i}" for i in range(len(sentences))]
)

print(f"✅ Stored {len(sentences)} sentences in ChromaDB!")

# Now query ChromaDB
print("\n" + "="*50)
print("QUERYING CHROMADB")
print("="*50)

# Embed the query
query_text = "animals resting at home"
query_embedding = model.encode([query_text]).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print(f"\nQuery: '{query_text}'")
print("\nTop 3 results from ChromaDB:")
for i, (doc, distance) in enumerate(zip(
    results['documents'][0],
    results['distances'][0]
)):
    print(f"{i+1}. {doc}  (distance: {distance:.3f})")

    from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("\n" + "="*50)
print("MINI RAG PIPELINE")
print("="*50)

def mini_rag(question):
    # Step 1: Embed the question
    question_embedding = model.encode([question]).tolist()
    
    # Step 2: Retrieve top 3 from ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    retrieved_chunks = results['documents'][0]
    
    # Step 3: Build prompt with context
    context = "\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below.
    
Context:
{context}

Question: {question}
Answer:"""
    
    # Step 4: Send to Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content, retrieved_chunks

# Test your mini RAG!
questions = [
    "What animals are mentioned?",
    "Which programming languages are discussed?",
    "What sports are mentioned?"
]

for question in questions:
    print(f"\n❓ Question: {question}")
    answer, chunks = mini_rag(question)
    print(f"📚 Retrieved: {chunks}")
    print(f"🤖 Answer: {answer}")