import warnings
warnings.filterwarnings("ignore")

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

print("="*50)
print("DAY 6 — FIRST COMPLETE RAG PIPELINE")
print("="*50)

# 20 paragraphs about AI/RAG topic
paragraphs = [
    "RAG stands for Retrieval Augmented Generation. It combines vector search with large language models to answer questions from documents accurately.",
    "ChromaDB is an open source vector database that stores embeddings and allows fast similarity search using HNSW algorithm.",
    "LangChain is a framework that makes it easy to build LLM applications by chaining together prompts, models and output parsers.",
    "Sentence Transformers convert text into dense vector embeddings of fixed size. The all-MiniLM-L6-v2 model outputs 384 dimensional vectors.",
    "Cosine similarity measures the angle between two vectors. A score of 1.0 means identical meaning and 0.0 means completely unrelated.",
    "Groq provides ultra fast LLM inference API. It uses LPU hardware to run LLaMA models much faster than traditional GPU servers.",
    "Qdrant is a production grade vector database built in Rust. It supports filtering, payload storage and horizontal scaling.",
    "PyMuPDF is a Python library for extracting text and images from PDF files. It is faster and more accurate than PyPDF2.",
    "FastAPI is a modern Python web framework for building REST APIs. It supports async await and auto generates Swagger documentation.",
    "CLIP is a multimodal model from OpenAI that creates embeddings for both text and images in the same vector space.",
    "RAGAS is an evaluation framework for RAG systems. It measures faithfulness, answer relevancy, context precision and context recall.",
    "Chunk size affects RAG quality significantly. Too small loses context. Too large adds noise. 256 to 512 words is usually optimal.",
    "Chunk overlap prevents information loss at boundaries. Overlapping 10 to 20 percent of chunk size captures split sentences.",
    "Metadata filtering in ChromaDB allows searching specific subsets. For example filtering by source PDF or user ID.",
    "JWT authentication ensures only authorized users access the API. Tokens are generated at login and verified on every request.",
    "Docker containerizes the entire application including Python dependencies. This ensures consistent behavior across all environments.",
    "Next.js is a React framework with server side rendering. Combined with Tailwind CSS it enables building fast modern web UIs.",
    "Three.js enables 3D graphics in the browser using WebGL. React Three Fiber provides React components for Three.js objects.",
    "Hybrid search combines dense vector search with sparse BM25 keyword search. This improves retrieval accuracy significantly.",
    "IKS stands for Indian Knowledge System. It includes Ayurveda, Yoga Sutras and Sanskrit texts which can be queried using RAG.",
]

# STEP 1: Embed paragraphs
print("\n Step 1: Embedding 20 paragraphs...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(paragraphs)
print(f" Shape: {embeddings.shape}")

# STEP 2: Store in ChromaDB
print("\n Step 2: Storing in ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db_day6")
collection = client.get_or_create_collection(
    name="rag_paragraphs",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    documents=paragraphs,
    embeddings=embeddings.tolist(),
    ids=[f"para_{i}" for i in range(len(paragraphs))],
    metadatas=[{
        "para_index": i,
        "word_count": len(p.split()),
        "source": "day6_knowledge_base"
    } for i, p in enumerate(paragraphs)]
)
print(f" Stored {collection.count()} paragraphs!")

# STEP 3: Build RAG pipeline
print("\n Step 3: Building RAG pipeline...")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)
parser = StrOutputParser()

def rag_pipeline(question, n_results=5):
    # Retrieve top 5
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    
    retrieved = results["documents"][0]
    distances = results["distances"][0]
    
    # Build context
    context = "\n\n".join([
        f"Source {i+1}: {doc}"
        for i, doc in enumerate(retrieved)
    ])
    
    # RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI assistant.
Answer using ONLY the sources below.
Always cite which Source number your answer came from.
If answer not found say 'Not in knowledge base.'

Sources:
{context}"""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | parser
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return answer, retrieved, distances

# STEP 4: Test with questions
print("\n" + "="*50)
print("TESTING RAG PIPELINE")
print("="*50)

questions = [
    "What is RAG and how does it work?",
    "Which vector databases are available?",
    "What is chunk overlap and why is it important?",
    "What is IKS?",
    "What is the best programming language for web development?",
]

for question in questions:
    print(f"\n {question}")
    answer, sources, distances = rag_pipeline(question)
    print(f" Top source: {sources[0][:60]}...")
    print(f" Distance: {distances[0]:.3f}")
    print(f" {answer}")