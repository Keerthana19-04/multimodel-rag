from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Create LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# Step 2: Create Prompt Template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI assistant. Answer clearly and concisely."),
#     ("human", "{question}")
# ])


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI assistant specializing in:
- RAG (Retrieval Augmented Generation) systems
- Vector databases like ChromaDB and Qdrant
- LLM frameworks like LangChain
- Machine learning and embeddings
Answer accurately and concisely based on your expertise."""),
    ("human", "{question}")
])


# Step 3: Create Output Parser
parser = StrOutputParser()

# Step 4: Build Chain
chain = prompt | llm | parser

# Step 5: Run Chain
print("-"*25)
print("LANGCHAIN BASIC CHAIN")
print("-"*25)

questions = [
    "What is RAG in one sentence?",
    "What is ChromaDB used for?",
    "Why do we use vector embeddings?",
]

for question in questions:
    print(f"\n {question}")
    answer = chain.invoke({"question": question})
    print(f" {answer}")


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

print("\n" + "-"*50)
print("LANGCHAIN + CHROMADB RAG CHAIN")
print("-"*50)

# Load existing ChromaDB from Day 4
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="rag_chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant.
Answer using ONLY the context below.
If answer not in context say 'I don't know'.

Context: {context}"""),
    ("human", "{question}")
])

# Helper to format docs
def format_docs(docs):
    return "\n\n".join(
        f"Chunk {i+1}: {doc.page_content}" 
        for i, doc in enumerate(docs)
    )

# Build RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | parser
)

# Test RAG chain
rag_questions = [
    "What vector databases are mentioned?",
    "What frontend frameworks are available?",
    "What is used for AI evaluation?",
]

for question in rag_questions:
    print(f"\n {question}")
    answer = rag_chain.invoke(question)
    print(f" {answer}")

