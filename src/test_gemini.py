from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Explain RAG (Retrieval Augmented Generation) in 3 simple sentences."}
    ]
)

print("=" * 50)
print("GROQ RESPONSE:")
print("=" * 50)
print(response.choices[0].message.content)



# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# model = SentenceTransformer("all-MiniLM-L6-v2")

# s1 = "I love eating pizza"
# s2 = "Pizza is my favourite food"
# s3 = "I love coding Python"

# emb = model.encode([s1, s2, s3])

# print("S1 vs S2:", cosine_similarity([emb[0]], [emb[1]])[0][0])
# print("S1 vs S3:", cosine_similarity([emb[0]], [emb[2]])[0][0])
# print("S2 vs S3:", cosine_similarity([emb[1]], [emb[2]])[0][0])





