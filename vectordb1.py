import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(model="nomic-embed-text")

documents = [
  "AI is the simulation of human intelligence",
  "Machine learning is a subset of AI",
  "Football is a sport",
  "Python is used in AI"
]

embeddings = []
for doc in documents:
  embeddings.append(embed.embed_query(doc))

vector = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(len(vector[0]))

index.add(vector)

query = input("ask anything: ")

q_vec = embed.embed_query(query)

q = np.array([q_vec]).astype("float32")

k = 2

distances, indices = index.search(q, k)

top_docs = []

for i in indices[0]:
  top_docs.append(documents[i])
