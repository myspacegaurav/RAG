doc_path = [r"notes\coa.txt", r"notes\ml.txt", r"notes\ml.txt", r"notes\rdbms.txt"]

def chunker(text, size, overlap):
  chunk = text.split()
  chunks = []

  i = 0
  while i < len(chunk):
    chunks.append(" ".join(chunk[i : i + size]))
    i = i + size - overlap
  
  return chunks

metadata = []
chunks = []

for doc in doc_path:
  with open(doc, "r") as file:
    text = file.read()

    doc_chunks = (chunker(text,30, 5))

    for chunk in doc_chunks:
        chunks.append(chunk)
        metadata.append({
            "source": doc,
            "chunk_index": len(chunks) - 1
        })

from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model="nomic-embed-text")

embeddings = []
for c in chunks:
  embeddings.append(embed.embed_query(c))

import numpy as np
import faiss

vector = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(len(vector[0]))
index.add(vector)

query = input("Ask me anything: ")

q_embed = np.array([embed.embed_query(query)]).astype("float32")

_, indices = index.search(q_embed, 2)

top_docs = []
sources = []
for i in indices[0]:
  top_docs.append(chunks[i])
  sources.append(metadata[i])

context_str = "\n".join(top_docs)
print(context_str)

for i in sources:
  print(i)