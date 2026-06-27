#better chunking
doc_path = [r"notes\os.txt", r"notes\coa.txt", r"notes\rdbms.txt", r"notes\ml.txt"]

#split document - size based
def chunker(text, size, overlap):
  words = text.split()
  chunks = []

  i = 0
  while i < len(words):
      chunk = " ".join(words[i:i+size])
      chunks.append(chunk)
      i += size - overlap
  
  return chunks

#embeddings
from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model="nomic-embed-text")

import numpy as np
import faiss
import os
import pickle

INDEX_PATH = "index.faiss"
CHUNKS_PATH = 'chunks.pkl'

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
  index = faiss.read_index(INDEX_PATH)
  with open(CHUNKS_PATH, "rb") as f:
    data = pickle.load(f)
    chunks, metadata = data["chunks"], data["metadata"]

else:
  chunks = []
  metadata = []

  for doc in doc_path:
   with open(doc, "r") as file:
    document = file.read()

    chunk_doc = chunker(document, 30, 5)
    for c in chunk_doc:
      chunks.append(c)
      metadata.append({
        "sources" : doc,
        "chunk" : len(chunks) - 1
      })

  embeddings = []
  for i in chunks:
    embeddings.append(embed.embed_query(i))

  vector = np.array(embeddings).astype('float32')
  faiss.normalize_L2(vector)
  index = faiss.IndexFlatIP(len(vector[0]))
  index.add(vector)

  faiss.write_index(index, INDEX_PATH)
  with open(CHUNKS_PATH, "wb") as f:
    pickle.dump({"chunks": chunks, "metadata": metadata}, f)

query = input("ask your question: ")
q_embed = np.array([embed.embed_query(query)]).astype('float32')
faiss.normalize_L2(q_embed)

k = 5
distance, indices = index.search(q_embed, k)

top_doc = []
sources = []
THRESHOLD = 0.65
for j,score in zip(indices[0], distance[0]):
  if score >= THRESHOLD:
    top_doc.append((chunks[j]))
    sources.append(metadata[j])

if not top_doc:
    print("I don't know")
    exit()
else:
  context_str = "\n".join(top_doc)


# combine retrieval + LLM
from langchain_ollama import ChatOllama

chat_model = ChatOllama(
  model="phi4-mini",
  temperature=0
)

prompt = f"""
You are a helpful assistant. You MUST follow these rules strictly:
- Answer ONLY using the context given below
Answer:- Do NOT use any outside knowledge
- Do NOT guess
- keep simple short answer

Context:
{context_str}

Question:
{query}

Answer:
"""

#validation-step to minimize hallucination
response = chat_model.invoke(prompt).content

response_embed = np.array([embed.embed_query(response)]).astype('float32')
faiss.normalize_L2(response_embed)
k = 1
res_dist, _ = index.search(response_embed, k)

if res_dist[0][0] > THRESHOLD:
  print(response)
  print("\n--- Sources Used ---")
  for s in sources:
      print(s)
else:
   print("I Don't know")