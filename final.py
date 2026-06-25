#better chunking

with open('notes.txt', "r") as file:
  document = file.read()

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
    chunks = pickle.load(f)

else:
  chunks = chunker(document, 8, 6)
  embeddings = []
  for i in chunks:
    embeddings.append(embed.embed_query(i))

  vector = np.array(embeddings).astype('float32')
  index = faiss.IndexFlatL2(len(vector[0]))
  index.add(vector)

  faiss.write_index(index, INDEX_PATH)
  with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

query = input("ask your question :")
q_embed = np.array([embed.embed_query(query)]).astype('float32')

k = 2
distance, indices = index.search(q_embed, k)

top_doc = []
sources = []
for j in indices[0]:
  top_doc.append((chunks[j]))
  sources.append(f"Chunk [{j}] : {chunks[j]}")


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
- Is the answer to the question present in the context above? If yes, answer   using ONLY the context. If no, say "I don't know".
Answer:- Do NOT use any outside knowledge
- Do NOT guess
- keep simple short answer

Context:
{context_str}

Question:
{query}

Answer:
"""


response = chat_model.invoke(prompt)
if "I don't know" in response.content:
  print("Low confidence answer")

print(response.content)
print("\n--- Sources Used ---")
for s in sources:
    print(s)