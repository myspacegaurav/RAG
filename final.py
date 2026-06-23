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

chunks = chunker(document, 8, 6)

#embeddings
from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model="nomic-embed-text")

embeddings = []
for i in chunks:
  embeddings.append(embed.embed_query(i))

import numpy as np
import faiss

vector = np.array(embeddings).astype('float32')

index = faiss.IndexFlatL2(len(vector[0]))

index.add(vector)

query = input("ask your question :")
q_embed = np.array([embed.embed_query(query)]).astype('float32')

k = 2
distance, indices = index.search(q_embed, k)

top_doc = []
for j in indices[0]:
  top_doc.append(chunks[j])
      

context_str = "\n".join(top_doc)


#combine retrieval + LLM
from langchain_ollama import ChatOllama

chat_model = ChatOllama(
  model="llama3.2:1b",
  temperature=0
)

prompt = f"""
You are a helpful assistant. You MUST follow these rules strictly:
- Answer ONLY using the context below
- If the context does not contain the answer, respond with exactly: "I don't know"
- Do NOT use any outside knowledge
- Do NOT guess

Context:
{context_str}

Question:
{query}

Answer:
"""

response = chat_model.invoke(prompt)
print(response.content)
