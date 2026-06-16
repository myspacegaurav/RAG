#better chunking

with open('notes.txt', "r") as file:
  document = file.read()

#split document - size based
def chunker(text, size):
  text = text.split()
  chunks = []

  for i in range(0,len(text), size):
    words = " ".join(text[i : i + size])
    chunks.append(words) 
  
  return chunks

chunks = chunker(document, 50)

#embeddings
from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model="nomic-embed-text")

embeddings = []
for i in chunks:
  embeddings.append(embed.embed_query(i))

import numpy as np

def cos_sim(a, b):
  return np.dot(a,b)/ (np.linalg.norm(a)*np.linalg.norm(b))

query = input("ask your question :")
q_embed = embed.embed_query(query)

#top k retrieval
#get the k top similar context, sort them on the basis of cosine similarity

k = 2
context = []

for i in range(len(embeddings)):
  cos = cos_sim(q_embed, embeddings[i])
  context.append([cos, chunks[i]])
  

context.sort(reverse=True)

top_doc = []
for i in range(min(k, len(context))):
  top_doc.append(context[i][1])

context_str = "\n".join(top_doc)
print(context_str)

#combine retrieval + LLM
from langchain_ollama import ChatOllama

chat_model = ChatOllama(
  model="llama3.2:1b",
  temperature=0
)

prompt = f"""
  query : {query}
  context : {context_str}
  answer:
"""

response = chat_model.invoke(prompt)
print(response.content)
