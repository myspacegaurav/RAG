from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(model="nomic-embed-text")


with open("notes.txt", "r") as file:
  document = file.readlines()

embedding = []
for doc in document:
  embedding.append(embed.embed_query(doc))

import numpy as np

def cos_sim(a, b):
  return np.dot(a,b)/ (np.linalg.norm(a)*np.linalg.norm(b))


query = input("Ask you question :")
e_query = embed.embed_query(query)

best_ans = ""
best_cos = -1

for i in range(len(embedding)):
  cos = cos_sim(embedding[i],e_query)
  if(cos > best_cos):
    best_cos = cos
    best_ans = document[i]


print(best_ans)
