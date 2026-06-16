from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model = "nomic-embed-text")

vocab = ["cat", "my", "is", "fluffy", "kitten"]

#embed all
embedvector = []
for i in vocab:
  embedvector.append(embed.embed_query(i)); 

import numpy as np

def consine_similarity(a, b) :
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i in range(len(embedvector)) :
  for j in range(i + 1, len(embedvector)) :
    print(f"{vocab[i]} vs {vocab[j]} = {consine_similarity(embedvector[i], embedvector[j])}")

