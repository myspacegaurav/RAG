from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(model="nomic-embed-text")


# text1 = "AI is powerful"
# text2 = "Artificial Intelligence is strong"

# vec1 = embed.embed_query(text1)
# print(len(vec1))

import numpy as np

#these are embedded vectors, here i type manually
cat = [0.23, -0.71, 0.45]
kitten = [0.25, -0.69, 0.44]
car = [0.91, 0.32, -0.21]

def cosine_similarity(a, b) :
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(cat, kitten)) 
print(cosine_similarity(cat, car)) 