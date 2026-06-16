from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(model="nomic-embed-text")

#lets determine similarity

text1 = "I love programming"
text2 = "Coding is fun"
text3 = "I like football"

#create embedded vector
vec1 = embed.embed_query(text1)
vec2 = embed.embed_query(text2)
vec3 = embed.embed_query(text3)

list = [vec1, vec2, vec3]

#find angle between vectors
import numpy as np

def cosine_similarity(a, b) :
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1--similar
# 0 --different
# -1 --opposite

for i in range(len(list) - 1) :
  for j in range(i + 1, len(list)) :
    print(f"text : {i + 1} and {j + 1} = {cosine_similarity(list[i], list[j])}")
