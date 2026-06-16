from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

embed = OllamaEmbeddings(model="nomic-embed-text")
chat_model = ChatOllama(
  model="llama3.2:1b",
  temperature=0
)

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


prompt = f"""
  you are a helpful assistant.
  Use the context below to answer the question.
  If the answer is not in the context that is {best_cos < 0.3}, say "I don't know".

  context:{best_ans}
  question:{query}
  answer:
"""
response = chat_model.invoke(prompt)
print(response.content)