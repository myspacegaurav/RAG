from langchain_ollama import ChatOllama

chat = ChatOllama(
  model="phi4-mini",
  temperature=0
)

query = input()
queries = [
    query,
    "Explain " + query,
    "Definition of " + query
]

response = chat.invoke(queries)
# queries = [q.strip() for q in response.content.split('\n') if q.strip()]

# print(queries)

print(response.content)