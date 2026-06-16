from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama3.2:1b")

response = chat_model.invoke(
  ["hi","what is AI","why python is used in AI"]
)
print(response.content)