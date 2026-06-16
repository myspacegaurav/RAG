from langchain_ollama import ChatOllama

chatObject = ChatOllama(model="llama3.2:1b")

questions = ["hi","yo"]

for i in questions:
  response = chatObject.invoke(i)
  data = dict(response)
  print(data["content"])       
