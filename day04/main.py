#tokenization - lines into tokens
#tokens - chunks of text
#"Ai is powereful" -> ["Ai", "is", "powerful"]
#unbelievable -> ["un", "believ", "able"]

#temperature -> controls creativity

from langchain_ollama import ChatOllama

chat_object = ChatOllama(
  model = "llama3.2:1b",
  temperature = 0.2
)

# response = chat_object.invoke("what is your name?")

# print(response.content)

questions = ["Explain AI", "Explain AI in one sentence", "You are a teacher. Explain AI simply in 2 lines"]

for q in questions:
  response = chat_object.invoke(q)
  print(response.content)
