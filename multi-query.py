from langchain_ollama import ChatOllama

chat = ChatOllama(
  model="phi4-mini",
  temperature=0
)

query = input()

prompt = f"""Generate {4} different ways to ask the following question.
Return only the questions, one per line, no numbering, no extra text.
Question: {query}"""

queries = chat.invoke(prompt)

multiquery = []
for str in queries.content.split('\n') :
  str = str.strip()
  multiquery.append(str)

print(multiquery)