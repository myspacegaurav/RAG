#document loader
doc_path = [r"notes\os.txt", r"notes\coa.txt", r"notes\rdbms.txt", r"notes\ml.txt"]

from langchain_community.document_loaders import TextLoader

docs = []
for file in doc_path:
  loader = TextLoader(file)
  docs.extend(loader.load())

print(type(docs))

#splitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
  chunk_size=200,
  chunk_overlap=50,
  separators=["\n\n", "\n", ".", "?", "!"]
)

chunks = splitter.split_documents(docs)
for chunk in chunks:
    print(chunk.page_content)
    print("---")
#embeddings

from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(
  model="nomic-embed-text"
)

#vector store
from langchain_community.vectorstores import FAISS
index = FAISS.from_documents(chunks, embed)

query = input("Ask questions: ")

results = index.similarity_search_with_score(query, k=2)

context = "\n".join([doc.page_content for doc, score in results])
# combine retrieval + LLM
from langchain_ollama import ChatOllama

chat_model = ChatOllama(
  model="phi4-mini",
  temperature=0
)

prompt = f"""
You are a helpful assistant. You MUST follow these rules strictly:
- Answer ONLY using the context given below
Answer:- Do NOT use any outside knowledge
- Do NOT guess
- keep simple short answer

Context:
{context}

Question:
{query}

Answer:
"""

response = chat_model.invoke(prompt)
print(response.content)