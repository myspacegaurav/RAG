#better chunking
doc_path = [r"notes\os.txt", r"notes\coa.txt", r"notes\rdbms.txt", r"notes\ml.txt"]

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

#split document - size based
def chunker(text, size, overlap):
  words = text.split()
  chunks = []

  i = 0
  while i < len(words):
      chunk = " ".join(words[i:i+size])
      chunks.append(chunk)
      i += size - overlap
  
  return chunks

#embeddings
from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model="nomic-embed-text")

from langchain_community.vectorstores import FAISS
import os
import pickle

INDEX_PATH = "index.faiss"
CHUNKS_PATH = 'chunks.pkl'

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
  embeddings = FAISS.load_local(INDEX_PATH, embed, allow_dangerous_deserialization=True)
  with open(CHUNKS_PATH, "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]

else:
  documents = []
  chunks = []
  for doc in doc_path:
    loader = TextLoader(doc)  
    documents.extend(loader.load())

  chunks = []
  for chunk in documents:
     c = chunk.page_content
     for chunk_text in chunker(c, 200, 50):
      chunks.append(Document(page_content= chunk_text, 
                              metadata = {"source":chunk.metadata["source"]}))

  embeddings = FAISS.from_documents(chunks, embed)
  
  embeddings.save_local(INDEX_PATH)
  with open(CHUNKS_PATH, "wb") as f:
    pickle.dump({"chunks": chunks}, f)


from langchain_ollama import ChatOllama

chat_model = ChatOllama(
  model="phi4-mini",
  temperature=0
)

query = input("ask your question: ")

#multiquerying

def generate_queries(original_query, n=3):
    prompt = f"""Generate {n} different ways to ask the following question.
Return only the questions, one per line, no numbering, no extra text.
Question: {original_query}"""
    
    response = chat_model.invoke(prompt).content
    queries = [q.strip() for q in response.split('\n') if q.strip()]
    queries.append(original_query)
    return queries

queries = generate_queries(query)

#deduplication using set 

def multiquery(queries, embeddings, k = 2, threshold = 0.65):
   seen = set()
   unique_docs = []

   for q in queries:
    results = embeddings.similarity_search_with_score(q, k)      
    for doc, score in results:
          if score <= threshold and doc.page_content not in seen:
              seen.add(doc.page_content)
              unique_docs.append(doc)
    
    return unique_docs


top_doc = multiquery(queries, embeddings)
   
#augumentation

if not top_doc:
    print("I don't know")
    exit()
else:
  context_str = "\n".join([doc.page_content for doc in top_doc])
  



#generation 

prompt = f"""
You are a helpful assistant. You MUST follow these rules strictly:
- Answer ONLY using the context given below
Answer:- Do NOT use any outside knowledge
- Do NOT guess

Context:
{context_str}

Question:
{query}

Answer:
"""

#validation-step to minimize hallucination
response = chat_model.invoke(prompt).content

hallucinate = embeddings.similarity_search_with_score(response, k = 1)

THRESHOLD = 0.65
for _, s in hallucinate:
   if(s <= THRESHOLD):
    print(response)
    