# Count words in a document

input_document = "RAG is a powerful technique in AI"

def count_words(doc):
  return len(doc.split())

print(f"words : {count_words(input_document)}")

# Count characters in a document
def count_characters(doc):
  return len(doc)

print(f"characters : {count_characters(input_document)}")


print(input_document.lower())
print(input_document.upper())

print(input_document.replace("AI","Artificial Intelligence"))