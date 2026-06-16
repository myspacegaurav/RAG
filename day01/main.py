print("Hello RAG World!")
name = "gaurav"
age = 20
print(f"My name is {name} and I am {age} years old.")

documents = ["AI is transforming the world.", "RAG is a powerful technique for information retrieval.", "Python is a versatile programming language."]

for doc in documents:
    print(f"Document: {doc}")

def count_words(document, words_list):
    word_list = document.split()
    words_list.extend(word_list)
    word_count = len(word_list)
    return word_count

words = []
print(f"Number of words in the first document: {count_words(documents[0], words)}")
print(f"words list: {words}")