#Document Analyzer
#read, count

def count_words(text):
  return len(text.split())

with open("notes.txt", "r") as file:
  content = file.read()
  

print("Words:", count_words(content))
print("characters:", len(content))
print("lines:", len(content.split("\n")))

chunks = content.split("\n")
print(chunks)