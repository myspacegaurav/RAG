with open(r"notes\os.txt","r") as f:
  document = f.read()

print(type(document))

word = document.split(".")
word = word[:3]

chunks = []
text = "H i."
print(text.strip("H"))
