def count_words(text):
  return len(text.split())

#read file
with open("notes.txt", "r") as file:
  content = file.read()


# line by line reading
for line in content:
  print(line.strip())

print("Words:", count_words(content))
print("characters:", len(content))