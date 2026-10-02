import re

query = "sakdfjlkl23;l;aksfdjl;jsdlskksdkkdd+sfajhlkej.sdfkjhal,afhohf iefwifu"
words = set()
pattern = re.compile(r"\W+")
for word in re.split(pattern, query):
    if not word.strip():
        continue
    words.add(word)

print(words)
