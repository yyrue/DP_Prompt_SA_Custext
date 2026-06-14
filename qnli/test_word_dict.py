import pandas as pd
total = 0
with open("embeddings/glove_840B-300d.txt", "r") as f:
    for i, line in enumerate(f):
        total += 1
print(total)
total2 = 0
with open("embeddings/ct_vectors.txt", "r") as f:
    for i, line in enumerate(f):
        total2 += 1
print(total2)