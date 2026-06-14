total_ct = 0
with open("ct_vectors.txt", "r") as f:
    for i,line in enumerate(f):
        total_ct += 1
print(total_ct)
total_glove = 0
with open("glove_840B-300d.txt", "r") as f:
    for i , line in enumerate(f):
        total_glove += 1
print(total_glove)
