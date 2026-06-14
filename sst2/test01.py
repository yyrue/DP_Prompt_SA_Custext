import os

glove = open(os.path.join('embeddings/', 'glove.840B.300d.txt'), 'r')
print(glove.readline())
total = 0
for i in range(300):
    print(glove.readline().split()[1:][i])
    total += float(glove.readline().split()[1:][i])
glove.close()
print(total)