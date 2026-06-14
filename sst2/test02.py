# import torch
# w = torch.tensor(2.0, requires_grad=True)
# print(w)
# print("="*20)
# print(w.grad)
# x = torch.tensor(3.0)
# loss = (w * x) ** 2   # 产生计算图
# loss.backward()        # 计算 d(loss)/dw
# print(w.grad)          # 梯度
import pandas as pd
df_train = pd.read_csv(f"datasets/sst2/train.tsv",sep='\t')
df_dev = pd.read_csv(f"datasets/sst2/dev.tsv",sep = "\t")
#print(type(df_train))#dataframe

train_corpus = " ".join(df_train.sentence)
print(type(train_corpus))
dev_corpus = " ".join(df_dev.sentence)
print(type(dev_corpus))
print(len(train_corpus))
print(len(dev_corpus))
print(len(train_corpus)+len(dev_corpus))
corpus = train_corpus + dev_corpus
print(len(corpus))
print("corpus type:",type(corpus))
from collections import Counter
total_words = len(corpus.split())
print("total_words",total_words)
word_freq = [x[0] for x in Counter(corpus.split()).most_common()]
print(len(word_freq))
print(type(word_freq))
print(word_freq[:10])

from tqdm import trange
for i in trange(len(word_freq)):
    word = word_freq[i]
    print(i)
