import pandas as pd
import numpy as np
import json
from args import *
from collections import Counter
# parse = get_parser()
# args = parse.parse_args()
# print(args.dataset)
# train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv",sep="\t")
# print(type(train_df))
# print(len(train_df))
# print(train_df.head().sentence)
# print(train_df.label)


df_train = pd.read_csv(f"datasets/sst2/train.tsv",sep='\t')
df_dev = pd.read_csv(f"datasets/sst2/dev.tsv",sep='\t')
print(df_train.head())
print(df_dev.head())
print("="*20)

train_corpus = " ".join(df_train.sentence)
dev_corpus = " ".join(df_dev.sentence)
# print(type(train_corpus)) 
# print(type(dev_corpus))
# print(len(train_corpus))
# print(len(dev_corpus))

corpus = train_corpus + " " + dev_corpus
# print(len(corpus))

word_freq = [x[0] for x in Counter(corpus.split()).most_common()]
embedding_path = f"embeddings/glove_840B-300d.txt"
embeddings = []
idx2word = []
word2idx = {}
with open(embedding_path,'r') as file:
    for i,line in enumerate(file):
        tokens = line.strip().split()
        word = " ".join(tokens[:-300])
        embedding = [float(num) for num in tokens[-300:]]
        idx2word.append(word)
        word2idx[word] = i
        
print(len(embeddings))
print(len(idx2word))
print(len(word2idx))