import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm, trange
from collections import Counter,defaultdict
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings
from args import *
from transformers import BertTokenizer,BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import datetime
from logger import get_logger

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()


def load_data(dataset=None):

    print(f'__loading__{args.dataset}__')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv", sep='\t')
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv", sep='\t')
    return train_df,test_df


class Bert_dataset(Dataset):
    def __init__(self,df):
        self.df=df
        self.tokenizer = BertTokenizer.from_pretrained(f"{args.model_type}",do_lower_case=True)

    def __getitem__(self,index):
        question = self.df.loc[index,'question']
        sentence = self.df.loc[index,'sentence']

        encoded_dict = self.tokenizer.encode_plus(
            question,sentence,              # sentence to encode
            add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
            max_length = args.max_len,
            pad_to_max_length= True,
            truncation='longest_first',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        target = torch.tensor(self.df.loc[index,'label'])

        sample = (input_ids,attention_mask,token_type_ids,target)
        return sample

    def __len__(self):
        return len(self.df)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


#构造映射，mapping，聚类（只构建 sim_word_dict 和 sim_dist_dict，不涉及 eps）
def build_sim_word_dict(top_k):
    df_train = pd.read_csv(f"datasets/{args.dataset}/train.tsv", sep='\t')
    train_corpus = " ".join(df_train.sentence)
    dev_corpus = " ".join(df_train.question)
    corpus = train_corpus + " " + dev_corpus
    word_freq = [x[0] for x in Counter(corpus.split()).most_common()]


    if args.embedding_type == "glove_840B-300d":
        embedding_path = f'./embeddings/{args.embedding_type}.txt'
        embeddings = []
        idx2word = []
        word2idx = {}
        with open(embedding_path, 'r') as file:
            for i, line in enumerate(file):
                tokens = line.strip().split()
                word = ' '.join(tokens[:-300])
                embedding = [float(num) for num in tokens[-300:]]
                idx2word.append(word)
                word2idx[word] = i
                embeddings.append(embedding)

        embeddings = np.asarray(embeddings)
        idx2word = np.asarray(idx2word)
    else:
        embedding_path = f"./embeddings/{args.embedding_type}.txt"
        embeddings = []
        idx2word = []
        word2idx = {}
        with open(embedding_path,'r') as file:
            for i,line in enumerate(file):
                tokens = line.strip().split()
                # GloVe 300d: last 300 tokens are the embedding, the rest is the word
                embedding = [float(num) for num in tokens[-300:]]
                word = ' '.join(tokens[:-300])
                embeddings.append(embedding)
                idx2word.append(word)
                word2idx[word] = i
        embeddings = np.array(embeddings)
        idx2word = np.asarray(idx2word)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        print(embeddings.T.shape)


    # 构建 sim_word_dict 和 sim_dist_dict（记录距离，供后续计算概率用）
    if args.embedding_type == "glove_840B-300d":
        word_hash = defaultdict(str)
        sim_word_dict = defaultdict(list)
        sim_dist_dict = defaultdict(list)  # 新增：缓存距离信息
        for i in trange(len(word_freq)):
            word = word_freq[i]
            if word in word2idx:
                if word not in word_hash:
                    index_list = euclidean_distances(embeddings[word2idx[word]].reshape(1,-1),embeddings)[0].argsort()[:top_k]
                    word_list = [idx2word[x] for x in index_list]
                    embedding_list = np.array([embeddings[x] for x in index_list])
                        
                    if args.mapping_strategy == "aggressive":
                        sim_dist_list = euclidean_distances(embeddings[word2idx[word]].reshape(1,-1), embedding_list)[0]
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [-(x-min_dist)/min_max_dist for x in sim_dist_list]
                        sim_word_dict[word] = word_list
                        sim_dist_dict[word] = new_sim_dist_list
                    else:
                        for x in word_list:
                            if x not in word_hash:
                                word_hash[x] = word
                                sim_dist_list = euclidean_distances(embeddings[word2idx[x]].reshape(1,-1), embedding_list)[0]
                                min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                                min_dist = min(sim_dist_list)
                                new_sim_dist_list = [-(x-min_dist)/min_max_dist for x in sim_dist_list]
                                sim_word_dict[x] = word_list
                                sim_dist_dict[x] = new_sim_dist_list
                        if args.mapping_strategy == "conservative":
                            inf_embedding = [1e9] * 300
                            for i in index_list:
                                embeddings[i,:] = inf_embedding
    else:
        word_hash = defaultdict(str)
        sim_word_dict = defaultdict(list)
        sim_dist_dict = defaultdict(list)  # 新增：缓存距离信息
        for i in trange(len(word_freq)):
            word = word_freq[i]
            if word in word2idx:
                if word not in word_hash:
                    index_list = np.dot(embeddings[word2idx[word]], embeddings.T).argsort()[::-1][:top_k]
                    word_list = [idx2word[x] for x in index_list]
                    embedding_list = np.array([embeddings[x] for x in index_list])
                        
                    if args.mapping_strategy == "aggressive":
                        sim_dist_list = np.dot(embeddings[word2idx[x]], embedding_list.T)
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [(x-min_dist)/min_max_dist for x in sim_dist_list]
                        sim_word_dict[word] = word_list
                        sim_dist_dict[word] = new_sim_dist_list
                    else:
                        for x in word_list:
                            if x not in word_hash:
                                word_hash[x] = word
                                sim_dist_list = np.dot(embeddings[word2idx[x]], embedding_list.T)
                                min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                                min_dist = min(sim_dist_list)
                                new_sim_dist_list = [(x-min_dist)/min_max_dist for x in sim_dist_list]
                                sim_word_dict[x] = word_list
                                sim_dist_dict[x] = new_sim_dist_list
                        if args.mapping_strategy == "conservative":
                            inf_embedding = [0] * 300
                            for i in index_list:
                                embeddings[i,:] = inf_embedding

    # 缓存 sim_word_dict 和 sim_dist_dict（按 top_k 缓存，与 eps 无关）
    sim_word_dict_dir = f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}"
    sim_dist_dict_dir = f"./sim_dist_dict/{args.embedding_type}/{args.mapping_strategy}"
    os.makedirs(sim_word_dict_dir, exist_ok=True)
    os.makedirs(sim_dist_dict_dir, exist_ok=True)

    with open(f"{sim_word_dict_dir}/top_{args.top_k}.txt", 'w') as json_file:
        json_file.write(json.dumps(sim_word_dict, ensure_ascii=False, indent=4))

    with open(f"{sim_dist_dict_dir}/top_{args.top_k}.txt", 'w') as json_file:
        json_file.write(json.dumps(sim_dist_dict, ensure_ascii=False, indent=4))

    return sim_word_dict, sim_dist_dict


# 根据 sim_dist_dict 和 eps 计算概率分布（很快，无需加载嵌入）
def compute_p_dict(sim_dist_dict, eps):
    p_dict = defaultdict(list)
    for word, new_sim_dist_list in sim_dist_dict.items():
        tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
        norm_val = sum(tmp)
        p = [x/norm_val for x in tmp]
        p_dict[word] = p

    # 缓存 p_dict（按 eps + top_k 缓存）
    p_dict_dir = f"./p_dict/{args.embedding_type}/{args.mapping_strategy}"
    os.makedirs(p_dict_dir, exist_ok=True)

    with open(f"{p_dict_dir}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
        json_file.write(json.dumps(p_dict, ensure_ascii=False, indent=4))

    return p_dict

def generate_new_sents_s1(df,sim_word_dict,p_dict,save_stop_words,type="train"):

    punct = list(string.punctuation)
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    
    sentence = df.sentence
    new_sentence = []

    for i in trange(len(sentence)):
        record = sentence[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict): 
                if is_number(word):
                    try:
                        word = str(round(float(word))+np.random.randint(1000))
                    except:
                        pass                   
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word],1,p=p)[0]
                new_record.append(new_word)
        new_sentence.append(" ".join(new_record))

    df.sentence = new_sentence

    question = df.question
    new_question = []

    for i in trange(len(question)):
        record = question[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict): 
                if is_number(word):
                    try:
                        word = str(round(float(word))+np.random.randint(1000))
                    except:
                        pass                   
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word],1,p=p)[0]
                new_record.append(new_word)
        new_question.append(" ".join(new_record))

    df.question = new_question

    if not os.path.exists(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}"):
        os.makedirs(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}", exist_ok=True)
    if type == "train":
        df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/train.tsv", sep="\t", index=False)
    else:
        df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/test.tsv", sep="\t", index=False)

    return df

