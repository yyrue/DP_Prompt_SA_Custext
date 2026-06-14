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
import matplotlib.pyplot as plt
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

#加载数据，每个df有两列：sentence,label
def load_data(dataset=None):

    print(f'__loading__{args.dataset}__')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv", sep='\t', keep_default_na=False)
    dev_df = pd.read_csv(f"datasets/{args.dataset}/dev.tsv", sep='\t', keep_default_na=False)
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv", sep='\t', keep_default_na=False)
    return train_df,dev_df,test_df


class Bert_dataset(Dataset):
    # 把 DataFrame 包装成 PyTorch 可用的数据集
    # 用 BertTokenizer 把句子编码成 input_ids / attention_mask / token_type_ids
    # 返回 (input_ids, attention_mask, token_type_ids, label) 四元组
    def __init__(self,df):
        self.df=df
        self.tokenizer = BertTokenizer.from_pretrained(f"{args.model_type}",do_lower_case=True)
    #用[]访问对象时自动调用这个方法
    def __getitem__(self,index):
        # get the sentence from the dataframe
        sentence = self.df.loc[index,'sentence']

        encoded_dict = self.tokenizer.encode_plus(
            sentence,              # sentence to encode
            add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
            max_length = args.max_len,
            pad_to_max_length= True,
            truncation='longest_first',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        # These are torch tensors already
        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        #Convert the target to a torch tensor
        target = torch.tensor(self.df.loc[index,'label'])

        sample = (input_ids,attention_mask,token_type_ids,target)
        return sample
    #对对象调用len()函数时自动调用这个方法
    #print(len(dataset))
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


#构造映射，mapping，对齐论文 Algorithm 1（data-independent，遍历整个词表）
def build_sim_word_dict(top_k):
    # ---- 1. 加载词表和嵌入（data-independent，不依赖训练数据） ----
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

    embeddings = np.asarray(embeddings, dtype="float64")
    idx2word = np.asarray(idx2word)
    vocab_size = len(idx2word)
    print(f"Vocabulary size: {vocab_size}")

    # ---- 2. 判断相似度类型：GloVe 用欧氏距离（负相关），Counter-Fitting 用余弦相似度（正相关） ----
    is_negative_corr = (args.embedding_type == "glove_840B-300d")

    if is_negative_corr:
        # GloVe: 使用欧氏距离
        # 对每个 token，计算它与词表中所有 token 的欧氏距离，取 top_k 最近邻
        def get_topk_indices(src_emb, all_emb, k):
            dists = euclidean_distances(src_emb.reshape(1, -1), all_emb)[0]
            return dists.argsort()[:k]

        def compute_pairwise_dist(src_emb_list, tgt_emb_list):
            """计算 |X'| x |Y'| 的欧氏距离矩阵"""
            return euclidean_distances(np.array(src_emb_list), np.array(tgt_emb_list))
    else:
        # Counter-Fitting: 使用余弦相似度（已归一化，点积即余弦）
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1  # 防止除零
        embeddings = embeddings / norm

        def get_topk_indices(src_emb, all_emb, k):
            sims = np.dot(src_emb, all_emb.T)
            return sims.argsort()[::-1][:k]

        def compute_pairwise_dist(src_emb_list, tgt_emb_list):
            """计算 |X'| x |Y'| 的余弦相似度矩阵"""
            src_arr = np.array(src_emb_list)
            tgt_arr = np.array(tgt_emb_list)
            norm_s = np.linalg.norm(src_arr, axis=1, keepdims=True)
            norm_s[norm_s == 0] = 1
            norm_t = np.linalg.norm(tgt_arr, axis=1, keepdims=True)
            norm_t[norm_t == 0] = 1
            src_arr = src_arr / norm_s
            tgt_arr = tgt_arr / norm_t
            return np.dot(src_arr, tgt_arr.T)  # shape: (|X'|, |Y'|)

    # ---- 3. 算法1：遍历词表，逐轮构建 K-to-K 映射 ----
    sim_word_dict = {}
    sim_dist_dict = {}
    unmapped = set(range(vocab_size))  # 尚未映射的 token 索引集合

    # search_embeddings: 用于搜索最近邻的嵌入矩阵副本
    # 所有策略都需要将已映射 token 从搜索空间中移除，严格对齐算法1第9行 "X ← X \ Y', Y ← Y \ Y'"
    search_embeddings = embeddings.copy()

    pbar = trange(vocab_size, desc="Algorithm 1 mapping")
    while len(unmapped) >= top_k:
        # Step 2: Pick an arbitrary token x from X (取 unmapped 中的第一个)
        x_idx = next(iter(unmapped))
        # Step 3-6: 初始化 Y' = {x}，找 top-(K-1) 语义最近邻
        index_list = get_topk_indices(embeddings[x_idx], search_embeddings, top_k)
        word_list = [idx2word[i] for i in index_list]
        # Y' 的嵌入矩阵
        y_prime_emb = [embeddings[i] for i in index_list]

        # Step 7-8: 对 Y' 中的每个 token，分配相同的输出集 Y'（K-to-K 映射）
        # 同时按论文 4.2 节做全局归一化（X'=Y'，所以 pairwise 是 |Y'| x |Y'|）
        dist_matrix = compute_pairwise_dist(y_prime_emb, y_prime_emb)  # (K, K)
        d_min = dist_matrix.min()
        d_max = dist_matrix.max()
        if d_max - d_min == 0:
            d_max = d_min + 1  # 防止除零

        for local_j, global_idx in enumerate(index_list):
            w = idx2word[global_idx]
            dist_row = dist_matrix[local_j]
            if is_negative_corr:
                # 负相关度量：u(x,y) = -(d'(x,y))
                scores = [-(v - d_min) / (d_max - d_min) for v in dist_row]
            else:
                # 正相关度量：u(x,y) = d'(x,y)
                scores = [(v - d_min) / (d_max - d_min) for v in dist_row]
            sim_word_dict[w] = word_list
            sim_dist_dict[w] = scores

        # Step 9: 从 X 和 Y 中移除已映射的 token
        unmapped -= set(index_list)

        # 将已映射 token 的 embedding 置为极端值，使其不再被后续搜索选中
        # 对齐算法1第9行 "X ← X \ Y', Y ← Y \ Y'"
        if is_negative_corr:
            inf_emb = np.full(embeddings.shape[1], 1e9)
        else:
            inf_emb = np.zeros(embeddings.shape[1])
        for i in index_list:
            search_embeddings[i, :] = inf_emb

        pbar.update(len(index_list))

    # Step 10: 处理剩余 token（K' < K），做 K'-to-K' 映射
    if len(unmapped) > 0:
        remain_list = sorted(unmapped)
        k_prime = len(remain_list)
        word_list = [idx2word[i] for i in remain_list]
        y_prime_emb = [embeddings[i] for i in remain_list]

        dist_matrix = compute_pairwise_dist(y_prime_emb, y_prime_emb)  # (K', K')
        d_min = dist_matrix.min()
        d_max = dist_matrix.max()
        if d_max - d_min == 0:
            d_max = d_min + 1

        for local_j, global_idx in enumerate(remain_list):
            w = idx2word[global_idx]
            dist_row = dist_matrix[local_j]
            if is_negative_corr:
                scores = [-(v - d_min) / (d_max - d_min) for v in dist_row]
            else:
                scores = [(v - d_min) / (d_max - d_min) for v in dist_row]
            sim_word_dict[w] = word_list
            sim_dist_dict[w] = scores

        pbar.update(k_prime)

    pbar.close()
    print(f"Total tokens mapped: {len(sim_word_dict)}")

    # ---- 4. 缓存 ----
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
    '''
    df: 原始数据集（train.tsv或dev.tsv），有sentence列和label列
    sim_word_dict: 每个词的相似词列表
    p_dict: 每个词的替换概率的分布
    save_stop_words: 是否保存停用词
    type: train或test,决定保存的文件名
    '''
    punct = list(string.punctuation) #标点符号列表（实际没有用到）

    nltk.download('stopwords') #下载英文停用词表
    nltk.download('punkt') #下载分词工具（实际没有用到）
    stop_words = set(stopwords.words('english')) #stop_words = {"the", "a", "an", "is", "are", "was", ...} 约180个词
    
    cnt = 0 #总词数
    raw_cnt = 0 #原始词数
    stop_cnt = 0 #停用词数
    dataset = df.sentence 
    new_dataset = []

    for i in trange(len(dataset)):
        record = dataset[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict): #是停用词或不在sim_word_dict中的词(OOV)
                if word in stop_words:
                    stop_cnt += 1  
                    raw_cnt += 1   
                if is_number(word):#数字特殊处理，加随机噪声
                    try:
                        word = str(round(float(word))+np.random.randint(1000))
                    except:
                        pass                   
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word],1,p=p)[0]
                new_record.append(new_word)
                if new_word == word:
                    raw_cnt += 1 

            cnt += 1 
        new_dataset.append(" ".join(new_record))

    df.sentence = new_dataset

    priv_dir = f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}_seed_{args.seed}"
    if not os.path.exists(priv_dir):
        os.makedirs(priv_dir, exist_ok=True)
    if type == "train":
        df.to_csv(f"{priv_dir}/train.tsv", sep="\t", index=False)
    else:
        df.to_csv(f"{priv_dir}/test.tsv", sep="\t", index=False)

    return df
