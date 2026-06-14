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
from collections import defaultdict
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings
from args import *
from transformers import BertTokenizer,BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import datetime

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()


def load_data(dataset=None):

    print(f'__loading__{args.dataset}__')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv", sep='\t', keep_default_na=False)
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv", sep='\t', keep_default_na=False)
    return train_df, test_df


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


# 构造映射，对齐论文 Algorithm 1（data-independent，遍历整个词表；与 sst2_new 一致）
def build_sim_word_dict(top_k):
    if args.mapping_strategy != "paper":
        raise ValueError(
            "qnli_new 已与 sst2_new 对齐，请使用 --mapping_strategy paper。"
            "conservative/aggressive 为旧版语料相关映射，已不再实现。"
        )

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

    is_negative_corr = (args.embedding_type == "glove_840B-300d")

    if is_negative_corr:
        def get_topk_indices(src_emb, all_emb, k):
            dists = euclidean_distances(src_emb.reshape(1, -1), all_emb)[0]
            return dists.argsort()[:k]

        def compute_pairwise_dist(src_emb_list, tgt_emb_list):
            return euclidean_distances(np.array(src_emb_list), np.array(tgt_emb_list))
    else:
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1
        embeddings = embeddings / norm

        def get_topk_indices(src_emb, all_emb, k):
            sims = np.dot(src_emb, all_emb.T)
            return sims.argsort()[::-1][:k]

        def compute_pairwise_dist(src_emb_list, tgt_emb_list):
            src_arr = np.array(src_emb_list)
            tgt_arr = np.array(tgt_emb_list)
            norm_s = np.linalg.norm(src_arr, axis=1, keepdims=True)
            norm_s[norm_s == 0] = 1
            norm_t = np.linalg.norm(tgt_arr, axis=1, keepdims=True)
            norm_t[norm_t == 0] = 1
            src_arr = src_arr / norm_s
            tgt_arr = tgt_arr / norm_t
            return np.dot(src_arr, tgt_arr.T)

    sim_word_dict = {}
    sim_dist_dict = {}
    unmapped = set(range(vocab_size))
    search_embeddings = embeddings.copy()

    pbar = trange(vocab_size, desc="Algorithm 1 mapping")
    while len(unmapped) >= top_k:
        x_idx = next(iter(unmapped))
        index_list = get_topk_indices(embeddings[x_idx], search_embeddings, top_k)
        word_list = [idx2word[i] for i in index_list]
        y_prime_emb = [embeddings[i] for i in index_list]

        dist_matrix = compute_pairwise_dist(y_prime_emb, y_prime_emb)
        d_min = dist_matrix.min()
        d_max = dist_matrix.max()
        if d_max - d_min == 0:
            d_max = d_min + 1

        for local_j, global_idx in enumerate(index_list):
            w = idx2word[global_idx]
            dist_row = dist_matrix[local_j]
            if is_negative_corr:
                scores = [-(v - d_min) / (d_max - d_min) for v in dist_row]
            else:
                scores = [(v - d_min) / (d_max - d_min) for v in dist_row]
            sim_word_dict[w] = word_list
            sim_dist_dict[w] = scores

        unmapped -= set(index_list)
        if is_negative_corr:
            inf_emb = np.full(embeddings.shape[1], 1e9)
        else:
            inf_emb = np.zeros(embeddings.shape[1])
        for i in index_list:
            search_embeddings[i, :] = inf_emb

        pbar.update(len(index_list))

    if len(unmapped) > 0:
        remain_list = sorted(unmapped)
        k_prime = len(remain_list)
        word_list = [idx2word[i] for i in remain_list]
        y_prime_emb = [embeddings[i] for i in remain_list]

        dist_matrix = compute_pairwise_dist(y_prime_emb, y_prime_emb)
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

    with open(f"{p_dict_dir}/eps_{eps}_top_{args.top_k}.txt", 'w') as json_file:
        json_file.write(json.dumps(p_dict, ensure_ascii=False, indent=4))

    return p_dict


def _privatize_text(text, sim_word_dict, p_dict, save_stop_words, stop_words):
    record = str(text).split()
    new_record = []
    for word in record:
        if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
            if is_number(word):
                try:
                    word = str(round(float(word)) + np.random.randint(1000))
                except Exception:
                    pass
            new_record.append(word)
        else:
            p = p_dict[word]
            new_word = np.random.choice(sim_word_dict[word], 1, p=p)[0]
            new_record.append(new_word)
    return " ".join(new_record)


def generate_new_sents_s1(df, sim_word_dict, p_dict, save_stop_words, type="train"):
    """对 QNLI 的 sentence、question 两列脱敏并落盘（路径含 seed，与 sst2_new 一致）。"""
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    sentence_col = df.sentence
    new_sentence = []
    for i in trange(len(sentence_col), desc=f"privatize sentence ({type})"):
        new_sentence.append(
            _privatize_text(
                sentence_col.iloc[i], sim_word_dict, p_dict, save_stop_words, stop_words
            )
        )
    df.sentence = new_sentence

    question_col = df.question
    new_question = []
    for i in trange(len(question_col), desc=f"privatize question ({type})"):
        new_question.append(
            _privatize_text(
                question_col.iloc[i], sim_word_dict, p_dict, save_stop_words, stop_words
            )
        )
    df.question = new_question

    priv_dir = (
        f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
        f"/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{args.save_stop_words}_seed_{args.seed}"
    )
    os.makedirs(priv_dir, exist_ok=True)
    out_name = "train.tsv" if type == "train" else "test.tsv"
    df.to_csv(os.path.join(priv_dir, out_name), sep="\t", index=False)

    return df

