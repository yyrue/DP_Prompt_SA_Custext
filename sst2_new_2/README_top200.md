# sst2_new_2 目录说明 —— `top_k = 200` 实验指南

本目录是 **CusText**（论文 *CusText: A Customized Text Sanitization Mechanism with Differential Privacy*）在 SST-2 数据集上的复现 / 实验工程，重点用于跑 **`top_k = 200`** 配置下的：

1. 脱敏数据集生成（不同 `eps`、不同 `seed`）
2. 下游分类任务（BERT 微调）
3. 隐私攻击评估（Mask Token Inference Attack、KNN Attack）
4. 与 Sample Amplification（混合策略）的对比实验

> **`top_k` 是 CusText 的核心超参**：对词表中每个 token，仅在其语义最近的 `K` 个 token 中按概率分布做 DP 随机替换。
> `top_k = 200` 表示输出集合大小 K = 200。词表来自 `embeddings/glove_840B-300d.txt`（**65 713 词**），所以共有 `⌈65713 / 200⌉ ≈ 329` 个 K-to-K 映射簇。

---

## 一、目录结构（与 `top_k=200` 相关的文件）

```
sst2_new_2/
├── args.py                            # 全局命令行参数（top_k / eps / embedding_type / ...）
├── utils.py                           # 核心算法：build_sim_word_dict / compute_p_dict / generate_new_sents_s1
├── main.py                            # 单次训练入口（BERT 微调）
├── batch_generate_private_data.py     # ★ 批量生成脱敏数据集（top_200 用它）
├── build_mapping_cache.py             # 仅构建映射缓存（不生成 TSV、不训练）
│
├── datasets/sst2/                     # 原始 SST-2 数据：train.tsv / dev.tsv / test.tsv
├── embeddings/
│   └── glove_840B-300d.txt            # 词向量（65 713 词 × 300 维）
│
├── sim_word_dict/glove_840B-300d/paper/top_200.txt    # ✅ 已生成 (251 MB) - Top200 自定义输出集合
├── sim_dist_dict/glove_840B-300d/paper/top_200.txt    # ✅ 已生成 (380 MB) - Top200 归一化相似度
├── p_dict/glove_840B-300d/paper/eps_1.0_top_200.txt   # ✅ 已生成 (198 MB) - eps=1.0 时的采样概率
│
├── privatized_dataset/                # 生成的脱敏 train.tsv / test.tsv（按 eps × seed 分目录）
├── experiment_results/                # main.py 训练日志和准确率
├── attack_results/                    # mask token 攻击结果
└── knn_attack_results/                # KNN 攻击结果
```

---

## 二、CusText 数据流（top_k=200 视角）

```
embeddings/glove_840B-300d.txt
        │
        │  build_sim_word_dict(top_k=200)   ← 算法1，data-independent，只与 K 有关
        ▼
sim_word_dict/.../top_200.txt   每个词 → 200 个候选词列表
sim_dist_dict/.../top_200.txt   每个词 → 200 个归一化相似度
        │
        │  compute_p_dict(sim_dist_dict, eps)   ← p_i ∝ exp(eps·u_i / 2)，只与 eps、K 有关
        ▼
p_dict/.../eps_<ε>_top_200.txt  每个词 → 长度 200 的概率分布
        │
        │  generate_new_sents_s1(df, sim_word_dict, p_dict, seed)   ← 抽样替换，依赖 seed
        ▼
privatized_dataset/glove_840B-300d/paper/
    eps_<ε>_top_200_s1_save_stop_words_False_seed_<S>/
        ├── train.tsv
        └── test.tsv
        │
        │  main.py --use_saved_private_data   ← BERT 微调
        ▼
experiment_results/eps_<ε>_topk_200_seed_<S>.log   下游任务准确率
```

三个层次的缓存粒度不同：

| 产物 | 依赖参数 | 重算成本 |
|---|---|---|
| `sim_word_dict` / `sim_dist_dict` | `embedding_type`, `mapping_strategy`, `top_k` | **慢**（O(V²)，需要遍历全词表 + 欧氏距离） |
| `p_dict` | + `eps` | 快（仅一个 `exp` + 归一化） |
| 脱敏 TSV | + `seed` | 中等（依赖语料长度） |

`top_200` 的最贵那一步（前两个文件）**已经做好**，所以现在跑任何 eps × seed 组合都很快。

---

## 三、当前 `top_k=200` 已就绪的产物

| 产物 | 路径 | 状态 |
|---|---|---|
| Top200 自定义输出集合 | `sim_word_dict/glove_840B-300d/paper/top_200.txt` | ✅ 已生成 (251 MB) |
| Top200 归一化相似度 | `sim_dist_dict/glove_840B-300d/paper/top_200.txt` | ✅ 已生成 (380 MB) |
| Top200 在 `eps=1.0` 下的采样概率分布 | `p_dict/glove_840B-300d/paper/eps_1.0_top_200.txt` | ✅ 已生成 (198 MB) |
| Top200 脱敏数据集 | `privatized_dataset/.../*_top_200_*/{train,test}.tsv` | ❌ 待生成 |

> 其它 `eps` 值（如 `0`, `2`, `4`, `18`, ...）的 `p_dict` 还没缓存，但生成时会自动按需计算并落盘。

---

## 四、操作流程（top_k=200 全链路）

### Step 1（可选）：补齐其它 eps 的 `p_dict`

如果只跑 `eps=1.0` 可以跳过本步。要预生成多组 eps 的 `p_dict`：

```bash
conda activate custext
python build_mapping_cache.py \
    --top_k 200 \
    --embedding_type glove_840B-300d \
    --mapping_strategy paper \
    --eps_list 0 1 2 4 6 8 10 12 14 16 18 20
```

会写入：

```
p_dict/glove_840B-300d/paper/eps_0_top_200.txt
p_dict/glove_840B-300d/paper/eps_1_top_200.txt
...
p_dict/glove_840B-300d/paper/eps_20_top_200.txt
```

### Step 2：生成 `top_k=200` 的脱敏数据集

`batch_generate_private_data.py` 一次性遍历所有 `(eps, seed)` 组合，复用已缓存的字典：

```bash
# 跑 5 个 seed + 单个 eps=1.0
python batch_generate_private_data.py \
    --top_k 200 \
    --embedding_type glove_840B-300d \
    --mapping_strategy paper \
    --privatization_strategy s1 \
    --dataset sst2 \
    --eps 1.0 \
    --seeds 42 43 44 45 46

# 跑多 eps × 多 seed
python batch_generate_private_data.py \
    --top_k 200 \
    --eps_list 0 1.0 2.0 4.0 18 \
    --seeds 42 43 44 45 46 47 48 49 50 51
```

输出（每个组合一个目录）：

```
privatized_dataset/glove_840B-300d/paper/
    eps_1.0_top_200_s1_save_stop_words_False_seed_42/
        ├── train.tsv
        └── test.tsv
    eps_1.0_top_200_s1_save_stop_words_False_seed_43/
        ...
```

### Step 3：BERT 微调（使用 top_200 脱敏数据）

修改 `run_experiments.sh` 顶部：

```bash
EPS_VALUES=(1.0)      # 或者 (0 1.0 2.0 4.0 18)
NUM_RUNS=10
TOP_K=200             # ★ 改成 200
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
```

然后跑：

```bash
bash run_experiments.sh 0   # GPU 0
```

它会自动带 `--use_saved_private_data` 从 Step 2 产物加载数据，并把每个 seed 的训练日志写到：

```
experiment_results/eps_<ε>_topk_200_seed_<S>.log
```

最后调用 `collect_results.py` 汇总成 csv。

#### 仅跑一次的单条命令

```bash
python main.py \
    --dataset sst2 \
    --eps 1.0 \
    --top_k 200 \
    --embedding_type glove_840B-300d \
    --mapping_strategy paper \
    --privatization_strategy s1 \
    --model_type /data/youyaru/SanText-main/bert-base-uncased \
    --epochs 3 \
    --use_cuda \
    --seed 42 \
    --use_saved_private_data
```

### Step 4：隐私攻击评估（top_k=200）

把对应脚本里的 `TOP_K=20` 改成 `TOP_K=200`：

```bash
# Mask Token Inference Attack（BERT MLM 还原被替换的词）
bash run_mask_token_attack.sh 0      # GPU 0

# KNN Attack（嵌入空间最近邻还原）
bash run_knn_attack.sh
```

结果分别在 `attack_results/` 和 `knn_attack_results/`。

### Step 5（可选）：Sample Amplification 混合实验

```bash
bash run_sample_amplification.sh
```

它会在 `eps_low` 和 `eps_high` 两个隐私预算之间按一定比例混合输出，对应到 `privatized_dataset_mixed/`。

---

## 五、关键参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--top_k` | `20` | **输出集合大小 K**，本目录主跑 `200` |
| `--eps` | `1.0` | 差分隐私预算 ε，越小越私 |
| `--embedding_type` | `ct_vectors` | 此处统一用 `glove_840B-300d`（论文设置） |
| `--mapping_strategy` | `paper` | 与论文 Algorithm 1 对齐的 K-to-K data-independent 映射 |
| `--privatization_strategy` | `s1` | 论文中的 S1 策略 |
| `--save_stop_words` | `False` | True 表示停用词不被替换 |
| `--seed` | `42` | 随机种子；同一 (eps, K) 下不同 seed 等价于不同的脱敏样本 |
| `--use_saved_private_data` | `False` | True 时从磁盘加载已生成的脱敏 TSV（推荐用于批量训练） |

---

## 六、常见问题

**Q1. 已经有 `eps_1.0_top_200.txt`，还能跑别的 eps 吗？**
能。`batch_generate_private_data.py` 检测到 `p_dict/.../eps_<ε>_top_200.txt` 不存在时，会调用 `compute_p_dict(sim_dist_dict, eps)` 现场算一份并落盘。`sim_dist_dict/top_200.txt` 已存在，所以这一步只需几秒到一分钟。

**Q2. 切换 `top_k` 需要重新做什么？**
切换 K（例如从 200 → 500）后，`sim_word_dict` / `sim_dist_dict` / `p_dict` **必须全部重算**（K 不同 → 候选集不同 → 相似度归一化也不同）。用 `build_mapping_cache.py --top_k 500 ...` 即可。

**Q3. 如何核对脱敏数据是否合理？**
打开任一 `privatized_dataset/.../eps_*_top_200_*/train.tsv`，对照原始 `datasets/sst2/train.tsv` 看同一行的句子被替换成了哪些词，再去 `sim_word_dict/.../top_200.txt` 里查询原词的候选集，正常情况下替换词应该都来自该候选集。

**Q4. 为什么 `eps=0` 也有意义？**
`eps=0` 时 `exp(0·u/2) = 1`，每个候选词等概率，相当于在 K=200 个最近邻里**均匀随机**采样，可作为完全无隐私偏好的极端 baseline。

---

## 七、最小复现命令清单（topk=200）

```bash
# 0. 激活环境
conda activate custext

# 1. 仅生成 top_200 脱敏数据 (10 seeds × eps=1.0)
python batch_generate_private_data.py \
    --top_k 200 --eps 1.0 \
    --seeds 42 43 44 45 46 47 48 49 50 51

# 2. 训练（先把 run_experiments.sh 里 TOP_K 改成 200, EPS_VALUES 改成 (1.0)）
bash run_experiments.sh 0

# 3. （可选）跑攻击
bash run_mask_token_attack.sh 0
bash run_knn_attack.sh
```

