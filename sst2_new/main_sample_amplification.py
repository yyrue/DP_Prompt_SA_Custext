"""
Sample Amplification 下游任务训练脚本

直接加载预生成的混合数据（由 generate_sample_amplification.py 生成），
跳过 CusText 的扰动步骤，只做 BERT 微调 + 评估。

用法:
    python main_sample_amplification.py \
        --eps 16 --eps_low 0.0 --eps_high 32.0 \
        --model_type /data/youyaru/SanText-main/bert-base-uncased \
        --use_cuda --seed 42
"""

import datetime
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
)

# import 时 utils.py / logger.py 会解析 args，这里兼容 args.py 的参数
from utils import Bert_dataset
from logger import get_logger
from training import Trainer
from args import get_parser


def build_mixed_data_path(base_dir, eps_low, eps_high, eps_target,
                           top_k, strategy, save_stop_words, seed):
    """构建混合数据集目录路径（与 generate_sample_amplification.py 的输出路径一致）。"""
    eps_str = str(int(eps_target)) if eps_target == int(eps_target) else str(eps_target)
    folder = (f"eps_{eps_str}_top_{top_k}_{strategy}"
              f"_save_stop_words_{save_stop_words}_seed_{seed}")
    return os.path.join(base_dir, f"mix_{eps_low}_{eps_high}", folder)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # 日志保存到区分 mix 来源的子目录
    log_subdir = os.path.join("log", f"mix_{args.eps_low}_{args.eps_high}")
    os.makedirs(log_subdir, exist_ok=True)
    log_file = (f"SA_mix_{args.eps_low}_{args.eps_high}_eps_{args.eps}"
                f"_seed_{args.seed}"
                f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt")
    logger = get_logger(log_path=log_subdir, log_file=log_file)
    logger.info(f"Sample Amplification experiment, args: {args}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # ---- 加载混合数据 ----
    mixed_dir = build_mixed_data_path(
        args.mixed_data_dir, args.eps_low, args.eps_high, args.eps,
        args.top_k, args.privatization_strategy, args.save_stop_words,
        args.seed,
    )
    train_file = os.path.join(mixed_dir, "train.tsv")
    test_file = os.path.join(mixed_dir, "test.tsv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"混合数据不存在: {mixed_dir}\n"
            f"请先运行 generate_sample_amplification.py 生成数据。"
        )

    train_data = pd.read_csv(train_file, sep="\t", keep_default_na=False).reset_index(drop=True)
    test_data = pd.read_csv(test_file, sep="\t", keep_default_na=False).reset_index(drop=True)
    # 填充空值：tsv 中空行会被 pandas 解析为 NaN，tokenizer 无法处理
    train_data['sentence'] = train_data['sentence'].fillna('')
    test_data['sentence'] = test_data['sentence'].fillna('')

    # dev 数据用原始的（未扰动），与原始实验一致
    dev_data = pd.read_csv(f"datasets/{args.dataset}/dev.tsv", sep="\t").reset_index(drop=True)
    dev_data['sentence'] = dev_data['sentence'].fillna('')

    logger.info(f"混合数据目录: {mixed_dir}")
    logger.info(f"train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}")

    # ---- 构建 DataLoader ----
    train_loader = DataLoader(Bert_dataset(train_data),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(Bert_dataset(dev_data),
                            batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Bert_dataset(test_data),
                             batch_size=args.batch_size, shuffle=True)

    # ---- 模型 ----
    model = BertForSequenceClassification.from_pretrained(
        args.model_type,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.epochs,
    )

    trainer = Trainer(
        model, scheduler, optimizer,
        args.epochs, args.log_steps, args.eval_steps,
        args.use_cuda, logger,
    )

    # ---- 训练 + 评估 ----
    trainer.train(train_loader, test_loader)
    acc = trainer.predict(test_loader)
    logger.info(f"test acc = {acc:.4f}.")
