"""
Sample Amplification 下游训练脚本（QNLI 版）。

直接加载预生成的 mixed private 数据（由 generate_sample_amplification.py 生成），
跳过 CusText 扰动步骤，只做 BERT 微调 + 评估。
"""

import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup

from args import get_parser
from logger import get_logger
from training import Trainer
from utils import Bert_dataset


def _format_eps_for_folder(eps):
    if abs(eps - round(eps)) < 1e-9:
        return str(int(round(eps)))
    return str(eps).replace(".", "p")


def build_mixed_data_path(
    base_dir, eps_low, eps_high, eps_target, top_k, strategy, save_stop_words, seed
):
    """构建 mixed 数据目录路径（与 generate_sample_amplification.py 输出一致）。"""
    eps_str = _format_eps_for_folder(eps_target)
    folder = (
        f"eps_{eps_str}_top_{top_k}_{strategy}"
        f"_save_stop_words_{save_stop_words}_seed_{seed}"
    )
    return os.path.join(base_dir, f"mix_{eps_low}_{eps_high}", folder)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log_subdir = os.path.join("log", f"mix_{args.eps_low}_{args.eps_high}")
    os.makedirs(log_subdir, exist_ok=True)
    log_file = (
        f"SA_mix_{args.eps_low}_{args.eps_high}_eps_{args.eps}_seed_{args.seed}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"
    )
    logger = get_logger(log_path=log_subdir, log_file=log_file)
    logger.info(f"QNLI Sample Amplification experiment, args: {args}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mixed_dir = build_mixed_data_path(
        args.mixed_data_dir,
        args.eps_low,
        args.eps_high,
        args.eps,
        args.top_k,
        args.privatization_strategy,
        args.save_stop_words,
        args.seed,
    )
    train_file = os.path.join(mixed_dir, "train.tsv")
    test_file = os.path.join(mixed_dir, "test.tsv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"混合数据不存在: {mixed_dir}\n"
            "请先运行 generate_sample_amplification.py 生成数据。"
        )

    train_data = pd.read_csv(train_file, sep="\t", keep_default_na=False).reset_index(drop=True)
    test_data = pd.read_csv(test_file, sep="\t", keep_default_na=False).reset_index(drop=True)

    for df in (train_data, test_data):
        df["sentence"] = df["sentence"].fillna("")
        df["question"] = df["question"].fillna("")

    logger.info(f"混合数据目录: {mixed_dir}")
    logger.info(f"train: {len(train_data)}, test: {len(test_data)}")

    train_loader = DataLoader(Bert_dataset(train_data), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Bert_dataset(test_data), batch_size=args.batch_size, shuffle=True)

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
        model,
        scheduler,
        optimizer,
        args.epochs,
        args.log_steps,
        args.eval_steps,
        args.use_cuda,
        logger,
    )

    trainer.train(train_loader, test_loader)
    acc = trainer.predict(test_loader)
    logger.info(f"test acc = {acc:.4f}.")
