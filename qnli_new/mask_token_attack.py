"""
Mask Token Inference Attack for QNLI (CusText).

QNLI 为双句输入，攻击时会分别在 `question` 与 `sentence` 两列逐 token 打 mask，
然后用 BertForMaskedLM 预测该位置 token，并与原始文本对应 token 比较。
"""

import argparse
import copy
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def get_attack_parser():
    parser = argparse.ArgumentParser(description="Mask Token Inference Attack for QNLI")
    parser.add_argument("--dataset", type=str, default="qnli")
    parser.add_argument("--model_path", type=str, default="/data/youyaru/SanText-main/bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./attack_results")
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    parser.add_argument("--mapping_strategy", type=str, default="paper")
    parser.add_argument("--privatization_strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--attack_original", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--eps_list", type=float, nargs="+", default=None)
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_original_data(dataset):
    test_df = pd.read_csv(f"datasets/{dataset}/test.tsv", sep="\t", keep_default_na=False)
    test_df["question"] = test_df["question"].fillna("")
    test_df["sentence"] = test_df["sentence"].fillna("")
    return test_df


def load_privatized_data(args):
    priv_dir = (
        f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
        f"/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{args.save_stop_words}_seed_{args.seed}"
    )
    test_path = os.path.join(priv_dir, "test.tsv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"未找到脱敏数据集: {test_path}\n"
            f"请先生成对应 eps={args.eps}, seed={args.seed} 的 QNLI 脱敏数据。"
        )
    test_df = pd.read_csv(test_path, sep="\t", keep_default_na=False).reset_index(drop=True)
    test_df["question"] = test_df["question"].fillna("")
    test_df["sentence"] = test_df["sentence"].fillna("")
    return test_df


def _encode_pair(tokenizer, question, sentence, max_seq_length):
    encoded = tokenizer.encode_plus(
        question,
        sentence,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    return encoded


def run_mask_token_attack(
    original_df,
    masked_df,
    tokenizer,
    mlm_model,
    device,
    max_seq_length=128,
    batch_size=256,
    max_tokens=0,
):
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    all_mask_positions = []
    all_labels = []

    total_tokens = 0
    unmatched_rows = 0
    skipped_mask_not_found = 0

    for i in tqdm(range(len(masked_df)), desc="构建 MLM 输入"):
        oq = str(original_df.iloc[i]["question"]).split()
        osen = str(original_df.iloc[i]["sentence"]).split()
        mq = str(masked_df.iloc[i]["question"]).split()
        msen = str(masked_df.iloc[i]["sentence"]).split()

        if len(oq) != len(mq) or len(osen) != len(msen):
            unmatched_rows += 1
            continue

        # 在 question 上逐 token mask
        for j in range(len(mq)):
            tmp_q = copy.deepcopy(mq)
            tmp_q[j] = tokenizer.mask_token
            encoded = _encode_pair(tokenizer, " ".join(tmp_q), " ".join(msen), max_seq_length)
            input_ids = encoded["input_ids"][0]
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                skipped_mask_not_found += 1
                continue
            label_ids = tokenizer.encode(oq[j], add_special_tokens=False)
            label = label_ids[0] if len(label_ids) > 0 else -1
            all_input_ids.append(encoded["input_ids"])
            all_token_type_ids.append(encoded["token_type_ids"])
            all_attention_masks.append(encoded["attention_mask"])
            all_mask_positions.append(mask_positions[0].item())
            all_labels.append(label)
            total_tokens += 1
            if max_tokens > 0 and total_tokens >= max_tokens:
                break
        if max_tokens > 0 and total_tokens >= max_tokens:
            break

        # 在 sentence 上逐 token mask
        for j in range(len(msen)):
            tmp_s = copy.deepcopy(msen)
            tmp_s[j] = tokenizer.mask_token
            encoded = _encode_pair(tokenizer, " ".join(mq), " ".join(tmp_s), max_seq_length)
            input_ids = encoded["input_ids"][0]
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                skipped_mask_not_found += 1
                continue
            label_ids = tokenizer.encode(osen[j], add_special_tokens=False)
            label = label_ids[0] if len(label_ids) > 0 else -1
            all_input_ids.append(encoded["input_ids"])
            all_token_type_ids.append(encoded["token_type_ids"])
            all_attention_masks.append(encoded["attention_mask"])
            all_mask_positions.append(mask_positions[0].item())
            all_labels.append(label)
            total_tokens += 1
            if max_tokens > 0 and total_tokens >= max_tokens:
                break
        if max_tokens > 0 and total_tokens >= max_tokens:
            break

    if total_tokens == 0:
        print("没有有效 token 可供攻击。")
        return None

    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_token_type_ids = torch.cat(all_token_type_ids, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    all_mask_positions = torch.tensor(all_mask_positions, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_token_type_ids, all_attention_masks, all_mask_positions, all_labels
    )
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    mlm_model.eval()
    correct_total = 0
    valid_total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MLM 推理"):
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            mask_positions = batch[3].to(device)
            labels = batch[4].to(device)

            outputs = mlm_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            logits = outputs.logits
            batch_size_cur = logits.size(0)
            mask_logits = logits[torch.arange(batch_size_cur, device=device), mask_positions]
            predictions = torch.argmax(mask_logits, dim=-1)

            valid_mask = labels != -1
            correct_total += ((predictions == labels) & valid_mask).sum().item()
            valid_total += valid_mask.sum().item()

    return {
        "total_tokens": total_tokens,
        "valid_tokens": valid_total,
        "correct": correct_total,
        "attack_accuracy": (correct_total / valid_total) if valid_total > 0 else 0.0,
        "unmatched_rows": unmatched_rows,
        "skipped_mask_not_found": skipped_mask_not_found,
    }


def print_results(results, eps=None, seed=None, is_baseline=False):
    tag = "Baseline (原始数据)" if is_baseline else f"Privatized (eps={eps}, seed={seed})"
    print(f"\n{'=' * 50}")
    print(f"Mask Token Attack: {tag}")
    print(f"{'=' * 50}")
    print(f"总 token 数:      {results['total_tokens']}")
    print(f"有效 token 数:    {results['valid_tokens']}")
    print(f"预测正确数:       {results['correct']}")
    print(f"Attack Accuracy:  {results['attack_accuracy']:.4f}")
    print(f"{'=' * 50}")


def save_results(results, args, eps=None):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.attack_original:
        result_file = os.path.join(args.output_dir, f"attack_baseline_seed_{args.seed}.txt")
    else:
        cur_eps = args.eps if eps is None else eps
        result_file = os.path.join(
            args.output_dir,
            f"attack_{args.embedding_type}_{args.mapping_strategy}"
            f"_eps_{cur_eps}_top_{args.top_k}_seed_{args.seed}"
            f"_save_stop_words_{args.save_stop_words}.txt",
        )

    with open(result_file, "w") as f:
        f.write("Mask Token Inference Attack Results (QNLI)\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.attack_original:
            f.write("mode=baseline\n")
        else:
            f.write(f"eps={eps}\n")
            f.write(f"top_k={args.top_k}\n")
            f.write(f"seed={args.seed}\n")
        f.write(f"model_path={args.model_path}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"max_seq_length={args.max_seq_length}\n")
        f.write(f"{'=' * 50}\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"结果已保存到: {result_file}")


def run_attack_single(args, tokenizer, mlm_model, device):
    set_seed(args.seed)
    original_df = load_original_data(args.dataset)
    if args.attack_original:
        masked_df = original_df
    else:
        masked_df = load_privatized_data(args)

    if len(original_df) != len(masked_df):
        raise ValueError(f"原始与脱敏行数不一致: {len(original_df)} vs {len(masked_df)}")

    return run_mask_token_attack(
        original_df=original_df,
        masked_df=masked_df,
        tokenizer=tokenizer,
        mlm_model=mlm_model,
        device=device,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )


def main():
    args = get_attack_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    mlm_model = BertForMaskedLM.from_pretrained(args.model_path).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        mlm_model = torch.nn.DataParallel(mlm_model)

    eps_list = args.eps_list if args.eps_list else [args.eps]
    all_results = []

    if args.attack_original:
        results = run_attack_single(args, tokenizer, mlm_model, device)
        if results is not None:
            print_results(results, is_baseline=True)
            save_results(results, args, eps=None)
            results["seed"] = args.seed
            all_results.append(results)
    else:
        for eps in eps_list:
            args.eps = eps
            print(f"\n{'#' * 60}\n# eps={eps}\n{'#' * 60}")
            results = run_attack_single(args, tokenizer, mlm_model, device)
            if results is not None:
                print_results(results, eps=eps, seed=args.seed, is_baseline=False)
                save_results(results, args, eps=eps)
                results["eps"] = eps
                results["seed"] = args.seed
                all_results.append(results)

    if len(all_results) > 1 and not args.attack_original:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_file = os.path.join(
            args.output_dir,
            f"attack_summary_{args.embedding_type}_{args.mapping_strategy}"
            f"_top_{args.top_k}_seed_{args.seed}.csv",
        )
        pd.DataFrame(all_results).to_csv(summary_file, index=False)
        print(f"汇总结果已保存: {summary_file}")


if __name__ == "__main__":
    main()
