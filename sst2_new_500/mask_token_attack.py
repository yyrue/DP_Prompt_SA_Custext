"""
Mask Token Inference Attack for CusText

核心思路：
  1. 对文本的每个词位置，将该词替换为 MASK oreexchangeRate》，其余位置保持不变
  2. 用 BertForMaskedLM 预测 MASK 位置最可能的词
  3. 将预测词与原始文本对应位置的词比较，计算预测准确率 (Attack Accuracy)
  4. 分别对 原始文本(baseline) 和 脱敏文本(privatized) 执行，得到两个准确率

用法示例：
  # 对原始数据执行 attack（得到 baseline accuracy）
  python mask_token_attack.py --attack_original --seed 42

  # 对脱敏数据执行 attack（得到 privatized accuracy）
  python mask_token_attack.py --eps 1.0 --top_k 20 --seed 42 --embedding_type glove_840B-300d --mapping_strategy paper

  # 批量对多个 eps 执行 attack
  python mask_token_attack.py --eps_list 0.5 1.0 2.0 4.0 8.0 --top_k 20 --seed 42
"""

import os
import copy
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler


def get_attack_parser():
    parser = argparse.ArgumentParser(description="Mask Token Inference Attack for CusText")

    # --- 数据与路径参数 ---
    parser.add_argument("--dataset", type=str, default="sst2",
                        help="数据集名称，对应 datasets/ 下的子目录")
    parser.add_argument("--model_path", type=str, default="/data/youyaru/SanText-main/bert-base-uncased",
                        help="用于 MLM 推理的 BERT 模型路径")
    parser.add_argument("--output_dir", type=str, default="./attack_results",
                        help="攻击结果输出目录")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="BERT 输出的最大序列长度")

    # --- CusText 脱敏参数（用于定位脱敏数据路径） ---
    parser.add_argument("--eps", type=float, default=1.0,
                        help="隐私预算 epsilon")
    parser.add_argument("--top_k", type=int, default=20,
                        help="CusText 映射组大小 K")
    parser.add_argument("--embedding_type", type=str, default="ct_vectors",
                        help="词嵌入类型 (ct_vectors / glove_840B-300d)")
    parser.add_argument("--mapping_strategy", type=str, default="paper",
                        help="映射策略 (paper / conservative / aggressive)")
    parser.add_argument("--privatization_strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", action="store_true", default=False,
                        help="是否保留停用词不替换（需与脱敏时一致）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（需与脱敏时一致）")

    # --- Attack 参数 ---
    parser.add_argument("--attack_original", action="store_true", default=False,
                        help="对原始未脱敏数据执行 attack（得到 baseline accuracy）")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="MLM 推理的 batch size")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="最多处理的 token 数量（0=不限制，用于调试）")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="不使用 GPU")

    # --- 批量参数 ---
    parser.add_argument("--eps_list", type=float, nargs="+", default=None,
                        help="批量 eps 列表，不指定则只跑 --eps 的值")

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_original_data(dataset):
    """加载原始数据集"""
    train_df = pd.read_csv(f"datasets/{dataset}/train.tsv", sep='\t', keep_default_na=False)
    dev_df = pd.read_csv(f"datasets/{dataset}/dev.tsv", sep='\t', keep_default_na=False)
    test_df = pd.read_csv(f"datasets/{dataset}/test.tsv", sep='\t', keep_default_na=False)
    return train_df, dev_df, test_df


def load_privatized_data(args):
    """加载 CusText 已脱敏的数据集"""
    priv_dir = (
        f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
        f"/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{args.save_stop_words}_seed_{args.seed}"
    )
    test_path = os.path.join(priv_dir, "test.tsv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"未找到脱敏数据集，请先运行批量生成脚本：\n"
            f"  python batch_generate_private_data.py --eps {args.eps} --seed {args.seed}\n"
            f"  期望路径: {test_path}"
        )

    print(f"加载脱敏数据集: {priv_dir}")
    test_data = pd.read_csv(test_path, sep='\t', keep_default_na=False).reset_index(drop=True)
    test_data['sentence'] = test_data['sentence'].fillna('')
    return test_data


def run_mask_token_attack(original_sentences, masked_sentences, tokenizer, mlm_model, device,
                          max_seq_length=128, batch_size=256, max_tokens=0):
    """
    对文本执行 Mask Token Inference Attack。

    对 masked_sentences 中的每个词位置：
      1. 将该位置替换为 MASK
      2. 用 BertForMaskedLM 预测 MASK 位置的词
      3. 比较预测词与 original_sentences 对应位置的词
      4. 一致则攻击成功

    Args:
        original_sentences: List[str], 原始文本（ground truth）
        masked_sentences: List[str], 被mask的文本（脱敏文本 或 原始文本自身用于baseline）
        tokenizer: BertTokenizer
        mlm_model: BertForMaskedLM
        device: torch.device
        max_seq_length: BERT 最大序列长度
        batch_size: 推理 batch size
        max_tokens: 最多处理的 token 数（0=不限制）

    Returns:
        dict: 包含 attack_accuracy 等统计
    """
    # 将句子拆分为词列表
    original_docs = [s.split() for s in original_sentences]
    masked_docs = [s.split() for s in masked_sentences]

    # 构造 MLM 输入：对每个词位置生成一条 MASK 样本
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    all_mask_positions = []
    all_labels = []

    total_tokens = 0
    unmatched_count = 0

    for i in tqdm(range(len(masked_docs)), desc="构建 MLM 输入"):
        orig_doc = original_docs[i]
        mask_doc = masked_docs[i]

        if len(orig_doc) != len(mask_doc):
            unmatched_count += 1
            continue

        for j in range(len(mask_doc)):
            orig_word = orig_doc[j]

            # 将 mask_doc 位置 j 替换为 MASK
            tmp_doc = copy.deepcopy(mask_doc)
            tmp_doc[j] = tokenizer.mask_token

            masked_text = " ".join(tmp_doc)
            encoded = tokenizer.encode_plus(
                masked_text,
                add_special_tokens=True,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt"
            )

            # 找到 MASK token 在 input_ids 中的位置
            input_ids = encoded['input_ids'][0]
            mask_token_id = tokenizer.mask_token_id
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                continue

            mask_pos = mask_positions[0].item()
            orig_token_ids = tokenizer.encode(orig_word, add_special_tokens=False)

            # 标签为原始词的第一个 subword token id
            label = orig_token_ids[0] if len(orig_token_ids) > 0 else -1

            all_input_ids.append(encoded['input_ids'])
            all_token_type_ids.append(encoded['token_type_ids'])
            all_attention_masks.append(encoded['attention_mask'])
            all_mask_positions.append(mask_pos)
            all_labels.append(label)

            total_tokens += 1

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        if max_tokens > 0 and total_tokens >= max_tokens:
            break

    if total_tokens == 0:
        print("没有有效的 token 可供攻击！")
        return None

    print(f"总 token 数: {total_tokens}, 长度不匹配句子数: {unmatched_count}")

    # 构造 TensorDataset
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_token_type_ids = torch.cat(all_token_type_ids, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    all_mask_positions = torch.tensor(all_mask_positions, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_masks,
                            all_mask_positions, all_labels)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    # MLM 推理
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

            outputs = mlm_model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            logits = outputs.logits

            # 取 MASK 位置的预测
            batch_size_cur = logits.size(0)
            mask_logits = logits[torch.arange(batch_size_cur, device=device), mask_positions]
            predictions = torch.argmax(mask_logits, dim=-1)

            # 统计正确预测（过滤 label=-1 的无效样本）
            valid_mask = (labels != -1)
            correct_total += ((predictions == labels) & valid_mask).sum().item()
            valid_total += valid_mask.sum().item()

    attack_accuracy = correct_total / valid_total if valid_total > 0 else 0

    results = {
        'total_tokens': total_tokens,
        'valid_tokens': valid_total,
        'correct': correct_total,
        'attack_accuracy': attack_accuracy,
        'unmatched_sentences': unmatched_count,
    }

    return results


def run_attack_single(args, tokenizer, mlm_model, device):
    """对单组参数执行 mask token attack"""
    set_seed(args.seed)

    # 加载原始数据（ground truth）
    _, _, test_data_orig = load_original_data(args.dataset)
    original_sentences = test_data_orig['sentence'].tolist()

    if args.attack_original:
        # baseline：对原始文本自身做 mask token attack
        masked_sentences = original_sentences
        print("\n=== 对原始数据执行 Mask Token Attack (Baseline) ===")
    else:
        # 加载脱敏数据
        priv_test = load_privatized_data(args)
        masked_sentences = priv_test['sentence'].tolist()

    assert len(original_sentences) == len(masked_sentences), \
        f"原始数据({len(original_sentences)})和脱敏数据({len(masked_sentences)})行数不一致！"

    results = run_mask_token_attack(
        original_sentences=original_sentences,
        masked_sentences=masked_sentences,
        tokenizer=tokenizer,
        mlm_model=mlm_model,
        device=device,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    return results


def print_results(results, eps=None, seed=None, is_baseline=False):
    """打印攻击结果"""
    tag = "Baseline (原始数据)" if is_baseline else f"Privatized (eps={eps}, seed={seed})"
    print(f"\n{'='*50}")
    print(f"  Mask Token Attack: {tag}")
    print(f"{'='*50}")
    print(f"  总 token 数:        {results['total_tokens']}")
    print(f"  有效 token 数:      {results['valid_tokens']}")
    print(f"  预测正确数:         {results['correct']}")
    print(f"  Attack Accuracy:    {results['attack_accuracy']:.4f}")
    print(f"{'='*50}")


def save_results(results, args, eps=None):
    """保存攻击结果到文件"""
    os.makedirs(args.output_dir, exist_ok=True)

    if args.attack_original:
        result_file = os.path.join(
            args.output_dir,
            f"attack_baseline_seed_{args.seed}.txt"
        )
    else:
        if eps is None:
            eps = args.eps
        result_file = os.path.join(
            args.output_dir,
            f"attack_{args.embedding_type}_{args.mapping_strategy}"
            f"_eps_{eps}_top_{args.top_k}_seed_{args.seed}"
            f"_save_stop_words_{args.save_stop_words}.txt"
        )

    with open(result_file, 'w') as f:
        f.write(f"Mask Token Inference Attack Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters:\n")
        if args.attack_original:
            f.write(f"  mode=baseline (original data)\n")
            f.write(f"  seed={args.seed}\n")
        else:
            f.write(f"  eps={eps}, top_k={args.top_k}, seed={args.seed}\n")
            f.write(f"  embedding_type={args.embedding_type}\n")
            f.write(f"  mapping_strategy={args.mapping_strategy}\n")
            f.write(f"  privatization_strategy={args.privatization_strategy}\n")
            f.write(f"  save_stop_words={args.save_stop_words}\n")
        f.write(f"  model_path={args.model_path}\n")
        f.write(f"  batch_size={args.batch_size}\n")
        f.write(f"  max_seq_length={args.max_seq_length}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Results:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")

    print(f"结果已保存到: {result_file}")
    return result_file


def main():
    parser = get_attack_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")

    # 加载 MLM 模型
    print(f"加载 BertForMaskedLM: {args.model_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    mlm_model = BertForMaskedLM.from_pretrained(args.model_path)
    mlm_model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        mlm_model = torch.nn.DataParallel(mlm_model)

    # 确定要跑的 eps 列表
    eps_list = args.eps_list if args.eps_list else [args.eps]

    all_results = []

    if args.attack_original:
        # Baseline 模式：与 eps 无关，直接运行
        print(f"\n{'#'*60}")
        print(f"# Baseline (原始数据)")
        print(f"{'#'*60}")

        results = run_attack_single(args, tokenizer, mlm_model, device)
        if results is not None:
            print_results(results, is_baseline=True)
            save_results(results, args, eps=None)
            results['seed'] = args.seed
            all_results.append(results)
    else:
        # 脱敏数据模式：遍历 eps 列表
        for eps in eps_list:
            args.eps = eps
            print(f"\n{'#'*60}")
            print(f"# eps = {eps}")
            print(f"{'#'*60}")

            results = run_attack_single(args, tokenizer, mlm_model, device)
            if results is not None:
                print_results(results, eps=eps, seed=args.seed, is_baseline=False)
                save_results(results, args, eps=eps)
                results['eps'] = eps
                results['seed'] = args.seed
                all_results.append(results)

    # 汇总输出
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  汇总：不同 eps 下的 Attack Accuracy")
        print(f"{'='*60}")
        print(f"  {'eps':<10} {'attack_accuracy':<20} {'correct/total':<20}")
        print(f"  {'-'*10} {'-'*20} {'-'*20}")
        for r in all_results:
            print(f"  {r['eps']:<10} {r['attack_accuracy']:<20.4f} {r['correct']}/{r['valid_tokens']}")

        # 保存汇总 CSV
        os.makedirs(args.output_dir, exist_ok=True)
        summary_file = os.path.join(
            args.output_dir,
            f"attack_summary_{args.embedding_type}_{args.mapping_strategy}"
            f"_top_{args.top_k}_seed_{args.seed}.csv"
        )
        df = pd.DataFrame(all_results)
        df.to_csv(summary_file, index=False)
        print(f"\n汇总结果已保存到: {summary_file}")

        print(f"\n  提示: 请同时运行 baseline 获取原始数据的准确率作为对照:")
        print(f"    python mask_token_attack.py --attack_original --seed {args.seed}")


if __name__ == "__main__":
    main()