"""
Mask Token Inference Attack — 混合脱敏数据 (privatized_dataset_mixed)

与 mask_token_attack.py 相同攻击流程，但数据来自 Sample Amplification 混合输出：
  privatized_dataset_mixed/{embedding_type}/{mapping_strategy}/
    mix_{eps_low}_{eps_high}/
      eps_{eps_target}_top_{K}_{strategy}_save_stop_words_{bool}_seed_{seed}/test.tsv

其中 eps_target 为混合后的目标隐私预算 ε'（与 generate_sample_amplification.py 一致），
不是单一路径 privatized_dataset 下的单一 CusText eps。

用法示例：
  # 对 mix_0.0_14.0 下某一 ε' 跑 attack
  python mask_token_attack_mixed.py --eps_low 0.0 --eps_high 14.0 --eps 8 --seed 51

  # 单个 ε' 也会写入该 seed 的 attack_summary_mixed_*_seed_*.csv；分多次跑同一 seed 会按 eps_prime 合并去重

  # Baseline（原始 test，与原版脚本一致）
  python mask_token_attack_mixed.py --attack_original --seed 51
"""

import os
import argparse
import datetime
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM

from mask_token_attack import (
    set_seed,
    load_original_data,
    run_mask_token_attack,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Mask Token Attack on mixed (sample-amplification) privatized data"
    )
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--model_path", type=str, default="/data/youyaru/SanText-main/bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./attack_results_mixed",
                        help="混合实验结果输出目录（默认与单 eps 实验分开）")
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument(
        "--privatized_root",
        type=str,
        default="./privatized_dataset_mixed",
        help="混合脱敏数据根目录",
    )
    parser.add_argument("--eps_low", type=float, default=0.0,
                        help="混合区间下界（对应 mix_{eps_low}_{eps_high}）")
    parser.add_argument("--eps_high", type=float, default=14.0,
                        help="混合区间上界")
    parser.add_argument(
        "--eps",
        type=float,
        default=8.0,
        help="目标隐私预算 ε'（对应子目录名中的 eps_*，与 generate_sample_amplification 一致）",
    )
    parser.add_argument("--eps_list", type=float, nargs="+", default=None,
                        help="批量 ε' 列表；指定时忽略 --eps")

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

    return parser


def _mix_subdir(eps_low, eps_high):
    # 与 generate_sample_amplification 中 f"mix_{args.eps_low}_{args.eps_high}" 一致
    return f"mix_{float(eps_low)}_{float(eps_high)}"


def load_privatized_mixed(args, eps_target):
    """
    加载某一 ε' 下的混合脱敏 test 集。
    目录命名与 generate_sample_amplification.build_data_path 输出一致：
      eps_{int(eps_target)}_...  （脚本里 eps_target 为 range 的整数）
    """
    mix_dir = _mix_subdir(args.eps_low, args.eps_high)
    # 与生成脚本一致：文件夹名为 eps_{eps_target}，其中 eps_target 为整数步进
    eps_folder = int(eps_target)
    priv_dir = os.path.join(
        args.privatized_root,
        args.embedding_type,
        args.mapping_strategy,
        mix_dir,
        f"eps_{eps_folder}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{args.save_stop_words}_seed_{args.seed}",
    )
    test_path = os.path.join(priv_dir, "test.tsv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"未找到混合脱敏数据: {test_path}\n"
            f"请确认已运行 generate_sample_amplification.py，且 eps_low/eps_high/seed 与生成时一致。"
        )
    print(f"加载混合脱敏数据集: {priv_dir}")
    test_data = pd.read_csv(test_path, sep="\t", keep_default_na=False).reset_index(drop=True)
    test_data["sentence"] = test_data["sentence"].fillna("")
    return test_data


def run_attack_single_mixed(args, tokenizer, mlm_model, device, eps_target):
    set_seed(args.seed)
    _, _, test_data_orig = load_original_data(args.dataset)
    original_sentences = test_data_orig["sentence"].tolist()

    if args.attack_original:
        masked_sentences = original_sentences
        print("\n=== 对原始数据执行 Mask Token Attack (Baseline) ===")
    else:
        priv_test = load_privatized_mixed(args, eps_target)
        masked_sentences = priv_test["sentence"].tolist()

    assert len(original_sentences) == len(masked_sentences), (
        f"原始({len(original_sentences)})与脱敏({len(masked_sentences)})行数不一致"
    )

    return run_mask_token_attack(
        original_sentences=original_sentences,
        masked_sentences=masked_sentences,
        tokenizer=tokenizer,
        mlm_model=mlm_model,
        device=device,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )


def print_results(results, eps_target=None, seed=None, is_baseline=False, mix_tag=None):
    if is_baseline:
        tag = "Baseline (原始数据)"
    else:
        tag = f"Mixed privatized (mix={mix_tag}, eps'={eps_target}, seed={seed})"
    print(f"\n{'='*50}")
    print(f"  Mask Token Attack: {tag}")
    print(f"{'='*50}")
    print(f"  总 token 数:        {results['total_tokens']}")
    print(f"  有效 token 数:      {results['valid_tokens']}")
    print(f"  预测正确数:         {results['correct']}")
    print(f"  Attack Accuracy:    {results['attack_accuracy']:.4f}")
    print(f"{'='*50}")


def save_results_mixed(results, args, eps_target=None):
    os.makedirs(args.output_dir, exist_ok=True)
    mix_tag = _mix_subdir(args.eps_low, args.eps_high)
    # 文件名中避免裸点号用 d 代替小数点（语义仍可读）
    mix_slug = mix_tag.replace(".", "d")

    if args.attack_original:
        result_file = os.path.join(
            args.output_dir,
            f"attack_mixed_baseline_seed_{args.seed}.txt",
        )
    else:
        et = int(eps_target) if eps_target is not None else int(args.eps)
        result_file = os.path.join(
            args.output_dir,
            f"attack_mixed_{args.embedding_type}_{args.mapping_strategy}_{mix_slug}"
            f"_epsprime_{et}_top_{args.top_k}_seed_{args.seed}"
            f"_save_stop_words_{args.save_stop_words}.txt",
        )

    with open(result_file, "w") as f:
        f.write("Mask Token Inference Attack Results (mixed privatized data)\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Parameters:\n")
        if args.attack_original:
            f.write("  mode=baseline (original data)\n")
            f.write(f"  seed={args.seed}\n")
        else:
            f.write(f"  mix_eps_low={args.eps_low}, mix_eps_high={args.eps_high}\n")
            f.write(f"  eps_prime (target)={eps_target}\n")
            f.write(f"  top_k={args.top_k}, seed={args.seed}\n")
            f.write(f"  embedding_type={args.embedding_type}\n")
            f.write(f"  mapping_strategy={args.mapping_strategy}\n")
            f.write(f"  privatization_strategy={args.privatization_strategy}\n")
            f.write(f"  save_stop_words={args.save_stop_words}\n")
            f.write(f"  privatized_root={args.privatized_root}\n")
        f.write(f"  model_path={args.model_path}\n")
        f.write(f"  batch_size={args.batch_size}\n")
        f.write(f"  max_seq_length={args.max_seq_length}\n")
        f.write(f"{'='*50}\n\n")
        f.write("Results:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")

    print(f"结果已保存到: {result_file}")
    return result_file


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")

    print(f"加载 BertForMaskedLM: {args.model_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    mlm_model = BertForMaskedLM.from_pretrained(args.model_path)
    mlm_model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        mlm_model = torch.nn.DataParallel(mlm_model)

    mix_tag = _mix_subdir(args.eps_low, args.eps_high)
    mix_slug = mix_tag.replace(".", "d")
    eps_targets = args.eps_list if args.eps_list is not None else [args.eps]

    all_results = []

    if args.attack_original:
        print(f"\n{'#'*60}\n# Baseline (原始数据)\n{'#'*60}")
        results = run_attack_single_mixed(args, tokenizer, mlm_model, device, eps_target=0)
        if results is not None:
            print_results(results, is_baseline=True)
            save_results_mixed(results, args, eps_target=None)
            results["seed"] = args.seed
            results["mix_tag"] = mix_tag
            all_results.append(results)
    else:
        for eps_target in eps_targets:
            print(f"\n{'#'*60}\n# mix={mix_tag}, eps' = {eps_target}\n{'#'*60}")
            results = run_attack_single_mixed(args, tokenizer, mlm_model, device, eps_target)
            if results is not None:
                print_results(
                    results,
                    eps_target=eps_target,
                    seed=args.seed,
                    is_baseline=False,
                    mix_tag=mix_tag,
                )
                save_results_mixed(results, args, eps_target=eps_target)
                results["eps_prime"] = float(eps_target)
                results["eps_low"] = args.eps_low
                results["eps_high"] = args.eps_high
                results["seed"] = args.seed
                results["mix_tag"] = mix_tag
                all_results.append(results)

    # 每个 seed 都写出 attack_summary_mixed_*_seed_*.csv，便于 aggregate_mixed_attack_by_eps_prime.py 汇总。
    # 仅单条 ε' 时以前 len>1 才写文件，会导致只有 txt、无 per-seed CSV。
    if not args.attack_original and len(all_results) >= 1:
        print(f"\n{'='*60}")
        print(f"  汇总：{mix_tag} 下本 seed 各 ε' 的 Attack Accuracy")
        print(f"{'='*60}")
        print(f"  {'eps_prime':<12} {'attack_accuracy':<18} {'correct/total':<20}")
        print(f"  {'-'*12} {'-'*18} {'-'*20}")
        for r in all_results:
            print(
                f"  {r['eps_prime']:<12} {r['attack_accuracy']:<18.4f} "
                f"{r['correct']}/{r['valid_tokens']}"
            )
        summary_file = os.path.join(
            args.output_dir,
            f"attack_summary_mixed_{args.embedding_type}_{args.mapping_strategy}_"
            f"{mix_slug}_top_{args.top_k}_seed_{args.seed}.csv",
        )
        new_df = pd.DataFrame(all_results)
        if os.path.isfile(summary_file):
            old_df = pd.read_csv(summary_file)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
            new_df = new_df.drop_duplicates(subset=["eps_prime"], keep="last")
            new_df = new_df.sort_values("eps_prime").reset_index(drop=True)
        new_df.to_csv(summary_file, index=False)
        print(f"\n汇总已保存: {summary_file}")

    if not args.attack_original:
        print(f"\n  提示: 对照 baseline 请运行:\n"
              f"    python mask_token_attack_mixed.py --attack_original --seed {args.seed} "
              f"--output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
