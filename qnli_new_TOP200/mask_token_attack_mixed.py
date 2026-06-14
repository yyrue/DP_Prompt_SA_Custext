"""
QNLI mixed 数据 (Sample Amplification) 的 Mask Token Attack。
"""

import argparse
import datetime
import os

import pandas as pd
import torch
from transformers import BertForMaskedLM, BertTokenizer

from mask_token_attack import load_original_data, run_mask_token_attack, set_seed


def get_parser():
    parser = argparse.ArgumentParser(description="Mask Token Attack on QNLI mixed privatized data")
    parser.add_argument("--dataset", type=str, default="qnli")
    parser.add_argument("--model_path", type=str, default="/data/youyaru/SanText-main/bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./attack_results_mixed")
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--privatized_root", type=str, default="./privatized_dataset_mixed")
    parser.add_argument("--eps_low", type=float, default=0.0)
    parser.add_argument("--eps_high", type=float, default=14.0)
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--eps_list", type=float, nargs="+", default=None)

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


def _eps_folder(eps_target):
    if abs(eps_target - round(eps_target)) < 1e-9:
        return str(int(round(eps_target)))
    return str(eps_target).replace(".", "p")


def _mix_subdir(eps_low, eps_high):
    return f"mix_{float(eps_low)}_{float(eps_high)}"


def load_privatized_mixed(args, eps_target):
    eps_folder = _eps_folder(eps_target)
    priv_dir = os.path.join(
        args.privatized_root,
        args.embedding_type,
        args.mapping_strategy,
        _mix_subdir(args.eps_low, args.eps_high),
        f"eps_{eps_folder}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{args.save_stop_words}_seed_{args.seed}",
    )
    test_path = os.path.join(priv_dir, "test.tsv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"未找到 mixed 脱敏数据: {test_path}")
    test_df = pd.read_csv(test_path, sep="\t", keep_default_na=False).reset_index(drop=True)
    test_df["question"] = test_df["question"].fillna("")
    test_df["sentence"] = test_df["sentence"].fillna("")
    return test_df


def save_results_mixed(results, args, eps_target=None):
    os.makedirs(args.output_dir, exist_ok=True)
    mix_tag = _mix_subdir(args.eps_low, args.eps_high)
    mix_slug = mix_tag.replace(".", "d")

    if args.attack_original:
        result_file = os.path.join(args.output_dir, f"attack_mixed_baseline_seed_{args.seed}.txt")
    else:
        eps_str = _eps_folder(args.eps if eps_target is None else eps_target)
        result_file = os.path.join(
            args.output_dir,
            f"attack_mixed_{args.embedding_type}_{args.mapping_strategy}_{mix_slug}"
            f"_epsprime_{eps_str}_top_{args.top_k}_seed_{args.seed}"
            f"_save_stop_words_{args.save_stop_words}.txt",
        )

    with open(result_file, "w") as f:
        f.write("Mask Token Inference Attack Results (QNLI mixed)\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if not args.attack_original:
            f.write(f"mix_eps_low={args.eps_low}, mix_eps_high={args.eps_high}\n")
            f.write(f"eps_prime={eps_target}\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"结果已保存: {result_file}")


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    mlm_model = BertForMaskedLM.from_pretrained(args.model_path).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        mlm_model = torch.nn.DataParallel(mlm_model)

    original_df = load_original_data(args.dataset)
    eps_targets = args.eps_list if args.eps_list else [args.eps]
    all_results = []

    if args.attack_original:
        results = run_mask_token_attack(
            original_df=original_df,
            masked_df=original_df,
            tokenizer=tokenizer,
            mlm_model=mlm_model,
            device=device,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
        if results is not None:
            save_results_mixed(results, args, eps_target=None)
    else:
        mix_tag = _mix_subdir(args.eps_low, args.eps_high)
        mix_slug = mix_tag.replace(".", "d")
        for eps_target in eps_targets:
            masked_df = load_privatized_mixed(args, eps_target)
            results = run_mask_token_attack(
                original_df=original_df,
                masked_df=masked_df,
                tokenizer=tokenizer,
                mlm_model=mlm_model,
                device=device,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
            )
            if results is None:
                continue
            save_results_mixed(results, args, eps_target=eps_target)
            results["eps_prime"] = float(eps_target)
            results["eps_low"] = args.eps_low
            results["eps_high"] = args.eps_high
            results["seed"] = args.seed
            results["mix_tag"] = mix_tag
            all_results.append(results)

        if all_results:
            summary_file = os.path.join(
                args.output_dir,
                f"attack_summary_mixed_{args.embedding_type}_{args.mapping_strategy}_"
                f"{mix_slug}_top_{args.top_k}_seed_{args.seed}.csv",
            )
            new_df = pd.DataFrame(all_results).sort_values("eps_prime").reset_index(drop=True)
            if os.path.isfile(summary_file):
                old_df = pd.read_csv(summary_file)
                new_df = pd.concat([old_df, new_df], ignore_index=True)
                new_df = new_df.drop_duplicates(subset=["eps_prime"], keep="last")
                new_df = new_df.sort_values("eps_prime").reset_index(drop=True)
            new_df.to_csv(summary_file, index=False)
            print(f"汇总已保存: {summary_file}")


if __name__ == "__main__":
    main()
