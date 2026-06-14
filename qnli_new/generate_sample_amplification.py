"""
QNLI Sample Amplification 数据生成脚本。

基于论文 3.2/3.3 节的 Sample Amplification：
给定 eps_low 和 eps_high 两份已脱敏数据，对每个 token 做 Bernoulli 采样混合，
生成目标 eps' 对应的 mixed private 数据。

与 sst2_new 版本的主要差异：
  - QNLI 同时混合 `question` 与 `sentence` 两列。
"""

import argparse
import os

import numpy as np
import pandas as pd


def compute_mixing_probability(eps_target, eps_low, eps_high):
    """计算混合概率 p（取 eps_high token 的概率）。"""
    if eps_target <= eps_low:
        return 0.0
    if eps_target >= eps_high:
        return 1.0

    numerator = np.exp(eps_target) - np.exp(eps_low)
    term_a = np.exp((eps_low + eps_high) / 2.0) - np.exp(eps_low)
    term_b = (1.0 - np.exp((eps_low - eps_high) / 2.0)) * np.exp(eps_target)
    denominator = term_a + term_b

    probability = numerator / denominator
    return float(np.clip(probability, 0.0, 1.0))


def mix_text(text_low, text_high, probability, rng):
    """逐 token 混合两段文本；长度不一致时按较短长度截断。"""
    tokens_low = str(text_low).split()
    tokens_high = str(text_high).split()

    if len(tokens_low) != len(tokens_high):
        min_len = min(len(tokens_low), len(tokens_high))
        tokens_low = tokens_low[:min_len]
        tokens_high = tokens_high[:min_len]

    if not tokens_low:
        return ""

    mask = rng.random(len(tokens_low)) < probability
    mixed_tokens = [
        tokens_high[i] if mask[i] else tokens_low[i]
        for i in range(len(tokens_low))
    ]
    return " ".join(mixed_tokens)


def generate_mixed_dataset(df_low, df_high, probability, seed=42):
    """对 QNLI 的 question/sentence 两列逐行逐 token 混合。"""
    if len(df_low) != len(df_high):
        raise ValueError(f"行数不一致: low={len(df_low)}, high={len(df_high)}")

    for col in ("question", "sentence"):
        if col not in df_low.columns or col not in df_high.columns:
            raise KeyError(f"缺少列 `{col}`，无法进行 QNLI 混合。")

    rng = np.random.default_rng(seed)
    result_df = df_low.copy()

    mixed_questions = []
    mixed_sentences = []
    for idx in range(len(df_low)):
        q_low = df_low.iloc[idx]["question"]
        q_high = df_high.iloc[idx]["question"]
        s_low = df_low.iloc[idx]["sentence"]
        s_high = df_high.iloc[idx]["sentence"]

        mixed_questions.append(mix_text(q_low, q_high, probability, rng))
        mixed_sentences.append(mix_text(s_low, s_high, probability, rng))

    result_df["question"] = mixed_questions
    result_df["sentence"] = mixed_sentences
    return result_df


def build_data_path(base_dir, eps_value, top_k=20, strategy="s1", save_stop_words=False, seed=42):
    """构建 privatized_dataset 子目录路径。"""
    folder_name = (
        f"eps_{eps_value}_top_{top_k}_{strategy}"
        f"_save_stop_words_{save_stop_words}_seed_{seed}"
    )
    return os.path.join(base_dir, folder_name)


def format_eps_folder_name(eps_target):
    """把 eps 浮点数转换成目录友好的字符串。"""
    if abs(eps_target - round(eps_target)) < 1e-9:
        return str(int(round(eps_target)))
    return str(eps_target).replace(".", "p")


def get_eps_iter(args):
    if args.eps_targets is not None:
        return sorted({float(v) for v in args.eps_targets})
    return [
        float(v)
        for v in range(int(args.eps_low), int(args.eps_high) + 1, args.eps_step)
    ]


def main():
    parser = argparse.ArgumentParser(description="生成 QNLI mixed private 数据")
    parser.add_argument("--eps_low", type=float, default=0.0, help="低隐私预算")
    parser.add_argument("--eps_high", type=float, default=20.0, help="高隐私预算")
    parser.add_argument("--eps_step", type=int, default=2, help="eps' 扫描步长")
    parser.add_argument(
        "--eps_targets",
        type=float,
        nargs="+",
        default=None,
        help="指定目标 eps' 列表；指定后忽略 --eps_step",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", type=str, default="False")
    parser.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    parser.add_argument("--mapping_strategy", type=str, default="paper")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="要生成的 seed 列表",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="源脱敏数据根目录（默认 ./privatized_dataset/{embedding_type}/{mapping_strategy}）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出根目录（默认 ./privatized_dataset_mixed/{embedding_type}/{mapping_strategy}）",
    )
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
    if args.output_dir is None:
        args.output_dir = f"./privatized_dataset_mixed/{args.embedding_type}/{args.mapping_strategy}"

    if args.eps_low > args.eps_high:
        raise ValueError("--eps_low 不能大于 --eps_high")
    if args.eps_targets is None and args.eps_step <= 0:
        raise ValueError("--eps_step 必须为正整数")

    eps_iter = get_eps_iter(args)
    total = len(args.seeds)

    for idx, seed in enumerate(args.seeds):
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{total}] seed={seed}")
        print(f"{'=' * 60}")

        low_path = build_data_path(
            args.base_dir, args.eps_low, args.top_k, args.strategy, args.save_stop_words, seed
        )
        high_path = build_data_path(
            args.base_dir, args.eps_high, args.top_k, args.strategy, args.save_stop_words, seed
        )

        print(f"低隐私源: eps={args.eps_low}, seed={seed}  路径: {low_path}")
        print(f"高隐私源: eps={args.eps_high}, seed={seed}  路径: {high_path}")

        for split in ("train", "test"):
            low_file = os.path.join(low_path, f"{split}.tsv")
            high_file = os.path.join(high_path, f"{split}.tsv")

            if not os.path.exists(low_file):
                print(f"[警告] 文件不存在: {low_file}，跳过 {split}")
                continue
            if not os.path.exists(high_file):
                print(f"[警告] 文件不存在: {high_file}，跳过 {split}")
                continue

            df_low = pd.read_csv(low_file, sep="\t", keep_default_na=False)
            df_high = pd.read_csv(high_file, sep="\t", keep_default_na=False)
            print(f"已加载 {split} 数据: {len(df_low)} 行")

            for eps_target in eps_iter:
                probability = compute_mixing_probability(eps_target, args.eps_low, args.eps_high)
                eps_folder_str = format_eps_folder_name(eps_target)
                output_folder = os.path.join(
                    args.output_dir,
                    f"mix_{args.eps_low}_{args.eps_high}",
                    f"eps_{eps_folder_str}_top_{args.top_k}_{args.strategy}"
                    f"_save_stop_words_{args.save_stop_words}_seed_{seed}",
                )
                output_file = os.path.join(output_folder, f"{split}.tsv")

                if os.path.exists(output_file):
                    print(f"  eps'={eps_target:5.1f}  已存在，跳过: {output_file}")
                    continue

                print(
                    f"  eps'={eps_target:5.1f}  p={probability:.6f}  "
                    f"(取 eps={args.eps_high} 的 token 概率)"
                )
                mixed_df = generate_mixed_dataset(df_low, df_high, probability, seed=seed)

                os.makedirs(output_folder, exist_ok=True)
                mixed_df.to_csv(output_file, sep="\t", index=False)

    print("\n=== 生成完成 ===")
    print(f"输出目录: {os.path.join(args.output_dir, f'mix_{args.eps_low}_{args.eps_high}')}")


if __name__ == "__main__":
    main()
