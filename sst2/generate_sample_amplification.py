"""
Sample Amplification 数据生成脚本

基于论文 3.2 节的 Sample Amplification 方法：
给定 eps1（低隐私预算，如 eps=0）和 eps2（高隐私预算，如 eps=32）的已扰动数据，
通过逐 token 的 Bernoulli 采样，混合生成满足目标隐私预算 eps' 的新数据。

3.2 节公式（eps1=0 的特例）:
    p = (e^{eps'} - 1) / ((e^{eps2/2} - 1) * (e^{eps' - eps2/2} + 1))

3.3 节公式（一般情形，eps1 > 0）:
    p = (e^{eps'} - e^{eps1}) / ((e^{(eps1+eps2)/2} - e^{eps1}) + (1 - e^{(eps1-eps2)/2}) * e^{eps'})

对每个 token：以概率 p 取 eps2 的 token，以概率 1-p 取 eps1 的 token。
"""

import os
import argparse
import numpy as np
import pandas as pd


def compute_mixing_probability(eps_target, eps_low, eps_high):
    """
    计算混合概率 p（取 eps_high 数据的概率）。

    当 eps_low = 0 时退化为 3.2 节公式:
        p = (e^{eps'} - 1) / ((e^{eps_high/2} - 1) * (e^{eps' - eps_high/2} + 1))

    一般情形使用 3.3 节公式:
        p = (e^{eps'} - e^{eps_low}) / ((e^{(eps_low+eps_high)/2} - e^{eps_low})
            + (1 - e^{(eps_low-eps_high)/2}) * e^{eps'})
    """
    if eps_target <= eps_low:
        return 0.0
    if eps_target >= eps_high:
        return 1.0

    numerator = np.exp(eps_target) - np.exp(eps_low)
    term_a = np.exp((eps_low + eps_high) / 2) - np.exp(eps_low)
    term_b = (1 - np.exp((eps_low - eps_high) / 2)) * np.exp(eps_target)
    denominator = term_a + term_b

    probability = numerator / denominator
    return float(np.clip(probability, 0.0, 1.0))


def mix_sentences(sentence_low, sentence_high, probability, rng):
    """
    逐 token 混合两个句子。
    以概率 p 选择 high 的 token，否则选择 low 的 token。
    """
    tokens_low = str(sentence_low).split()
    tokens_high = str(sentence_high).split()

    # 两个句子的 token 数量应一致（同一原始句子的不同扰动版本）
    if len(tokens_low) != len(tokens_high):
        # fallback: 按较短的截断
        min_length = min(len(tokens_low), len(tokens_high))
        tokens_low = tokens_low[:min_length]
        tokens_high = tokens_high[:min_length]

    bernoulli_mask = rng.random(len(tokens_low)) < probability
    mixed_tokens = [
        tokens_high[i] if bernoulli_mask[i] else tokens_low[i]
        for i in range(len(tokens_low))
    ]
    return " ".join(mixed_tokens)


def generate_mixed_dataset(df_low, df_high, probability, seed=42):
    """
    对整个数据集逐行逐 token 进行混合。
    """
    rng = np.random.default_rng(seed)
    mixed_sentences = []

    for idx in range(len(df_low)):
        sentence_low = df_low.iloc[idx]["sentence"]
        sentence_high = df_high.iloc[idx]["sentence"]
        mixed_sentence = mix_sentences(sentence_low, sentence_high, probability, rng)
        mixed_sentences.append(mixed_sentence)

    result_df = df_low.copy()
    result_df["sentence"] = mixed_sentences
    return result_df


def build_data_path(base_dir, eps_value, top_k=20, strategy="s1", save_stop_words=False):
    """构建 privatized_dataset 的目录路径"""
    folder_name = f"eps_{eps_value}_top_{top_k}_{strategy}_save_stop_words_{save_stop_words}"
    return os.path.join(base_dir, folder_name)


def main():
    parser = argparse.ArgumentParser(description="Sample Amplification 数据生成")
    parser.add_argument("--eps_low", type=float, default=0.0,
                        help="低隐私预算 (默认 0.0，即完全随机)")
    parser.add_argument("--eps_high", type=float, default=32.0,
                        help="高隐私预算 (默认 32.0)")
    # parser.add_argument("--eps_targets", type=float, nargs="+",
    #                     default=[x for x in range(int(args.eps_low)+1,int(args.eps_high))],
    #                     help="目标隐私预算列表")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", type=str, default="False")
    parser.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    parser.add_argument("--mapping_strategy", type=str, default="conservative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_dir", type=str,
                        default="./privatized_dataset/glove_840B-300d/conservative")
    parser.add_argument("--output_dir", type=str,
                        default="./privatized_dataset_mixed/glove_840B-300d/conservative")
    args = parser.parse_args()

    # 加载两份源数据
    low_path = build_data_path(args.base_dir, args.eps_low, args.top_k,
                               args.strategy, args.save_stop_words)
    high_path = build_data_path(args.base_dir, args.eps_high, args.top_k,
                                args.strategy, args.save_stop_words)

    print(f"=== Sample Amplification 数据生成 ===")
    print(f"低隐私源: eps={args.eps_low}  路径: {low_path}")
    print(f"高隐私源: eps={args.eps_high}  路径: {high_path}")
    print()

    for split in ["train", "test"]:
        low_file = os.path.join(low_path, f"{split}.tsv")
        high_file = os.path.join(high_path, f"{split}.tsv")

        if not os.path.exists(low_file):
            print(f"[警告] 文件不存在: {low_file}，跳过 {split}")
            continue
        if not os.path.exists(high_file):
            print(f"[警告] 文件不存在: {high_file}，跳过 {split}")
            continue

        df_low = pd.read_csv(low_file, sep="\t")
        df_high = pd.read_csv(high_file, sep="\t")
        print(f"已加载 {split} 数据: {len(df_low)} 行")

        for eps_target in range(int(args.eps_low)+1,int(args.eps_high)):
            if eps_target <= args.eps_low or eps_target >= args.eps_high:
                print(f"  [跳过] eps'={eps_target} 不在 ({args.eps_low}, {args.eps_high}) 范围内")
                continue

            probability = compute_mixing_probability(eps_target, args.eps_low, args.eps_high)
            print(f"  eps'={eps_target:5.1f}  p={probability:.6f}  "
                  f"(取 eps={args.eps_high} 的概率)")

            mixed_df = generate_mixed_dataset(df_low, df_high, probability, seed=args.seed)

            # 保存到输出目录，命名格式与原始一致，便于下游训练直接使用
            output_folder = os.path.join(
                args.output_dir,
                f"mix_{args.eps_low}_{args.eps_high}",
                f"eps_{eps_target}_top_{args.top_k}_{args.strategy}"
                f"_save_stop_words_{args.save_stop_words}"
            )
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{split}.tsv")
            mixed_df.to_csv(output_file, sep="\t", index=False)

    print()
    print("=== 生成完成 ===")
    print(f"输出目录: {os.path.join(args.output_dir, f'mix_{args.eps_low}_{args.eps_high}')}")


if __name__ == "__main__":
    main()
