"""
采样效率对比实验脚本（离线-在线框架）

将 SanText/CusText 的扰动流程分为离线阶段和在线阶段：
  - 离线阶段：加载词嵌入、构建 sim_word_dict / sim_dist_dict（所有方法相同）
  - 在线阶段：给定目标 ε，生成扰动数据

本脚本只测量在线阶段的时间开销，对比：
  - 原始方法：每个目标 ε 需要 p_dict 计算 + 逐 token 扰动
  - SA 方法：预计算 2 个端点 + 每个目标 ε 仅需轻量级混合采样

输出两张表：
  表1：单次在线生成效率对比（原始一次完整扰动 vs SA 一次混合采样）
  表2：多 ε 在线生成总时间对比（N 个目标 ε 的扩展性）

用法：
  python benchmark_efficiency.py --top_k 20 --eps_high 20
  python benchmark_efficiency.py --top_k 200 --eps_high 20 --num_eps_list 3 5 8 10 15 20
"""

import os
import io
import time
import argparse
import json
import random
import numpy as np
import pandas as pd


# ====================================================================
# 计时工具
# ====================================================================

class Timer:
    """简单的计时上下文管理器"""
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"    [{self.label}] {self.elapsed:.4f}s")


# ====================================================================
# SA 混合采样
# ====================================================================

def compute_mixing_probability(eps_target, eps_low, eps_high):
    """计算混合概率 p（取 eps_high 数据的概率）"""
    if eps_target <= eps_low:
        return 0.0
    if eps_target >= eps_high:
        return 1.0
    numerator = np.exp(eps_target) - np.exp(eps_low)
    term_a = np.exp((eps_low + eps_high) / 2) - np.exp(eps_low)
    term_b = (1 - np.exp((eps_low - eps_high) / 2)) * np.exp(eps_target)
    denominator = term_a + term_b
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def mix_dataset(df_low, df_high, probability, seed=42):
    """逐 token 混合两份数据集"""
    rng = np.random.default_rng(seed)
    mixed_sentences = []
    for idx in range(len(df_low)):
        tokens_low = str(df_low.iloc[idx]["sentence"]).split()
        tokens_high = str(df_high.iloc[idx]["sentence"]).split()
        min_len = min(len(tokens_low), len(tokens_high))
        tokens_low = tokens_low[:min_len]
        tokens_high = tokens_high[:min_len]
        mask = rng.random(min_len) < probability
        mixed = [tokens_high[i] if mask[i] else tokens_low[i] for i in range(min_len)]
        mixed_sentences.append(" ".join(mixed))
    result = df_low.copy()
    result["sentence"] = mixed_sentences
    return result


# ====================================================================
# 原始方法：p_dict 计算 + 逐 token 扰动
# ====================================================================

def compute_p_dict_local(sim_dist_dict, eps, top_k):
    """根据距离和 eps 计算每个词的采样概率字典"""
    p_dict = {}
    for word, dists in sim_dist_dict.items():
        if not isinstance(dists, list) or len(dists) == 0:
            continue
        dists_arr = np.array(dists[:top_k], dtype=np.float64)
        log_weights = eps * dists_arr / 2.0
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        probs = weights / weights.sum()
        p_dict[word] = probs.tolist()
    return p_dict


def generate_perturbed_data(df, sim_word_dict, p_dict, save_stop_words, seed, stop_words=None):
    """模拟 CusText/SanText 的逐句扰动过程"""
    rng = np.random.default_rng(seed)
    if stop_words is None:
        stop_words = set()

    new_sentences = []
    for idx in range(len(df)):
        tokens = str(df.iloc[idx]["sentence"]).split()
        new_tokens = []
        for token in tokens:
            token_lower = token.lower()
            if save_stop_words and token_lower in stop_words:
                new_tokens.append(token)
                continue
            if token_lower in p_dict and token_lower in sim_word_dict:
                candidates = sim_word_dict[token_lower]
                probs = p_dict[token_lower]
                actual_len = min(len(candidates), len(probs))
                if actual_len > 0:
                    prob_arr = np.array(probs[:actual_len])
                    prob_arr = prob_arr / prob_arr.sum()
                    chosen_idx = rng.choice(actual_len, p=prob_arr)
                    new_tokens.append(candidates[chosen_idx])
                else:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)
        new_sentences.append(" ".join(new_tokens))

    result = df.copy()
    result["sentence"] = new_sentences
    return result


# ====================================================================
# 辅助函数
# ====================================================================

def load_cached_dicts(embedding_type, mapping_strategy, top_k):
    """加载已缓存的 sim_word_dict、sim_dist_dict"""
    sim_word_path = f"./sim_word_dict/{embedding_type}/{mapping_strategy}/top_{top_k}.txt"
    sim_dist_path = f"./sim_dist_dict/{embedding_type}/{mapping_strategy}/top_{top_k}.txt"

    if not os.path.exists(sim_word_path) or not os.path.exists(sim_dist_path):
        raise FileNotFoundError(
            f"未找到缓存的 sim_word_dict/sim_dist_dict，请先运行:\n"
            f"  python build_mapping_cache.py --top_k {top_k}\n"
            f"  期望路径: {sim_word_path}"
        )

    with open(sim_word_path, 'r') as f:
        sim_word_dict = json.load(f)
    with open(sim_dist_path, 'r') as f:
        sim_dist_dict = json.load(f)

    return sim_word_dict, sim_dist_dict


def load_stop_words(save_stop_words):
    """加载停用词表"""
    stop_words = set()
    if save_stop_words:
        stop_words_path = "./stopwords_en.txt"
        if os.path.exists(stop_words_path):
            with open(stop_words_path, 'r') as f:
                stop_words = set(f.read().strip().split('\n'))
    return stop_words


def estimate_single_file_size_mb(df):
    """估算单份数据集的磁盘占用 (MB)"""
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False)
    return len(buf.getvalue().encode('utf-8')) / (1024 * 1024)


def count_total_tokens(df):
    """统计数据集中的 token 总数"""
    total = 0
    for idx in range(len(df)):
        total += len(str(df.iloc[idx]["sentence"]).split())
    return total


# ====================================================================
# 参数解析
# ====================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description="采样效率对比实验（离线-在线框架）"
    )
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    parser.add_argument("--mapping_strategy", type=str, default="paper")
    parser.add_argument("--save_stop_words", type=str, default="True")
    parser.add_argument("--eps_high", type=float, default=20.0,
                        help="SA 方法的 eps_high 端点")
    parser.add_argument("--eps_low", type=float, default=0.0,
                        help="SA 方法的 eps_low 端点")
    parser.add_argument("--num_eps_list", type=int, nargs="+",
                        default=[1, 2, 3, 5, 8, 10, 15, 20],
                        help="目标 eps 数量列表（用于表2）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="./efficiency_results",
                        help="结果输出目录")
    parser.add_argument("--repeat", type=int, default=3,
                        help="每组实验重复次数，取平均")
    return parser


# ====================================================================
# 实验 1：单次在线生成效率对比
# ====================================================================

def benchmark_single_online_step(df, sim_word_dict, sim_dist_dict, top_k,
                                  save_stop_words, stop_words, seed,
                                  eps_low, eps_high, repeat):
    """
    测量在线阶段的单次操作时间：
      - 原始方法：1 次 p_dict 计算 + 1 次逐 token 扰动
      - SA 方法：1 次混合采样（假设端点数据已预计算好）
    同时测量 SA 的端点预计算时间（只需做一次）。
    """
    eps_test = (eps_low + eps_high) / 2  # 用中间 eps 作为测试目标

    # --- 原始方法：单次完整扰动 ---
    original_p_dict_times = []
    original_generate_times = []
    original_total_times = []

    for _ in range(repeat):
        with Timer() as timer_p:
            p_dict = compute_p_dict_local(sim_dist_dict, eps_test, top_k)
        with Timer() as timer_g:
            _ = generate_perturbed_data(df, sim_word_dict, p_dict,
                                        save_stop_words, seed, stop_words)
        original_p_dict_times.append(timer_p.elapsed)
        original_generate_times.append(timer_g.elapsed)
        original_total_times.append(timer_p.elapsed + timer_g.elapsed)

    # --- SA 方法：端点预计算（一次性开销） ---
    precompute_times = []
    for _ in range(repeat):
        with Timer() as timer_pre:
            p_dict_low = compute_p_dict_local(sim_dist_dict, eps_low, top_k)
            df_low = generate_perturbed_data(df, sim_word_dict, p_dict_low,
                                              save_stop_words, seed, stop_words)
            p_dict_high = compute_p_dict_local(sim_dist_dict, eps_high, top_k)
            df_high = generate_perturbed_data(df, sim_word_dict, p_dict_high,
                                               save_stop_words, seed + 1, stop_words)
        precompute_times.append(timer_pre.elapsed)

    # 最后一次的端点数据用于后续混合测试
    # --- SA 方法：单次混合采样 ---
    mix_times = []
    probability = compute_mixing_probability(eps_test, eps_low, eps_high)
    for _ in range(repeat):
        with Timer() as timer_mix:
            _ = mix_dataset(df_low, df_high, probability, seed=seed)
        mix_times.append(timer_mix.elapsed)

    return {
        "eps_test": eps_test,
        # 原始方法
        "orig_p_dict_mean": np.mean(original_p_dict_times),
        "orig_p_dict_std": np.std(original_p_dict_times),
        "orig_generate_mean": np.mean(original_generate_times),
        "orig_generate_std": np.std(original_generate_times),
        "orig_total_mean": np.mean(original_total_times),
        "orig_total_std": np.std(original_total_times),
        # SA 方法
        "sa_precompute_mean": np.mean(precompute_times),
        "sa_precompute_std": np.std(precompute_times),
        "sa_mix_mean": np.mean(mix_times),
        "sa_mix_std": np.std(mix_times),
        # 比值
        "mix_vs_full_ratio": np.mean(mix_times) / np.mean(original_total_times),
    }


# ====================================================================
# 实验 2：多 ε 在线生成总时间对比
# ====================================================================

def benchmark_multi_eps(df, sim_word_dict, sim_dist_dict, top_k,
                         save_stop_words, stop_words, seed,
                         eps_low, eps_high, num_eps_list, repeat):
    """
    变化目标 ε 的数量 N，对比在线阶段总时间：
      - 原始方法：N × (p_dict + 扰动)
      - SA 方法（含预计算）：2 × (p_dict + 扰动) + N × 混合采样
      - SA 方法（不含预计算）：N × 混合采样（假设端点已缓存）
    """
    all_eps_candidates = np.linspace(eps_low, eps_high,
                                     num=max(num_eps_list) + 2).tolist()
    results = []

    for num_eps in num_eps_list:
        indices = np.linspace(0, len(all_eps_candidates) - 1,
                              num=num_eps, dtype=int)
        eps_subset = [all_eps_candidates[i] for i in indices]

        orig_times = []
        sa_full_times = []       # 含预计算
        sa_mix_only_times = []   # 仅混合（端点已缓存）

        for _ in range(repeat):
            # --- 原始方法：N 次完整扰动 ---
            with Timer() as timer_orig:
                for eps in eps_subset:
                    p_dict = compute_p_dict_local(sim_dist_dict, eps, top_k)
                    _ = generate_perturbed_data(df, sim_word_dict, p_dict,
                                                save_stop_words, seed, stop_words)
            orig_times.append(timer_orig.elapsed)

            # --- SA 方法：预计算 2 端点 + N 次混合 ---
            with Timer() as timer_sa_pre:
                p_dict_low = compute_p_dict_local(sim_dist_dict, eps_low, top_k)
                df_low = generate_perturbed_data(df, sim_word_dict, p_dict_low,
                                                  save_stop_words, seed, stop_words)
                p_dict_high = compute_p_dict_local(sim_dist_dict, eps_high, top_k)
                df_high = generate_perturbed_data(df, sim_word_dict, p_dict_high,
                                                   save_stop_words, seed + 1, stop_words)

            with Timer() as timer_sa_mix:
                for eps_target in eps_subset:
                    probability = compute_mixing_probability(eps_target, eps_low, eps_high)
                    _ = mix_dataset(df_low, df_high, probability, seed=seed)

            sa_full_times.append(timer_sa_pre.elapsed + timer_sa_mix.elapsed)
            sa_mix_only_times.append(timer_sa_mix.elapsed)

        orig_mean = np.mean(orig_times)
        sa_full_mean = np.mean(sa_full_times)
        sa_mix_mean = np.mean(sa_mix_only_times)

        speedup_full = orig_mean / sa_full_mean if sa_full_mean > 0 else float('inf')
        speedup_cached = orig_mean / sa_mix_mean if sa_mix_mean > 0 else float('inf')

        results.append({
            "num_eps": num_eps,
            "orig_time_mean": round(orig_mean, 4),
            "orig_time_std": round(np.std(orig_times), 4),
            "sa_full_time_mean": round(sa_full_mean, 4),
            "sa_full_time_std": round(np.std(sa_full_times), 4),
            "sa_mix_only_mean": round(sa_mix_mean, 4),
            "sa_mix_only_std": round(np.std(sa_mix_only_times), 4),
            "speedup_with_precompute": round(speedup_full, 2),
            "speedup_cached": round(speedup_cached, 2),
        })

    return results


# ====================================================================
# 主函数
# ====================================================================

def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  采样效率对比实验（离线-在线框架）")
    print("=" * 70)
    print(f"  数据集:        {args.dataset}")
    print(f"  top_k:         {args.top_k}")
    print(f"  embedding:     {args.embedding_type}")
    print(f"  SA 端点:       eps_low={args.eps_low}, eps_high={args.eps_high}")
    print(f"  目标 N 列表:   {args.num_eps_list}")
    print(f"  重复次数:      {args.repeat}")
    print("=" * 70)

    # ---- 离线阶段（所有方法相同，不计入对比） ----
    print("\n[离线阶段] 加载数据与缓存字典（所有方法共享，不纳入对比）...")

    train_path = f"datasets/{args.dataset}/train.tsv"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"未找到训练数据: {train_path}")
    df = pd.read_csv(train_path, sep="\t", keep_default_na=False)
    df["sentence"] = df["sentence"].fillna("")
    total_tokens = count_total_tokens(df)
    print(f"  训练集: {len(df)} 条样本, {total_tokens} 个 token")

    sim_word_dict, sim_dist_dict = load_cached_dicts(
        args.embedding_type, args.mapping_strategy, args.top_k
    )
    print(f"  词表大小: {len(sim_word_dict)} 词")

    save_stop_words = (args.save_stop_words == "True" or args.save_stop_words is True)
    stop_words = load_stop_words(save_stop_words)

    single_file_mb = estimate_single_file_size_mb(df)
    print(f"  单份数据集大小: {single_file_mb:.2f} MB")

    # ================================================================
    # 表 1：单次在线生成效率对比
    # ================================================================
    print("\n" + "=" * 70)
    print("  [表1] 单次在线生成效率对比")
    print("=" * 70)

    single_result = benchmark_single_online_step(
        df, sim_word_dict, sim_dist_dict, args.top_k,
        save_stop_words, stop_words, args.seed,
        args.eps_low, args.eps_high, args.repeat
    )

    print(f"\n  测试 eps = {single_result['eps_test']}")
    print(f"  数据集: {len(df)} 条样本, {total_tokens} 个 token")
    print()
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  原始方法（单次完整扰动）                            │")
    print(f"  │    p_dict 计算:    {single_result['orig_p_dict_mean']:.4f} ± {single_result['orig_p_dict_std']:.4f} s │")
    print(f"  │    逐 token 扰动:  {single_result['orig_generate_mean']:.4f} ± {single_result['orig_generate_std']:.4f} s │")
    print(f"  │    单次总计:       {single_result['orig_total_mean']:.4f} ± {single_result['orig_total_std']:.4f} s │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  SA 方法                                            │")
    print(f"  │    端点预计算(×1): {single_result['sa_precompute_mean']:.4f} ± {single_result['sa_precompute_std']:.4f} s │")
    print(f"  │    单次混合采样:   {single_result['sa_mix_mean']:.4f} ± {single_result['sa_mix_std']:.4f} s │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  混合采样 / 完整扰动 = {single_result['mix_vs_full_ratio']*100:.2f}%                    │")
    print(f"  │  即混合采样仅为完整扰动的 {single_result['mix_vs_full_ratio']*100:.1f}%             │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # 每 token 平均时间
    per_token_orig = single_result['orig_total_mean'] / total_tokens * 1000  # ms
    per_token_mix = single_result['sa_mix_mean'] / total_tokens * 1000  # ms
    per_token_speedup = per_token_orig / per_token_mix if per_token_mix > 0 else float('inf')
    print(f"\n  单 token 平均时间:")
    print(f"    原始方法: {per_token_orig:.4f} ms/token")
    print(f"    SA 混合:  {per_token_mix:.4f} ms/token")
    print(f"    加速比:   {per_token_speedup:.1f}×")

    # 保存表1
    table1_data = {
        "method": ["Original (full perturbation)", "SA (mix sampling)"],
        "single_online_time_mean_s": [
            round(single_result['orig_total_mean'], 4),
            round(single_result['sa_mix_mean'], 4),
        ],
        "single_online_time_std_s": [
            round(single_result['orig_total_std'], 4),
            round(single_result['sa_mix_std'], 4),
        ],
        "per_token_time_ms": [
            round(per_token_orig, 4),
            round(per_token_mix, 4),
        ],
        "precompute_time_s": [
            "N/A",
            round(single_result['sa_precompute_mean'], 4),
        ],
    }
    table1_df = pd.DataFrame(table1_data)
    table1_path = os.path.join(args.output_dir, "table1_single_step_efficiency.csv")
    table1_df.to_csv(table1_path, index=False)
    print(f"\n  表1 已保存: {table1_path}")

    # ================================================================
    # 表 2：多 ε 在线生成总时间对比
    # ================================================================
    print("\n" + "=" * 70)
    print("  [表2] 多 ε 在线生成总时间对比")
    print("=" * 70)

    multi_results = benchmark_multi_eps(
        df, sim_word_dict, sim_dist_dict, args.top_k,
        save_stop_words, stop_words, args.seed,
        args.eps_low, args.eps_high, args.num_eps_list, args.repeat
    )

    # 打印汇总
    print(f"\n  {'N':>4} │ {'原始方法(s)':>14} │ {'SA含预计算(s)':>14} │ {'SA仅混合(s)':>14} │ {'加速比(含预)':>12} │ {'加速比(缓存)':>12} │ {'磁盘节省':>8}")
    print(f"  {'─'*4}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*8}")

    for r in multi_results:
        num = r['num_eps']
        disk_saving = (1 - 2 / num) * 100 if num > 0 else 0
        print(f"  {num:>4} │ "
              f"{r['orig_time_mean']:>7.3f}±{r['orig_time_std']:<5.3f} │ "
              f"{r['sa_full_time_mean']:>7.3f}±{r['sa_full_time_std']:<5.3f} │ "
              f"{r['sa_mix_only_mean']:>7.3f}±{r['sa_mix_only_std']:<5.3f} │ "
              f"{r['speedup_with_precompute']:>10.2f}× │ "
              f"{r['speedup_cached']:>10.2f}× │ "
              f"{disk_saving:>6.1f}%")

    # 添加磁盘信息
    for r in multi_results:
        num = r['num_eps']
        r["disk_original_MB"] = round(single_file_mb * num, 2)
        r["disk_sa_cached_MB"] = round(single_file_mb * 2, 2)
        r["disk_saving_pct"] = round((1 - 2 / num) * 100, 1) if num > 0 else 0

    # 保存表2
    table2_df = pd.DataFrame(multi_results)
    table2_path = os.path.join(args.output_dir, "table2_multi_eps_efficiency.csv")
    table2_df.to_csv(table2_path, index=False)
    print(f"\n  表2 已保存: {table2_path}")

    # ================================================================
    # 汇总信息
    # ================================================================
    print("\n" + "=" * 70)
    print("  实验汇总")
    print("=" * 70)
    print(f"\n  关键结论:")
    print(f"    1. SA 单次混合采样仅为原始完整扰动的 {single_result['mix_vs_full_ratio']*100:.1f}%")
    print(f"    2. 单 token 平均: 原始 {per_token_orig:.4f} ms vs SA {per_token_mix:.4f} ms ({per_token_speedup:.1f}× 加速)")

    if len(multi_results) > 0:
        last = multi_results[-1]
        print(f"    3. N={last['num_eps']} 时: "
              f"原始 {last['orig_time_mean']:.3f}s → "
              f"SA(含预计算) {last['sa_full_time_mean']:.3f}s "
              f"({last['speedup_with_precompute']:.1f}× 加速)")
        print(f"    4. N={last['num_eps']} 时磁盘节省: {last['disk_saving_pct']:.0f}%")

    print(f"\n  输出文件:")
    print(f"    表1: {table1_path}")
    print(f"    表2: {table2_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()