#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample Amplification 概率画图脚本
固定 eps_low，改变 eps_high，绘制不同 eps_high 下采样概率 p 随目标 eps' 的变化曲线。
公式：p = (e^{eps_target} - e^{eps_low}) / (term_a + term_b)

用法:
  python plot_sa_probability_fixed_eps_low.py --eps_low 5 --eps_high_list 16 18 20 22 24 26 28 30 32 34
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def compute_mixing_probability(eps_target, eps_low, eps_high):
    """3.3节通用公式"""
    if eps_target <= eps_low:
        return 0.0
    if eps_target >= eps_high:
        return 1.0
    numerator = np.exp(eps_target) - np.exp(eps_low)
    term_a = np.exp((eps_low + eps_high) / 2) - np.exp(eps_low)
    term_b = (1 - np.exp((eps_low - eps_high) / 2)) * np.exp(eps_target)
    denominator = term_a + term_b
    return np.clip(numerator / denominator, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="绘制 Sample Amplification 概率曲线（固定 eps_low，改变 eps_high）"
    )
    parser.add_argument(
        "--eps_low", type=float, default=5,
        help="固定的 eps_low 值 (默认: 5)"
    )
    parser.add_argument(
        "--eps_high_list", type=float, nargs="+",
        default=[16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
        help="eps_high 值列表 (默认: 16 18 20 22 24 26 28 30 32 34)"
    )
    args = parser.parse_args()

    EPS_LOW = args.eps_low
    EPS_HIGH_LIST = sorted(args.eps_high_list)

    # 颜色映射：用渐变色区分不同 eps_high
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(EPS_HIGH_LIST) - 1)) for i in range(len(EPS_HIGH_LIST))]

    plt.figure(figsize=(10, 6))

    for eps_high, color in zip(EPS_HIGH_LIST, colors):
        # 对每个 eps_high，在 (EPS_LOW, eps_high) 范围内计算 p
        eps_targets = np.linspace(EPS_LOW, eps_high, 200)
        probabilities = np.array([
            compute_mixing_probability(et, EPS_LOW, eps_high)
            for et in eps_targets
        ])

        plt.plot(eps_targets, probabilities, color=color, linewidth=2.0,
                 label=rf"$\varepsilon_{{high}}$ = {eps_high}")

    # 标注关键点：eps' = (eps_high + eps_low) / 2 时 p ≈ 0.5
    # 选取首、中、末三个 eps_high 做标注
    annotate_indices = [0, len(EPS_HIGH_LIST) // 2, len(EPS_HIGH_LIST) - 1]
    for idx in annotate_indices:
        eps_high = EPS_HIGH_LIST[idx]
        half_eps = (eps_high + EPS_LOW) / 2
        p_half = compute_mixing_probability(half_eps, EPS_LOW, eps_high)
        plt.scatter([half_eps], [p_half], color="red", s=40, zorder=5)
        plt.annotate(
            rf"$\varepsilon'$={half_eps}", (half_eps, p_half),
            textcoords="offset points", xytext=(5, -10),
            fontsize=9, color="red"
        )

    plt.xlabel(r"Target privacy budget $\varepsilon'$", fontsize=13)
    plt.ylabel(r"Sampling probability $p$ (prob. of picking $\varepsilon_{high}$)", fontsize=13)
    plt.title(
        rf"Sample Amplification: $p$ vs $\varepsilon'$  ($\varepsilon_{{low}}$={EPS_LOW})",
        fontsize=14
    )

    max_eps_high = max(EPS_HIGH_LIST)
    plt.xlim(0, max_eps_high)
    plt.ylim(-0.02, 1.02)

    # 虚线标注 p=0.5
    plt.axhline(y=0.5, linestyle="--", color="gray", alpha=0.5, linewidth=1.0)

    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存（文件名包含 EPS_LOW 信息）
    output_dir = "./experiment_results_sa"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"sa_probability_curves_eps_low_{EPS_LOW}.png"
    )
    plt.savefig(output_file, dpi=200)
    print(f"图片已保存到: {output_file}")


if __name__ == "__main__":
    main()