#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample Amplification 概率画图脚本
固定 eps_high，改变 eps_low，绘制不同 eps_low 下采样概率 p 随目标 eps' 的变化曲线。
公式：p = (e^{eps_target} - e^{eps_low}) / (term_a + term_b)

用法:
  python plot_sa_probability_fixed_eps_high.py --eps_high 30 --eps_low_list 0 2 4 6 8 10 12
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
        description="绘制 Sample Amplification 概率曲线（固定 eps_high，改变 eps_low）"
    )
    parser.add_argument(
        "--eps_high", type=float, default=30,
        help="固定的 eps_high 值 (默认: 30)"
    )
    parser.add_argument(
        "--eps_low_list", type=float, nargs="+",
        default=[0, 2, 4, 6, 8, 10, 12],
        help="eps_low 值列表 (默认: 0 2 4 6 8 10 12)"
    )
    args = parser.parse_args()

    EPS_HIGH = args.eps_high
    EPS_LOW_LIST = sorted(args.eps_low_list)

    # 过滤掉 >= eps_high 的 eps_low 值
    valid_eps_low_list = [el for el in EPS_LOW_LIST if el < EPS_HIGH]
    if len(valid_eps_low_list) < len(EPS_LOW_LIST):
        skipped = set(EPS_LOW_LIST) - set(valid_eps_low_list)
        print(f"警告: eps_low 值 {skipped} >= eps_high={EPS_HIGH}，已跳过")
    if not valid_eps_low_list:
        print("错误: 没有有效的 eps_low 值（均 >= eps_high），退出")
        return

    # 颜色映射：用渐变色区分不同 eps_low
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(valid_eps_low_list) - 1)) for i in range(len(valid_eps_low_list))]

    plt.figure(figsize=(10, 6))

    for eps_low, color in zip(valid_eps_low_list, colors):
        # 对每个 eps_low，在 (eps_low, EPS_HIGH) 范围内计算 p
        eps_targets = np.linspace(eps_low, EPS_HIGH, 200)
        probabilities = np.array([
            compute_mixing_probability(et, eps_low, EPS_HIGH)
            for et in eps_targets
        ])

        plt.plot(eps_targets, probabilities, color=color, linewidth=2.0,
                 label=rf"$\varepsilon_{{low}}$ = {eps_low}")

    # 标注关键点：eps' = (eps_high + eps_low) / 2 时 p ≈ 0.5
    # 选取首、中、末三个 eps_low 做标注
    annotate_indices = [0, len(valid_eps_low_list) // 2, len(valid_eps_low_list) - 1]
    for idx in annotate_indices:
        eps_low = valid_eps_low_list[idx]
        half_eps = (EPS_HIGH + eps_low) / 2
        p_half = compute_mixing_probability(half_eps, eps_low, EPS_HIGH)
        plt.scatter([half_eps], [p_half], color="red", s=40, zorder=5)
        plt.annotate(
            rf"$\varepsilon'$={half_eps}", (half_eps, p_half),
            textcoords="offset points", xytext=(5, -10),
            fontsize=9, color="red"
        )

    plt.xlabel(r"Target privacy budget $\varepsilon'$", fontsize=13)
    plt.ylabel(r"Sampling probability $p$ (prob. of picking $\varepsilon_{high}$)", fontsize=13)
    plt.title(
        rf"Sample Amplification: $p$ vs $\varepsilon'$  ($\varepsilon_{{high}}$={EPS_HIGH})",
        fontsize=14
    )

    plt.xlim(0, EPS_HIGH)
    plt.ylim(-0.02, 1.02)

    # 虚线标注 p=0.5
    plt.axhline(y=0.5, linestyle="--", color="gray", alpha=0.5, linewidth=1.0)

    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存（文件名包含 EPS_HIGH 信息）
    output_dir = "./experiment_results_sa"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"sa_probability_curves_eps_high_{EPS_HIGH}.png"
    )
    plt.savefig(output_file, dpi=200)
    print(f"图片已保存到: {output_file}")


if __name__ == "__main__":
    main()