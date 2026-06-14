#!/usr/bin/env python3
"""
绘制采样概率 p 随 eps_target 变化的曲线。
每条线对应不同的 eps_high（eps_low 固定为 0）。
在每条曲线上标注 p=0.5 对应的 eps_target。
"""

import numpy as np
import matplotlib.pyplot as plt


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


def find_eps_at_probability(target_prob, eps_low, eps_high, resolution=10000):
    """二分查找 p = target_prob 时对应的 eps_target。"""
    lo, hi = eps_low, eps_high
    for _ in range(resolution):
        mid = (lo + hi) / 2.0
        current_prob = compute_mixing_probability(mid, eps_low, eps_high)
        if current_prob < target_prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main():
    eps_low = 0.0
    eps_high_list = [14, 16, 18, 20, 22, 24, 26]
    target_prob = 0.5  # 标注的目标概率

    eps_range = np.linspace(0, 28, 1000)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(eps_high_list)))

    annotation_points = []

    for idx, eps_high in enumerate(eps_high_list):
        probabilities = [
            compute_mixing_probability(eps, eps_low, eps_high)
            for eps in eps_range
        ]
        color = colors[idx]
        label = rf"$\epsilon_{{high}}={eps_high}$"
        ax.plot(eps_range, probabilities, color=color, linewidth=2, label=label)

        # 找到 p = target_prob 对应的 eps_target 并标注
        eps_at_p = find_eps_at_probability(target_prob, eps_low, eps_high)
        ax.plot(eps_at_p, target_prob, "o", color=color, markersize=8, zorder=5)
        annotation_points.append((eps_at_p, eps_high, color))

    # 统一标注：每个点从 p=0.5 处垂直向下引线到不同高度，避免重叠
    for i, (eps_at_p, eps_h, color) in enumerate(annotation_points):
        text_y = -0.12 + i * 0.035  # 在图底部错开排列
        # 在点旁边直接写文字，不用箭头避免杂乱
        ax.vlines(eps_at_p, text_y + 0.02, target_prob, color=color,
                  linestyle=":", linewidth=1, alpha=0.6)
        ax.text(eps_at_p, text_y, rf"$\epsilon'$={eps_at_p:.1f}",
                fontsize=9, color=color, fontweight="bold",
                ha="center", va="top")

    # p = target_prob 的参考线
    ax.axhline(y=target_prob, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.3, target_prob + 0.02, f"p = {target_prob}", fontsize=10, color="gray")

    ax.set_xlabel(r"$\epsilon'$ (eps_target)", fontsize=13)
    ax.set_ylabel("Sample Probability  $p$", fontsize=13)
    ax.set_title(
        r"Sample Probability vs $\epsilon'$ for different $\epsilon_{high}$",
        #f"  ($\\epsilon_{{low}}={eps_low:.0f}$)",
        fontsize=14,
    )
    ax.set_xlim(0, 28)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = "plot_mixing_probability.png"
    plt.savefig(output_path, dpi=200)
    print(f"图已保存至: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
