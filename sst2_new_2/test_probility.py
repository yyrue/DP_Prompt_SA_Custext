import numpy as np
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


print(compute_mixing_probability(13, 0, 22))