# 根据 sim_dist_dict 和 eps 计算概率分布（很快，无需加载嵌入）
def compute_p_dict(sim_dist_dict, eps):
    p_dict = defaultdict(list)
    for word, new_sim_dist_list in sim_dist_dict.items():
        tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
        norm_val = sum(tmp)
        p = [x/norm_val for x in tmp]
        p_dict[word] = p

    # 缓存 p_dict（按 eps + top_k 缓存）
    p_dict_dir = f"./p_dict/eps_0.0_top_20.txt"
    os.makedirs(p_dict_dir, exist_ok=True)

    with open(f"{p_dict_dir}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
        json_file.write(json.dumps(p_dict, ensure_ascii=False, indent=4))

    return p_dict

from collections import defaultdict
import json
import numpy as np
p_dict = defaultdict(list)
print(type(p_dict))
sim_dist_dict = open("./sim_dist_dict/glove_840B-300d/conservative/top_20.txt", "r")
sim_dist_dict_js = json.load(sim_dist_dict)
total = 0
for word, new_smi_dist_list in sim_dist_dict_js.items():
    total += 1
    tmp = [np.exp(0 * x / 2) for x in new_smi_dist_list]
    
    norm_val = sum(tmp)
    p = [x/norm_val for x in tmp]
    print(word,p)
    p_dict[word] = p
    if total == 4:
        break
