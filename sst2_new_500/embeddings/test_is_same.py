"""
从 glove_840B-300d.txt 中只保留 ct_vectors.txt 里出现的词的 embedding，输出到 glove_filtered.txt
"""

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
ct_path = os.path.join(base_dir, "ct_vectors.txt")
glove_path = os.path.join(base_dir, "/data/youyaru/CusText-main/CusText/sst2/embeddings/glove_840B-300d.txt")
output_path = os.path.join(base_dir, "glove_6K.txt")


print("正在读取 ct_vectors.txt 中的词...")
ct_words = set()
with open(ct_path, "r", encoding="utf-8") as f:
    for line in f:
        word = line.split(" ", 1)[0]
        ct_words.add(word)
print(f"ct_vectors.txt 中共有 {len(ct_words)} 个不同的词")

# Step 2: 遍历 glove_840B-300d.txt 文件，只保留在 ct_words 中的行
print("正在筛选 glove_840B-300d.txt...")
matched = 0
total = 0
with open(glove_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        word = line.split(" ", 1)[0]
        if word in ct_words:
            fout.write(line)
            matched += 1
        if total % 500000 == 0:
            print(f"  已处理 {total:,} 行，匹配 {matched} 个词...")

print(f"处理完成！共处理 {total:,} 行，匹配 {matched} 个词")
print(f"输出文件: {output_path}")