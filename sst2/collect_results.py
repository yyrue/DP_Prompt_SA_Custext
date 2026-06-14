#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果收集脚本
功能：从日志文件中提取测试准确率，汇总到 CSV 文件
"""

import os
import re
import pandas as pd
from pathlib import Path

def extract_test_acc_from_log(log_file):
    """
    从日志文件中提取测试准确率
    匹配模式: "test acc = 0.xxxx" 或 "test acc = 0.xxxx."
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 匹配 "test acc = 0.xxxx"，只匹配数字（含一个小数点）
            match = re.search(r'test acc = (\d+\.\d+)', content)
            if match:
                return float(match.group(1))
            else:
                return None
    except Exception as e:
        print(f"读取文件 {log_file} 时出错: {e}")
        return None

def collect_results(result_dir='./experiment_results', log_dir='./log'):
    """
    收集所有实验结果
    """
    results = []
    
    # 定义实验参数
    eps_values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,34]
    seeds = [42, 43, 44, 45, 46]
    
    print("开始收集实验结果...")
    print("=" * 60)
    
    # 遍历所有 eps 和 seed 组合
    for eps in eps_values:
        for seed in seeds:
            # 尝试从 experiment_results 目录读取
            log_file = os.path.join(result_dir, f'eps_{eps}_seed_{seed}.log')
            
            if os.path.exists(log_file):
                test_acc = extract_test_acc_from_log(log_file)
                
                if test_acc is not None:
                    results.append({
                        'eps': eps,
                        'seed': seed,
                        'test_acc': test_acc
                    })
                    print(f"✓ eps={eps}, seed={seed}: test_acc={test_acc:.4f}")
                else:
                    print(f"✗ eps={eps}, seed={seed}: 未找到测试准确率")
            else:
                print(f"✗ eps={eps}, seed={seed}: 日志文件不存在")
    
    print("=" * 60)
    
    # 如果没有收集到结果，尝试从 log 目录读取
    if len(results) == 0:
        print("\n未在 experiment_results 目录找到结果，尝试从 log 目录读取...")
        print("=" * 60)
        
        if os.path.exists(log_dir):
            log_files = list(Path(log_dir).glob('*.txt'))
            for log_file in log_files:
                # 尝试从文件名提取 eps 和 seed
                filename = log_file.name
                eps_match = re.search(r'eps_([\d.]+)', filename)
                
                if eps_match:
                    eps = float(eps_match.group(1))
                    test_acc = extract_test_acc_from_log(log_file)
                    
                    if test_acc is not None:
                        # 尝试从文件名或内容提取 seed（如果有的话）
                        results.append({
                            'eps': eps,
                            'seed': 'unknown',
                            'test_acc': test_acc,
                            'log_file': filename
                        })
                        print(f"✓ {filename}: eps={eps}, test_acc={test_acc:.4f}")
    
    # 转换为 DataFrame
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        # 按 eps 和 seed 排序
        df = df.sort_values(['eps', 'seed'])
        
        # 保存到 CSV
        output_file = os.path.join(result_dir, 'results_summary.csv')
        df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
        
        # 计算统计信息
        print("\n" + "=" * 60)
        print("统计摘要:")
        print("=" * 60)
        
        if 'seed' in df.columns and df['seed'].dtype != 'object':
            stats = df.groupby('eps')['test_acc'].agg(['mean', 'std', 'min', 'max', 'count'])
            stats.columns = ['平均值', '标准差', '最小值', '最大值', '运行次数']
            print(stats.to_string())
            
            # 保存统计信息
            stats_file = os.path.join(result_dir, 'results_statistics.csv')
            stats.to_csv(stats_file)
            print(f"\n统计信息已保存到: {stats_file}")
        else:
            print(df.to_string(index=False))
        
        return df
    else:
        print("\n未找到任何实验结果！")
        return None

if __name__ == "__main__":
    # 创建结果目录
    result_dir = './experiment_results'
    os.makedirs(result_dir, exist_ok=True)
    
    # 收集结果
    df = collect_results(result_dir=result_dir)
