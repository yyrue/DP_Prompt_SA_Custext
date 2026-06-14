import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # ---training params---
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--save_path", type=str, default="./trained_model") 
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len",type=int,default=128)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--num_workers",type=int,default=os.cpu_count())
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--log_steps",type=int,default=50) 
    parser.add_argument("--eval_steps",type=int,default=50)

    # ---CusText params---
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--embedding_type", type=str, default="ct_vectors")
    parser.add_argument("--mapping_strategy", type=str, default="paper",
                        choices=["paper", "conservative", "aggressive"],
                        help="paper: 对齐论文Algorithm1的data-independent映射; "
                             "conservative/aggressive: 原有策略")
    parser.add_argument("--privatization_strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", action="store_true", default=False)
    parser.add_argument("--use_saved_private_data", action="store_true", default=False,
                        help="使用磁盘上已保存的脱敏数据集进行训练，而非在内存中重新生成")

    # ---Sample Amplification params---
    parser.add_argument("--eps_low", type=float, default=0.0,
                        help="SA 混合的低隐私预算源（默认 0.0，即完全随机）")
    parser.add_argument("--eps_high", type=float, default=32.0,
                        help="SA 混合的高隐私预算源")
    parser.add_argument("--mixed_data_dir", type=str,
                        default="./privatized_dataset_mixed/glove_840B-300d/conservative",
                        help="混合数据的根目录")
    return parser
