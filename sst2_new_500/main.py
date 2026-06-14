import datetime
from utils import *
import torch
import pandas as pd
import torch.nn as nn
from logger import get_logger
from training import Trainer
from transformers import AdamW,get_linear_schedule_with_warmup,BertModel,AutoConfig
from args import *

parser = get_parser()
args = parser.parse_args()

logger = get_logger(log_file=f"{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_eps_{args.eps}_top_{args.top_k}_seed_{args.seed}_save_{args.save_stop_words}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")
logger.info(f"{args.dataset}, args: {args}")

if __name__ == "__main__":
        parser = get_parser()
        args = parser.parse_args()

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # 脱敏数据集的保存/加载路径
        priv_dir = (
                f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
                f"/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}"
                f"_save_stop_words_{args.save_stop_words}_seed_{args.seed}"
        )
        priv_train_path = os.path.join(priv_dir, "train.tsv")
        priv_test_path = os.path.join(priv_dir, "test.tsv")

        if args.use_saved_private_data:
                # 模式1：从磁盘加载已保存的脱敏数据集
                if not os.path.exists(priv_train_path) or not os.path.exists(priv_test_path):
                        raise FileNotFoundError(
                                f"未找到已保存的脱敏数据集，请先运行 generate_private_data.py 生成：\n"
                                f"  python generate_private_data.py --seed {args.seed} --eps {args.eps}\n"
                                f"  期望路径: {priv_train_path}, {priv_test_path}"
                        )
                print(f"从磁盘加载已保存的脱敏数据集: {priv_dir}")
                train_data = pd.read_csv(priv_train_path, sep='\t', keep_default_na=False).reset_index(drop=True)
                test_data = pd.read_csv(priv_test_path, sep='\t', keep_default_na=False).reset_index(drop=True)
                # 填充空值：tsv 中空行会被 pandas 解析为 NaN，tokenizer 无法处理
                train_data['sentence'] = train_data['sentence'].fillna('')
                test_data['sentence'] = test_data['sentence'].fillna('')
                # dev_data 不脱敏，仍从原始数据加载
                _, dev_data, _ = load_data(args.dataset)
                dev_data['sentence'] = dev_data['sentence'].fillna('')
        else:
                # 模式2（原有模式）：在内存中生成脱敏数据
                #返回的dataframe为train_data,dev_data,test_data
                train_data,dev_data,test_data = load_data(args.dataset)

                # 第一步：构建映射表 sim_word_dict（耗时按 top_k 缓存，与 eps 无关）
                sim_word_dict_path = f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"
                sim_dist_dict_path = f"./sim_dist_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"

                if os.path.exists(sim_word_dict_path) and os.path.exists(sim_dist_dict_path):
                        print("加载已缓存的 sim_word_dict 和 sim_dist_dict...")
                        with open(sim_word_dict_path, 'r') as dic:
                                sim_word_dict = json.load(dic)
                        with open(sim_dist_dict_path, 'r') as dic:
                                sim_dist_dict = json.load(dic)
                else:
                        print("构建 sim_word_dict 和 sim_dist_dict（首次运行，耗时较长）...")
                        sim_word_dict, sim_dist_dict = build_sim_word_dict(top_k = args.top_k)

                # 第二步：计算概率分布 p_dict（很快，按 eps+top_k 缓存）
                p_dict_path = f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt"

                if os.path.exists(p_dict_path):
                        print("加载已缓存的 p_dict...")
                        with open(p_dict_path, 'r') as dic:
                                p_dict = json.load(dic)
                else:
                        print(f"计算 p_dict（eps={args.eps}）...")
                        p_dict = compute_p_dict(sim_dist_dict, eps = args.eps)
                        
                if args.privatization_strategy == "s1":
                        train_data = generate_new_sents_s1(df = train_data ,sim_word_dict = sim_word_dict ,p_dict = p_dict ,save_stop_words = args.save_stop_words)
                        test_data = generate_new_sents_s1(df = test_data ,sim_word_dict = sim_word_dict ,p_dict = p_dict ,save_stop_words = args.save_stop_words,type="test")

        train_dataset = Bert_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_dataset = Bert_dataset(dev_data)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = Bert_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        logger.info(f"train_data:{len(train_data)},dev_data:{len(dev_data)},test_data:{len(test_data)}")

        model = BertForSequenceClassification.from_pretrained(
        args.model_type,
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False)

        optimizer = AdamW(model.parameters(),lr=args.lr,eps=1e-8)  
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=len(train_loader)*args.epochs)
        trainer = Trainer(
                model,
                scheduler,
                optimizer,
                args.epochs,
                args.log_steps,
                args.eval_steps,
                args.use_cuda,
                logger
                )

        trainer.train(train_loader, test_loader)

        # evaluate test dataset #
        acc = trainer.predict(test_loader)
        logger.info(f"test acc = {acc:.4f}.")