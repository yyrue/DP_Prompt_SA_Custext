"""
Baseline 实验：用原始数据（不做隐私化替换）微调 BERT
用于对比 CusText 隐私化后的效用损失
"""
import datetime
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from utils import load_data, Bert_dataset
from logger import get_logger

def get_baseline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qnli")
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="./trained_model")
    parser.add_argument("--log_path", type=str, default="./log")
    return parser

parser = get_baseline_parser()
args = parser.parse_args()

logger = get_logger(log_file=f"baseline_{args.dataset}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")
logger.info(f"Baseline experiment, args: {args}")

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 加载原始数据（不做任何隐私化处理）
    train_data, test_data = load_data(args.dataset)
    logger.info(f"train_data:{len(train_data)}, test_data:{len(test_data)}")

    # 构建 DataLoader
    train_dataset = Bert_dataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Bert_dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        args.model_type,
        num_labels=args.num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.epochs
    )

    # 设备
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model.cuda()

    # 模型保存路径
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_name = f"baseline_{args.dataset}_model.pkl"

    # ========== 训练 ==========
    model.train()
    best_score = 0
    num_training_steps = args.epochs * len(train_loader)
    global_step = 0

    logger.info(f"Training baseline model for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        tr_loss = 0
        tr_examples = 0
        import time
        from tqdm import tqdm
        epoch_start = time.time()

        for batch_data in tqdm(train_loader):
            batch_data = tuple(data.to(device) for data in batch_data)
            inputs_ids, inputs_masks, token_type_ids, inputs_labels = batch_data

            optimizer.zero_grad()
            outputs = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks, labels=inputs_labels)
            loss = outputs['loss']
            tr_loss += loss.item()
            tr_examples += inputs_ids.size(0)

            if args.log_steps and global_step % args.log_steps == 0:
                logger.info(f"[Train] epoch:{epoch+1}/{args.epochs}, step: {global_step}/{num_training_steps}, loss:{loss.item():.4f}")

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # 验证
            if args.eval_steps > 0 and global_step != 0 and \
                (global_step % args.eval_steps == 0 or global_step == (num_training_steps - 1)):
                model.eval()
                y_list, y_hat_list = [], []
                for batch in tqdm(test_loader):
                    batch = tuple(data.to(device) for data in batch)
                    ids, masks, tids, labels = batch
                    with torch.no_grad():
                        preds = model(ids, token_type_ids=None, attention_mask=masks)
                    y_list.extend(labels.detach().cpu().numpy())
                    y_hat_list.extend(preds['logits'].detach().cpu().numpy())
                val_acc = np.sum(np.argmax(y_hat_list, axis=1) == np.array(y_list)) / len(y_list)
                logger.info(f"[Evaluate] epoch:{epoch+1}/{args.epochs}, step: {global_step}/{num_training_steps}, val_acc:{val_acc:.5f}")

                if val_acc > best_score:
                    torch.save(model.state_dict(), f'{save_path}/{model_save_name}')
                    logger.info(f"[Evaluate] best accuracy updated: {best_score:.5f} -> {val_acc:.5f}")
                    best_score = val_acc

                model.train()

        epoch_time = time.time() - epoch_start
        logger.info(f"[Epoch {epoch+1}] train_loss = {tr_loss/tr_examples:.4f}, [{epoch_time:.1f}s]")

    # ========== 测试 ==========
    model.load_state_dict(torch.load(f'{save_path}/{model_save_name}'))
    model.eval()
    y_list, y_hat_list = [], []
    for batch in tqdm(test_loader):
        batch = tuple(data.to(device) for data in batch)
        ids, masks, tids, labels = batch
        with torch.no_grad():
            preds = model(ids, token_type_ids=None, attention_mask=masks)
        y_list.extend(labels.detach().cpu().numpy())
        y_hat_list.extend(preds['logits'].detach().cpu().numpy())

    test_acc = np.sum(np.argmax(y_hat_list, axis=1) == np.array(y_list)) / len(y_list)
    logger.info(f"Baseline test acc = {test_acc:.4f}.")
    print(f"\n===== Baseline Test Accuracy: {test_acc:.4f} =====")