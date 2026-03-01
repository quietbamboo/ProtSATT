import os
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 引入你自己定义的模块
import utils
from Model import ProtSATT
from common import MyDataset_3input

# ================= 1. 设置随机种子 (必须与训练时完全一致) =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ================= 2. 推理评估函数 =================
def evaluate_fold(model, dataloader, args):
    model.eval()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x1, x2, x3, y in dataloader:
            # 数据移动与类型转换 (与训练代码保持一致)
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            x3 = x3.to(device).float()
            y = y.to(device)

            model_output = model(x1, x2, x3, device=device,
                                 first_self_query_dim=args.first_self_query_dim,
                                 deep_self=True,
                                 deep_self_query_dim=args.deep_self_query_dim,
                                 deep_cross_query_dim=args.deep_cross_query_dim)

            pred = torch.argmax(model_output, dim=1)

            y_pred_list.extend(pred.cpu().numpy())
            y_true_list.extend(y.cpu().numpy())

    # 计算各项指标
    acc = accuracy_score(y_true_list, y_pred_list)
    precision = precision_score(y_true_list, y_pred_list, average='macro', zero_division=0)
    recall = recall_score(y_true_list, y_pred_list, average='macro', zero_division=0)
    f1 = f1_score(y_true_list, y_pred_list, average='macro', zero_division=0)

    return acc, precision, recall, f1

# ================= 3. 主程序 =================
def run_inference(lr):
    parser = argparse.ArgumentParser(description='SOLUABLE_Multiclass_Inference')
    parser.add_argument('--datadir', type=str, default=r'H:\毕业项目\ProtSATT\整理\datasets\EColi\E_Coli_3labels')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=lr)

    # 模型参数必须与训练时完全一致，才能成功加载权重
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--first_self_query_dim', type=int, default=32)
    parser.add_argument('--first_self_return_dim', type=int, default=512)
    parser.add_argument('--first_self_num_head', type=int, default=1)
    parser.add_argument('--first_self_dropout', type=int, default=0.15)
    parser.add_argument('--first_self_residual_coef', type=float, default=0)
    parser.add_argument('--self_deep', type=int, default=1)
    parser.add_argument('--deep_self_query_dim', type=int, default=16)
    parser.add_argument('--deep_self_return_dim', type=int, default=128)
    parser.add_argument('--deep_self_num_head', type=int, default=1)
    parser.add_argument('--deep_self_dropout', type=float, default=0.15)
    parser.add_argument('--deep_self_residual_coef', type=float, default=0.5)
    parser.add_argument('--deep_cross_query_dim', type=int, default=8)
    parser.add_argument('--deep_cross_return_dim', type=int, default=32)
    parser.add_argument('--deep_cross_num_head', type=int, default=1)
    parser.add_argument('--deep_cross_dropout', type=int, default=0.15)
    parser.add_argument('--deep_cross_residual_coef', type=float, default=0)
    parser.add_argument('--out_scores', type=int, default=3)

    parser.add_argument('--seed', type=int, default=2024, help='Random seed MUST match training')

    args = parser.parse_args()

    # 1. 强制使用训练时的 seed 复原划分
    set_seed(args.seed)

    # 2. 加载数据
    x_dir_path1 = args.datadir + "\\x_EColi_unirep_dataset.csv"
    x_dir_path2 = args.datadir + "\\x_EColi_protT5_dataset.csv"
    x_dir_path3 = args.datadir + "\\x_EColi_esm2_dataset.csv"
    y_dir_path = args.datadir + "\\y_EColi_esm2_dataset.csv"

    x_dataset1 = np.loadtxt(x_dir_path1, delimiter=",", dtype=np.float32)
    x_dataset2 = np.loadtxt(x_dir_path2, delimiter=",", dtype=np.float32)
    x_dataset3 = np.loadtxt(x_dir_path3, delimiter=",", dtype=np.float32)
    y_dataset = np.loadtxt(y_dir_path, delimiter=",", dtype=np.float32)

    if y_dataset.ndim > 1:
        y_labels = np.argmax(y_dataset, axis=1)
    else:
        y_labels = y_dataset

    # 3. 确定模型权重保存路径
    res_dir = rf'H:\毕业项目\ProtSATT\整理\EColi\3labels\EColi_3labels_NEW_seed2024_lr0.008_best'
    res_model_dir = f'{res_dir}\\save_models'

    print(f"\n================ Evaluating Models for LR={args.lr}, SEED={args.seed} ================")

    if not os.path.exists(res_model_dir):
        print(f"Error: 找不到模型目录 {res_model_dir}")
        return

    # 4. 初始化存放 10 折结果的字典
    fold_results = {'acc': [], 'precision': [], 'recall': [], 'f1': []}

    Cross_Fold = 10
    skf = StratifiedKFold(n_splits=Cross_Fold, shuffle=True, random_state=args.seed)

    # 5. 遍历 10 折
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_dataset1, y_labels), 1):

        # 我们只需要验证集数据用于推理评估
        x_val1 = x_dataset1[val_idx]
        x_val2 = x_dataset2[val_idx]
        x_val3 = x_dataset3[val_idx]
        y_val = y_labels[val_idx]

        # 初始化模型架构
        model = ProtSATT(
            dropout=args.dropout,
            first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
            self_deep=args.self_deep,
            deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
            deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
            out_scores=args.out_scores,
        ).to(device)

        # 加载该折保存的最佳模型权重
        model_path = f'{res_model_dir}/best_model_fold{fold_idx}.pth'
        if not os.path.exists(model_path):
            print(f"Fold {fold_idx}: 找不到权重文件 {model_path}，跳过该折。")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))

        # 构建 Dataloader
        dataset_val = MyDataset_3input(x1=x_val1, x2=x_val2, x3=x_val3, y=y_val)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

        # 执行推理
        acc, precision, recall, f1 = evaluate_fold(model, dataloader_val, args)

        print(f"Fold {fold_idx:2d} -> ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # 记录指标
        fold_results['acc'].append(acc)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)

    # 6. 计算 10折 的平均值(mean) 和 标准差(std) (即方差的平方根，反映指标波动大小)
    if len(fold_results['acc']) > 0:
        print("\n================ Final 10-Fold CV Evaluation Results ================")
        print(f"Model Path: {res_dir}")
        print(f"ACC:       {np.mean(fold_results['acc']):.4f} ± {np.std(fold_results['acc']):.4f}")
        print(f"Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
        print(f"Recall:    {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")
        print(f"F1 Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")

        # 可选：将结果写入到文本文件中
        with open(rf'H:\毕业项目\ProtSATT\整理\EColi\3labels\predict\inference_results.txt', 'w') as f:
            f.write("========== 10-Fold Inference Results ==========\n")
            f.write(f"ACC:       {np.mean(fold_results['acc']):.4f} ± {np.std(fold_results['acc']):.4f}\n")
            f.write(f"Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}\n")
            f.write(f"Recall:    {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}\n")
            f.write(f"F1 Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}\n")
    else:
        print("未成功加载任何模型，无法计算最终指标。")

if __name__ == '__main__':
    # 因为你的训练代码中遍历了 lr (1/1000 到 10/1000)
    # 相应的，你可以遍历验证这些学习率训练出来的模型
    # for i in range(1, 11):
    #     lr = i / 1000
    run_inference(0.003)