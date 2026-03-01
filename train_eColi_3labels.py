import os
import argparse
import numpy as np
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import utils
from Model import *
from common import MyDataset_3input

_print_freq = 50

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, dataloader, optim, loss_fn, scheduler, args, e):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    y_pred_list = list()
    y_true_list = list()
    header = 'Training Epoch: [{}]'.format(e)

    for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
        x1 = x1.to(device).float()
        x2 = x2.to(device).float()
        x3 = x3.to(device).float()
        y = y.to(device)

        model_output = model(x1, x2, x3, device=device,
                             first_self_query_dim=args.first_self_query_dim,
                             deep_self=True,
                             deep_self_query_dim=args.deep_self_query_dim,
                             deep_cross_query_dim=args.deep_cross_query_dim)

        loss = loss_fn(model_output, y.long())
        pred = torch.argmax(model_output, dim=1)

        y_pred_list.extend(pred.cpu().numpy())
        y_true_list.extend(y.cpu().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler is not None and (args.scheduler == 'LambdaLR' or args.scheduler == 'Cos'):
            scheduler.step()

        metric_logger.update(loss=loss)

    acc = accuracy_score(y_true_list, y_pred_list)
    # print(f"Train Acc: {acc:.4f}")
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, acc

def evaluate_metrics(model, dataloader, loss_fn, args, e):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    y_true_list = list()
    y_pred_list = list()

    header = 'Evaluation Epoch: [{}]'.format(e)
    with torch.no_grad():
        for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            x3 = x3.to(device).float()
            y = y.to(device)

            model_output = model(x1, x2, x3, device=device,
                                 first_self_query_dim=args.first_self_query_dim,
                                 deep_self=True,
                                 deep_self_query_dim=args.deep_self_query_dim,
                                 deep_cross_query_dim=args.deep_cross_query_dim)

            loss = loss_fn(model_output, y.long())
            pred = torch.argmax(model_output, dim=1)

            y_pred_list.extend(pred.cpu().numpy())
            y_true_list.extend(y.cpu().numpy())

    acc = accuracy_score(y_true_list, y_pred_list)
    precision = precision_score(y_true_list, y_pred_list, average='macro', zero_division=0)
    recall = recall_score(y_true_list, y_pred_list, average='macro', zero_division=0)
    f1 = f1_score(y_true_list, y_pred_list, average='macro', zero_division=0)

    # print(f"Test Acc: {acc:.4f}, Macro F1: {f1:.4f}")
    return acc, precision, recall, f1, loss

def main(lr):
    parser = argparse.ArgumentParser(description='SOLUABLE_Multiclass')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--datadir', type=str, default=r'\datasets\EColi\3labels')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--optim', type=str, default='AdamW', choices=('SGD', 'Adam', 'AdamW'))
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--weight_decay', type=float, default=0.0003)
    parser.add_argument('--warmup', type=int, default=150)
    parser.add_argument('--scheduler', type=str, default='LambdaLR', choices=('None', 'StepLR', 'LambdaLR'))
    parser.add_argument('--letter_emb_size', type=int, default=32)

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

    parser.add_argument('--seed', type=int, default=2024, help='Random seed')

    args = parser.parse_args()

    set_seed(args.seed)

    x_dir_path3 = args.datadir+"\\x_EColi_esm2_dataset.csv"
    x_dir_path2 = args.datadir+"\\x_EColi_protT5_dataset.csv"
    x_dir_path1 = args.datadir+"\\x_EColi_unirep_dataset.csv"
    y_dir_path = args.datadir+"\\y_EColi_esm2_dataset.csv"

    x_dataset1 = np.loadtxt(x_dir_path1, delimiter=",", dtype=np.float32)
    x_dataset2 = np.loadtxt(x_dir_path2, delimiter=",", dtype=np.float32)
    x_dataset3 = np.loadtxt(x_dir_path3, delimiter=",", dtype=np.float32)
    y_dataset = np.loadtxt(y_dir_path, delimiter=",", dtype=np.float32)

    model_name = 'ProtSATT'
    res_dir = rf'results\EColi_3labels'
    os.makedirs(res_dir, mode=0o777, exist_ok=True)
    res_model_dir = f'{res_dir}\\save_models'
    os.makedirs(res_model_dir, mode=0o777, exist_ok=True)

    fold_results = {'acc': [], 'precision': [], 'recall': [], 'f1': []}

    Cross_Fold = 10
    skf = StratifiedKFold(n_splits=Cross_Fold, shuffle=True, random_state=args.seed)

    if y_dataset.ndim > 1:
        y_labels = np.argmax(y_dataset, axis=1)
    else:
        y_labels = y_dataset

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_dataset1, y_labels), 1):
        print(f"\n========== Fold {fold_idx}/{Cross_Fold} ==========")

        x_train1, x_val1 = x_dataset1[train_idx], x_dataset1[val_idx]
        x_train2, x_val2 = x_dataset2[train_idx], x_dataset2[val_idx]
        x_train3, x_val3 = x_dataset3[train_idx], x_dataset3[val_idx]
        y_train, y_val = y_labels[train_idx], y_labels[val_idx]

        if model_name == 'ProtSATT':
            model = ProtSATT(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep,
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
            ).to(device)

        def lambda_lr(s):
            warm_up = args.warmup
            s += 1
            return (args.letter_emb_size ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

        if args.optim == 'Adam':
            optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        elif args.optim == 'AdamW':
            optim = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        else:
            optim = SGD(model.parameters(), lr=args.lr, momentum=0.9)

        if args.scheduler == 'StepLR':
            scheduler = StepLR(optim, step_size=30, gamma=0.5)
        elif args.scheduler == 'LambdaLR':
            scheduler = LambdaLR(optim, lambda_lr)
        else:
            scheduler = None

        loss_fn = torch.nn.CrossEntropyLoss()

        dataset_train = MyDataset_3input(x1=x_train1, x2=x_train2, x3=x_train3, y=y_train)
        dataset_val = MyDataset_3input(x1=x_val1, x2=x_val2, x3=x_val3, y=y_val)

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        best_test_f1 = 0
        best_metrics_in_fold = {'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        best_epoch = 0

        epoch_list, train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], [], []

        for e in range(args.epoch):
            train_loss, train_acc = train(model, dataloader_train, optim, loss_fn, scheduler, args, e)

            epoch_list.append(e+1)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

            if scheduler is not None and args.scheduler == 'StepLR':
                scheduler.step()

            val_acc, val_precision, val_recall, val_f1, val_loss = evaluate_metrics(model, dataloader_val, loss_fn, args, e)
            print(f"Fold {fold_idx} Epoch {e} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss.item():.4f} F1: {val_f1:.4f}")

            val_loss_list.append(val_loss.cpu())
            val_acc_list.append(val_acc)

            if val_f1 > best_test_f1:
                best_test_f1 = val_f1
                best_epoch = e
                best_metrics_in_fold = {'acc': val_acc, 'precision': val_precision, 'recall': val_recall, 'f1': val_f1}
                torch.save(model.state_dict(), f'{res_model_dir}/best_model_fold{fold_idx}.pth')

        plt.figure()
        plt.plot(epoch_list, train_loss_list, label='train_loss')
        plt.plot(epoch_list, val_loss_list, label='val_loss')
        plt.plot(epoch_list, train_acc_list, label='train_acc')
        plt.plot(epoch_list, val_acc_list, label='val_acc')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title(f"Fold {fold_idx} Metric Figure")
        # plt.savefig(rf'{res_dir}/metrics_fold{fold_idx}.png')
        plt.close()

        print(f"==== Fold {fold_idx} Best (Epoch {best_epoch}) ====")
        print(f"ACC: {best_metrics_in_fold['acc']:.4f}, F1: {best_metrics_in_fold['f1']:.4f}")

        for k in fold_results:
            fold_results[k].append(best_metrics_in_fold[k])

        with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
            f.write(f'========== Fold {fold_idx} ============\n')
            f.write(f'Best Epoch: {best_epoch}\n')
            f.write(f'ACC: {best_metrics_in_fold["acc"]:.4f}\n')
            f.write(f'Macro Precision: {best_metrics_in_fold["precision"]:.4f}\n')
            f.write(f'Macro Recall: {best_metrics_in_fold["recall"]:.4f}\n')
            f.write(f'Macro F1: {best_metrics_in_fold["f1"]:.4f}\n\n')

    final_stats = {}
    for k, v in fold_results.items():
        final_stats[f'{k}_mean'] = np.mean(v)
        final_stats[f'{k}_std'] = np.std(v)

    print("\n================ Final Cross-Validation Results =================")
    print(f"ACC: {final_stats['acc_mean']:.4f} ± {final_stats['acc_std']:.4f}")
    print(f"Macro Precision: {final_stats['precision_mean']:.4f} ± {final_stats['precision_std']:.4f}")
    print(f"Macro Recall: {final_stats['recall_mean']:.4f} ± {final_stats['recall_std']:.4f}")
    print(f"Macro F1: {final_stats['f1_mean']:.4f} ± {final_stats['f1_std']:.4f}")

    with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
        f.writelines('\n========== Final Average Results ==========\n')
        f.writelines(f'Seed: {args.seed}\n')
        f.writelines(f'ACC: {final_stats["acc_mean"]:.4f} ± {final_stats["acc_std"]:.4f}\n')
        f.writelines(f'Macro Precision: {final_stats["precision_mean"]:.4f} ± {final_stats["precision_std"]:.4f}\n')
        f.writelines(f'Macro Recall: {final_stats["recall_mean"]:.4f} ± {final_stats["recall_std"]:.4f}\n')
        f.writelines(f'Macro F1: {final_stats["f1_mean"]:.4f} ± {final_stats["f1_std"]:.4f}\n')

if __name__ == '__main__':
    for i in range(1,11):
        main(i/1000)