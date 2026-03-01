import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

from Model import ProtSATT, multi_layer_attention_no_self, multi_layer_attention_no_cross, multi_layer_attention_2input, multi_layer_attention_1input
from common import MyDataset_3input, split_train_test_fold

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=68):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "PRECISION": precision_score(y_true, y_pred, zero_division=0),
        "RECALL": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "SPECIFICITY": specificity
    }

def train_epoch(model, dataloader, optim, loss_fn, scheduler, args):
    model.train()
    total_loss = 0
    for x1, x2, x3, y in dataloader:
        x1, x2, x3, y = x1.to(device).float(), x2.to(device).float(), x3.to(device).float(), y.to(device).long()
        optim.zero_grad()

        out = model(x1, x2, x3, device=device,
                    first_self_query_dim=args.first_self_query_dim, deep_self=True,
                    deep_self_query_dim=args.deep_self_query_dim, deep_cross_query_dim=args.deep_cross_query_dim)

        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, loss_fn, args):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x1, x2, x3, y in dataloader:
            x1, x2, x3, y = x1.to(device).float(), x2.to(device).float(), x3.to(device).float(), y.to(device).long()

            out = model(x1, x2, x3, device=device,
                        first_self_query_dim=args.first_self_query_dim, deep_self=True,
                        deep_self_query_dim=args.deep_self_query_dim, deep_cross_query_dim=args.deep_cross_query_dim)

            loss = loss_fn(out, y)
            total_loss += loss.item()

            prob = torch.softmax(out, dim=1)
            preds = torch.argmax(prob, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(prob[:, 1].cpu().numpy()) 

    metrics = calculate_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='ProtSATT 10-Fold Training')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--datadir', type=str, default= r'\datasets\EColi\2labels\\')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.004) # 0.001  AdamW 0.00001 0.004
    parser.add_argument('--warmup', type=int, default=150)
    parser.add_argument('--weight_decay', type=float, default=0.0003) # 0.01 0.0003
    parser.add_argument('--seed', type=int, default=68)

    parser.add_argument('--model_name', type=str, default='ProtSATT')
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
    parser.add_argument('--out_scores', type=int, default=2)

    args = parser.parse_args()
    set_seed(args.seed)

    res_dir = os.path.join(r'results\EColi\2labels\Ecoli_train_10CV', args.model_name)
    res_model_dir = os.path.join(res_dir, 'save_models')
    os.makedirs(res_model_dir, exist_ok=True)

    print(f"[INFO] Loading data from {args.datadir}...")
    x_dataset1 = np.loadtxt(os.path.join(args.datadir, "x_unirep_dataset.csv"), delimiter=",", dtype="float")
    x_dataset2 = np.loadtxt(os.path.join(args.datadir, "x_protT5_dataset.csv"), delimiter=",", dtype="float")
    x_dataset3 = np.loadtxt(os.path.join(args.datadir, "x_esm2_dataset.csv"), delimiter=",", dtype="float")
    y_dataset = np.loadtxt(os.path.join(args.datadir, "y_dataset.csv"), delimiter=",", dtype="float")

    # print("[INFO] Filtering data for Binary Classification (0 vs 2)...")
    # valid_idx = (y_dataset == 0) | (y_dataset == 2)
    # x_dataset1, x_dataset2, x_dataset3 = x_dataset1[valid_idx], x_dataset2[valid_idx], x_dataset3[valid_idx]
    # y_dataset = y_dataset[valid_idx]
    # y_dataset[y_dataset == 2] = 1 
    # print(f"[INFO] Total Samples after filtering: {len(y_dataset)}")

    Cross_Fold = 10
    all_fold_test_metrics = []

    for i in range(1, Cross_Fold + 1):
        print(f"\n{'='*25} Fold {i}/{Cross_Fold} {'='*25}")

        x_train_val1, x_train_val2, x_train_val3, y_train_val, x_test1, x_test2, x_test3, y_test = split_train_test_fold(
            x_dataset1, x_dataset2, x_dataset3, y_dataset, Cross_Fold, i
        )

        # 10% val，90% train
        indices = np.arange(len(y_train_val))
        tr_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=args.seed, stratify=y_train_val)

        x_tr1, x_tr2, x_tr3, y_tr = x_train_val1[tr_idx], x_train_val2[tr_idx], x_train_val3[tr_idx], y_train_val[tr_idx]
        x_v1, x_v2, x_v3, y_v = x_train_val1[val_idx], x_train_val2[val_idx], x_train_val3[val_idx], y_train_val[val_idx]

        print(f"  - Train samples: {len(y_tr)}, Val samples: {len(y_v)}, Test samples: {len(y_test)}")

        # 3. 构建 DataLoader
        train_loader = DataLoader(MyDataset_3input(x_tr1, x_tr2, x_tr3, y_tr), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(MyDataset_3input(x_v1, x_v2, x_v3, y_v), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(MyDataset_3input(x_test1, x_test2, x_test3, y_test), batch_size=args.batch_size, shuffle=False)

        model_class = globals()[args.model_name]
        model = model_class(
            dropout=args.dropout, first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim,
            first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
            self_deep=args.self_deep, deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim,
            deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
            deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head,
            deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef, out_scores=args.out_scores,
        ).to(device)

        # optimizer
        optim = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

        def lambda_lr(s):
            warm_up = args.warmup
            s += 1
            return (32 ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
        scheduler = LambdaLR(optim, lambda_lr)
        # scheduler = CosineAnnealingLR(optim, T_max=args.epoch)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = -1
        best_epoch = 0
        save_path = os.path.join(res_model_dir, f'best_model_fold{i}.pth')

        for e in range(1, args.epoch + 1):
            train_loss = train_epoch(model, train_loader, optim, loss_fn, scheduler, args)
            val_metrics = eval_epoch(model, val_loader, loss_fn, args)

            if e % 20 == 0 or e == 1:
                print(f"Epoch {e:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['ROC_AUC']:.4f}")

            if val_metrics['ACC'] > best_val_acc:
                best_val_acc = val_metrics['ACC']
                best_epoch = e
                torch.save(model.state_dict(), save_path)

        print(f"-> Fold {i} Best Val ACC: {best_val_acc:.4f} at Epoch {best_epoch}")

        # Test
        model.load_state_dict(torch.load(save_path, map_location=device))
        test_metrics_dict = eval_epoch(model, test_loader, loss_fn, args)

        test_metrics_dict['Fold'] = i
        all_fold_test_metrics.append(test_metrics_dict)

        print(f"-> Fold {i} TEST Metrics: AUC={test_metrics_dict['ROC_AUC']:.4f}, MCC={test_metrics_dict['MCC']:.4f}, ACC={test_metrics_dict['ACC']:.4f}")


    df_res = pd.DataFrame(all_fold_test_metrics)

    metrics_to_compute = df_res.drop(columns=['Fold', 'loss'])
    mean_metrics = metrics_to_compute.mean()
    std_metrics = metrics_to_compute.std()

    report = [f"========== 10-Fold Cross Validation Summary ==========\n"]
    for metric in mean_metrics.index:
        report.append(f"{metric:<15}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}\n")

    report_str = "".join(report)
    print("\n" + report_str)

    with open(os.path.join(res_dir, 'final_metrics_report.txt'), 'w') as f:
        f.write(report_str)
    df_res.to_csv(os.path.join(res_dir, 'all_folds_test_metrics.csv'), index=False)
    print(f"[Done] All results perfectly saved to {res_dir}")

if __name__ == '__main__':
    main()