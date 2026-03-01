import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    average_precision_score
)
from scipy.stats import pearsonr, spearmanr

from Model import ProtSATT
from common import MyDataset_3input
import utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        # x: pred, y: true
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return 1 - cost

def calculate_metrics(y_true, y_pred, thr=0.5):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson, _ = pearsonr(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)

    y_true_bin = (y_true >= thr).astype(int)
    y_pred_bin = (y_pred >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true_bin)) == 2:
        roc_auc = roc_auc_score(y_true_bin, y_pred)
        pr_auc = average_precision_score(y_true_bin, y_pred)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
    else:
        roc_auc, pr_auc, mcc = 0.0, 0.0, 0.0

    return {
        "R2": r2, "MAE": mae, "RMSE": rmse,
        "Pearson": pearson, "Spearman": spearman,
        "ACC": accuracy_score(y_true_bin, y_pred_bin),
        "PRECISION": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "RECALL": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "F1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "MCC": mcc,
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "SPECIFICITY": specificity,
        "CM": [tn, fp, fn, tp]
    }

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x1, x2, x3, y in tqdm(loader, desc="Training", leave=False):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x1, x2, x3, device=device,
                       first_self_query_dim=32, deep_self=False,
                       deep_self_query_dim=16, deep_cross_query_dim=8)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_epoch_combined(model, loader, optimizer, criterion_mse, criterion_pcc):
    model.train()
    total_loss = 0
    for x1, x2, x3, y in tqdm(loader, desc="Training", leave=False):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x1, x2, x3, device=device,
                       first_self_query_dim=32, deep_self=False,
                       deep_self_query_dim=16, deep_cross_query_dim=8)
        # 组合 Loss
        loss = criterion_mse(output, y) + 0.1 * criterion_pcc(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x1, x2, x3, y in loader:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            output = model(x1, x2, x3, device=device,
                           first_self_query_dim=32, deep_self=False,
                           deep_self_query_dim=16, deep_cross_query_dim=8)
            loss = criterion(output, y)
            total_loss += loss.item()
            y_pred.extend(output.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    return total_loss / len(loader), np.array(y_true), np.array(y_pred)

def validate_combined(model, loader, criterion_mse, criterion_pcc):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x1, x2, x3, y in loader:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            output = model(x1, x2, x3, device=device,
                           first_self_query_dim=32, deep_self=False,
                           deep_self_query_dim=16, deep_cross_query_dim=8)
            loss = criterion_mse(output, y) + 0.1 * criterion_pcc(output, y)
            total_loss += loss.item()
            y_pred.extend(output.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    return total_loss / len(loader), np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser(description='ProtSATT Training on TR')

    parser.add_argument('--datadir', type=str, default='\datasets\Tc-Riboswitches\\')
    parser.add_argument('--out_dir', type=str, default=r'\results\results_tr')

    parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE', 'SmoothL1', 'Huber', 'MSE_PCC'], help='Loss function type')

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.003)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--thr', type=float, default=0.5)

    parser.add_argument('--dropout', type=float, default=0.09)
    parser.add_argument('--first_self_query_dim', type=int, default=32)
    parser.add_argument('--first_self_return_dim', type=int, default=512)
    parser.add_argument('--first_self_num_head', type=int, default=1)
    parser.add_argument('--first_self_dropout', type=int, default=0.15)
    parser.add_argument('--first_self_residual_coef', type=float, default=0.2)

    parser.add_argument('--self_deep', type=int, default=1)
    parser.add_argument('--deep_self_query_dim', type=int, default=16)
    parser.add_argument('--deep_self_return_dim', type=int, default=128)
    parser.add_argument('--deep_self_num_head', type=int, default=1)
    parser.add_argument('--deep_self_dropout', type=float, default=0.15)
    parser.add_argument('--deep_self_residual_coef', type=float, default=0.1)

    parser.add_argument('--deep_cross_query_dim' , type=int, default=8)
    parser.add_argument('--deep_cross_return_dim', type=int, default=32)
    parser.add_argument('--deep_cross_num_head', type=int, default=1)
    parser.add_argument('--deep_cross_dropout', type=int, default=0.18)
    parser.add_argument('--deep_cross_residual_coef', type=float, default=0.2)

    parser.add_argument('--out_scores', type=int, default=1)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading data from {args.datadir}")
    try:
        x1 = np.loadtxt(args.datadir+"x_Tc_unirep_dataset.csv", delimiter=",", dtype="float")
        x2 = np.loadtxt(args.datadir+"x_Tc_protT5_dataset.csv", delimiter=",", dtype="float")
        x3 = np.loadtxt(args.datadir+"x_Tc_esm2_dataset.csv", delimiter=",", dtype="float")
        y = np.loadtxt(args.datadir+"y_Tc_esm2_dataset.csv", delimiter=",", dtype="float")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"[INFO] Applying Global MinMax Normalization...")
    y_min = np.min(y)
    y_max = np.max(y)
    denominator = y_max - y_min

    if denominator != 0:
        y_norm = (y - y_min) / denominator
    else:
        y_norm = y

    print(f"Global Y Range: [{y_min}, {y_max}] -> Normalized: [{np.min(y_norm)}, {np.max(y_norm)}]")

    print(f"[INFO] Splitting data (Seed={args.seed})...")
    idx_all = np.arange(len(y_norm))

    # Train (70%) vs Temp (30%)
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=(1.0 - args.train_ratio), random_state=args.seed, shuffle=True
    )
    # Temp -> Val (15%) vs Test (15%)
    val_rel_ratio = args.val_ratio / (args.val_ratio + args.test_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=(1.0 - val_rel_ratio), random_state=args.seed + 1, shuffle=True
    )

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    def prep_dataset(idxs, x1, x2, x3, y_n):
        sub_x1, sub_x2, sub_x3 = x1[idxs], x2[idxs], x3[idxs]
        sub_y = y_n[idxs]
        return MyDataset_3input(
            x1=torch.tensor(sub_x1, dtype=torch.float64),
            x2=torch.tensor(sub_x2, dtype=torch.float64),
            x3=torch.tensor(sub_x3, dtype=torch.float64),
            y=torch.tensor(sub_y, dtype=torch.float64)
        )

    ds_train = prep_dataset(idx_train, x1, x2, x3, y_norm)
    ds_val = prep_dataset(idx_val, x1, x2, x3, y_norm)
    ds_test = prep_dataset(idx_test, x1, x2, x3, y_norm)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    model = ProtSATT(
        dropout=args.dropout,
        first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
        self_deep=args.self_deep,
        deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
        deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
        out_scores=args.out_scores
    ).double().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[INFO] Using Loss: {args.loss_type}")
    criterion = None
    criterion_mse = None
    criterion_pcc = None

    if args.loss_type == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    elif args.loss_type == 'Huber':
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        # MSE + PCC
        criterion_mse = nn.MSELoss()
        criterion_pcc = PearsonLoss()

    print("Start Training...")
    best_val_loss = float('inf')
    best_epoch = -1
    patience_cnt = 0
    best_model_path = os.path.join(args.out_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        # Train
        if args.loss_type != 'MSE_PCC':
            train_loss = train_epoch(model, dl_train, optimizer, criterion)
        else:
            train_loss = train_epoch_combined(model, dl_train, optimizer, criterion_mse, criterion_pcc)

        # Validate
        if args.loss_type != 'MSE_PCC':
            val_loss, y_true_val, y_pred_val = validate(model, dl_val, criterion)
        else:
            val_loss, y_true_val, y_pred_val = validate_combined(model, dl_val, criterion_mse, criterion_pcc)

        #  Pearson
        if np.std(y_pred_val) < 1e-9 or np.std(y_true_val) < 1e-9:
            val_pearson = 0.0
        else:
            val_pearson, _ = pearsonr(y_true_val.ravel(), y_pred_val.ravel())

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PCC: {val_pearson:.4f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 8. Test Set
    print(f"\nTraining finished. Best Epoch: {best_epoch}")
    print("Loading best model for Test evaluation...")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: Best model file not found!")

    model.to(device)

    #  Test 
    if args.loss_type != 'MSE_PCC':
        _, y_test, p_test = validate(model, dl_test, criterion)
    else:
        _, y_test, p_test = validate_combined(model, dl_test, criterion_mse, criterion_pcc)

    y_test = y_test.ravel()
    p_test = p_test.ravel()

    metrics = calculate_metrics(y_test, p_test, thr=args.thr)

    print("\n" + "="*30)
    print(f"Test Set Results ({args.loss_type} Loss)")
    print("="*30)
    for k, v in metrics.items():
        if k != "CM":
            print(f"{k:<15}: {v:.4f}")

    # Predictions
    df_pred = pd.DataFrame({
        "y_true": y_test,
        "y_pred": p_test
    })
    df_pred.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    # Metrics Report
    with open(os.path.join(args.out_dir, "metrics_report.txt"), "w") as f:
        f.write(f"ProtSATT TR Training Report (Global Norm)\n")
        f.write(f"Loss Function: {args.loss_type}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<15}: {v}\n")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, p_test, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Test Scatter ({args.loss_type}, R2={metrics["R2"]:.3f})')
    plt.savefig(os.path.join(args.out_dir, "test_scatter.png"))
    plt.close()

    print(f"\n[Done] Results saved to {args.out_dir}")

if __name__ == '__main__':
    main()