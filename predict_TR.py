import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from scipy.stats import pearsonr, spearmanr

from Model import ProtSATT
from common import MyDataset_3input
import utils

# ==========================================
# Publication Ready
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def run_inference(model, loader, args):
    model.eval()
    y_true = []
    y_pred = []
    features_list = []

    with torch.no_grad():
        for x1, x2, x3, y in loader:
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)

            try:
                output, batch_features = model(
                    x1, x2, x3, device=device,
                    first_self_query_dim=args.first_self_query_dim,
                    deep_self=False, 
                    deep_self_query_dim=args.deep_self_query_dim,
                    deep_cross_query_dim=args.deep_cross_query_dim,
                    return_features=True
                )
                features_list.append(batch_features.cpu().numpy())
            except TypeError:
                output = model(
                    x1, x2, x3, device=device,
                    first_self_query_dim=args.first_self_query_dim,
                    deep_self=False,
                    deep_self_query_dim=args.deep_self_query_dim,
                    deep_cross_query_dim=args.deep_cross_query_dim
                )

            y_pred.extend(output.cpu().numpy())
            y_true.extend(y.numpy())

    if features_list:
        return np.array(y_true), np.array(y_pred), np.concatenate(features_list, axis=0)
    else:
        return np.array(y_true), np.array(y_pred), None

def plot_comprehensive_results(y_true, y_pred, features, threshold, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y_true_bin = (y_true >= threshold).astype(int)

    # 1. Predicted vs Actual
    plt.figure(figsize=(7, 6), dpi=300)
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.4, 's':20, 'color':'#3C5488'}, line_kws={'color':'#E64B35', 'linewidth':2}, label='Fit Line')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal ($y=x$)')
    plt.xlabel('Actual Normalized Value (TR)')
    plt.ylabel('Predicted Normalized Value')
    plt.title(f'Regression Performance\n$R^2$ = {metrics["R2"]:.3f}, Pearson = {metrics["Pearson"]:.3f}')
    plt.legend(loc='upper left')
    sns.despine()
    plt.savefig(os.path.join(output_dir, "01_Scatter_Pred_vs_Actual.pdf"), bbox_inches='tight')
    plt.close()

    # 2. t-SNE 
    if features is not None:
        print("[INFO] Computing t-SNE embeddings...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(y_true)-1), learning_rate=200, init='pca', random_state=42)
        tsne_res = tsne.fit_transform(features)

        # t-SNE
        plt.figure(figsize=(7.5, 6), dpi=300)
        scatter = plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=y_true, cmap='coolwarm', alpha=0.8, s=25, edgecolors='w', linewidths=0.2)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Actual Normalized TR Value')
        plt.title('t-SNE Feature Space (Regression View)')
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        sns.despine()
        plt.savefig(os.path.join(output_dir, "05_tSNE_Regression.pdf"), bbox_inches='tight')
        plt.close()

# ================= main =================

def main():
    parser = argparse.ArgumentParser(description='ProtSATT Inference on TR (Global Norm)')

    parser.add_argument('--datadir', type=str, default='\datasets\Tc-Riboswitches\\')

    parser.add_argument('--model_path', type=str, default=r'models\TR\best_model.pth')

    parser.add_argument('--out_dir', type=str, default=r'results\TR\predict')

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
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

    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[INFO] Loading data from {args.datadir}")
    try:
        x1 = np.loadtxt(args.datadir+"x_Tc_unirep_dataset.csv", delimiter=",", dtype="float")
        x2 = np.loadtxt(args.datadir+"x_Tc_protT5_dataset.csv", delimiter=",", dtype="float")
        x3 = np.loadtxt(args.datadir+"x_Tc_esm2_dataset.csv", delimiter=",", dtype="float")
        y = np.loadtxt(args.datadir+"y_Tc_esm2_dataset.csv", delimiter=",", dtype="float")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"[INFO] Reproducing Global MinMax Normalization...")
    y_min = np.min(y)
    y_max = np.max(y)
    denominator = y_max - y_min
    if denominator != 0:
        y_norm = (y - y_min) / denominator
    else:
        y_norm = y

    print(f"[INFO] Reproducing data split (Seed={args.seed})...")
    idx_all = np.arange(len(y_norm))
    _, idx_temp = train_test_split(
        idx_all, test_size=(1.0 - args.train_ratio), random_state=args.seed, shuffle=True
    )
    val_rel_ratio = args.val_ratio / (args.val_ratio + args.test_ratio)
    _, idx_test = train_test_split(
        idx_temp, test_size=(1.0 - val_rel_ratio), random_state=args.seed + 1, shuffle=True
    )
    print(f"Test Set Size Retrieved: {len(idx_test)}")

    ds_test = MyDataset_3input(
        x1=torch.tensor(x1[idx_test], dtype=torch.float64),
        x2=torch.tensor(x2[idx_test], dtype=torch.float64),
        x3=torch.tensor(x3[idx_test], dtype=torch.float64),
        y=torch.tensor(y_norm[idx_test], dtype=torch.float64)
    )
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Initializing Model & Loading Weights from {args.model_path}")
    model = ProtSATT(
        dropout=args.dropout,
        first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
        self_deep=args.self_deep,
        deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
        deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
        out_scores=args.out_scores
    ).double().to(device)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"[Error] Checkpoint not found at {args.model_path}")
        return

    print("[INFO] Running Inference...")
    y_test, p_test, features = run_inference(model, dl_test, args)
    y_test = y_test.ravel()
    p_test = p_test.ravel()

    print("[INFO] Calculating Metrics...")
    metrics = calculate_metrics(y_test, p_test, thr=args.thr)

    print("\n" + "="*30)
    print("Test Set Inference Results (TR Dataset)")
    print("="*30)
    for k, v in metrics.items():
        if k != "CM":
            print(f"{k:<15}: {v:.4f}")

    print("[INFO] Generating Plots...")
    plot_comprehensive_results(y_test, p_test, features, args.thr, metrics, plot_dir)

    df_pred = pd.DataFrame({
        "y_true_normalized": y_test,
        "y_pred_normalized": p_test,
        "y_true_original": y_test * denominator + y_min,
        "y_pred_original": p_test * denominator + y_min
    })
    df_pred.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(args.out_dir, "inference_metrics_report.txt"), "w") as f:
        f.write("ProtSATT TR Inference Report (Global Norm)\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<15}: {v}\n")

    print(f"\n[Done] All results and plots successfully saved to {args.out_dir}")

if __name__ == '__main__':
    main()