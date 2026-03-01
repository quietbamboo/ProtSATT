import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    average_precision_score, r2_score, mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from Model import ProtSATT, multi_layer_attention_no_self, multi_layer_attention_no_cross, multi_layer_attention_2input, multi_layer_attention_1input
from common import MyDataset_3input
import utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=68):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def choose_model(model_name, args):
    if model_name == 'ProtSATT':
        model = ProtSATT(
            dropout=args.dropout,
            first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
            self_deep=args.self_deep,
            deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
            deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
            out_scores=args.out_scores
        ).double().to(device)
    else:
        raise ValueError(f"Model {model_name} not supported in this inference script yet.")
    return model

def calculate_metrics(y_true, y_pred_score, threshold=0.5):
    r2 = r2_score(y_true, y_pred_score)
    mse = mean_squared_error(y_true, y_pred_score)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_score)
    pearson, _ = pearsonr(y_true, y_pred_score)

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        # Regression
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,

        # Classification
        "ACC": accuracy_score(y_true_bin, y_pred_bin),
        "PRECISION": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "RECALL": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "F1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "MCC": matthews_corrcoef(y_true_bin, y_pred_bin),
        "SPECIFICITY": specificity,
        "ROC_AUC": roc_auc_score(y_true_bin, y_pred_score),
        "PR_AUC": average_precision_score(y_true_bin, y_pred_score),

        # Counts
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    return metrics

def run_inference(model, dataloader, args):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x1, x2, x3, y in dataloader:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            model_output = model(x1, x2, x3, device=device,
                                 first_self_query_dim=args.first_self_query_dim,
                                 deep_self=False, 
                                 deep_self_query_dim=args.deep_self_query_dim,
                                 deep_cross_query_dim=args.deep_cross_query_dim)

            preds = model_output.cpu().detach().numpy()

            y_true_list.extend(y.cpu().numpy())
            y_pred_list.extend(preds)

    return np.array(y_true_list), np.array(y_pred_list)

def main():
    parser = argparse.ArgumentParser(description='ProtSATT eSOL Inference')

    parser.add_argument('--datadir', type=str, default=r'\datasets\eSOL')

    parser.add_argument('--model_path', type=str, default=r"C:\models\eSOL\best_epoch_377_1.pth",
                        help='Full path to .pth file (e.g., /results/eSOL/.../best_epoch_377.pth)')
    parser.add_argument('--output_dir', type=str, default=r'\eSOL\predict\inference_results_eSOL_timeCost1')


    parser.add_argument('--model_name', type=str, default='ProtSATT')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--out_scores', type=int, default=1) 

    # Attention Params
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
    parser.add_argument('--deep_self_residual_coef', type=float, default=0)

    parser.add_argument('--deep_cross_query_dim', type=int, default=8)
    parser.add_argument('--deep_cross_return_dim', type=int, default=32)
    parser.add_argument('--deep_cross_num_head', type=int, default=1)
    parser.add_argument('--deep_cross_dropout', type=int, default=0.15)
    parser.add_argument('--deep_cross_residual_coef', type=float, default=1) 

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=68)
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for binary metrics")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading Data from {args.datadir}...")
    # Test Set
    x_test1 = np.loadtxt(os.path.join(args.datadir, "x_eSol_test_esm2_dataset.csv"), delimiter=",", dtype="float")
    x_test2 = np.loadtxt(os.path.join(args.datadir, "x_eSol_test_protT5_dataset.csv"), delimiter=",", dtype="float")
    x_test3 = np.loadtxt(os.path.join(args.datadir, "x_eSol_test_unirep_dataset.csv"), delimiter=",", dtype="float")
    y_test = np.loadtxt(os.path.join(args.datadir, "y_eSol_test_dataset.csv"), delimiter=",", dtype="float")

    # x_test1 = np.loadtxt(os.path.join(args.datadir, "x_S.cerevisiae_test_esm2_dataset.csv"), delimiter=",", dtype="float")
    # x_test2 = np.loadtxt(os.path.join(args.datadir, "x_S.cerevisiae_test_protT5_dataset.csv"), delimiter=",", dtype="float")
    # x_test3 = np.loadtxt(os.path.join(args.datadir, "x_S.cerevisiae_test_unirep_dataset.csv"), delimiter=",", dtype="float")
    # y_test = np.loadtxt(os.path.join(args.datadir, "y_S.cerevisiae_test_esm2_dataset.csv"), delimiter=",", dtype="float")

    dataset_test = MyDataset_3input(x1=x_test1, x2=x_test2, x3=x_test3, y=y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"[INFO] Initializing Model {args.model_name}...")
    model = choose_model(args.model_name, args)

    print(f"[INFO] Loading Weights from {args.model_path}...")
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Error: Model file not found at {args.model_path}")
        return

    print("[INFO] Running Inference...")
    y_true, y_pred = run_inference(model, dataloader_test, args)

    if y_pred.ndim > 1: y_pred = y_pred.ravel()
    if y_true.ndim > 1: y_true = y_true.ravel()

    print("[INFO] Calculating Metrics...")
    metrics = calculate_metrics(y_true, y_pred, threshold=args.threshold)

    print("\n" + "="*30)
    print("eSOL Inference Results")
    print("="*30)
    print("--- Regression ---")
    print(f"R2        : {metrics['R2']:.4f}")
    print(f"RMSE      : {metrics['RMSE']:.4f}")
    print(f"MAE       : {metrics['MAE']:.4f}")
    print(f"Pearson   : {metrics['Pearson']:.4f}")
    print("\n--- Binary (Thr={}) ---".format(args.threshold))
    print(f"ACC       : {metrics['ACC']:.4f}")
    print(f"F1        : {metrics['F1']:.4f}")
    print(f"MCC       : {metrics['MCC']:.4f}")
    print(f"ROC_AUC   : {metrics['ROC_AUC']:.4f}")
    print(f"PR_AUC    : {metrics['PR_AUC']:.4f}")
    print(f"SPEC      : {metrics['SPECIFICITY']:.4f}")
    print("="*30)

    df_pred = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_cls": (y_true >= args.threshold).astype(int),
        "y_pred_cls": (y_pred >= args.threshold).astype(int)
    })
    pred_csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    df_pred.to_csv(pred_csv_path, index=False)

    report_path = os.path.join(args.output_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("ProtSATT eSOL Inference Report\n")
        f.write(f"Model: {args.model_path}\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<15}: {v:.6f}\n")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Solubility')
    plt.ylabel('Predicted Solubility')
    plt.title(f'eSOL Pred vs Actual (R2={metrics["R2"]:.3f})')
    plt.savefig(os.path.join(args.output_dir, "scatter_plot.png"))
    plt.close()

    print(f"\n[Done] Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()