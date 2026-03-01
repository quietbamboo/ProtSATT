import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

from Model import ProtSATT, multi_layer_attention_no_self, multi_layer_attention_no_cross, multi_layer_attention_2input, multi_layer_attention_1input
from common import MyDataset_3input, split_train_test_fold
import utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=68):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calculate_metrics(y_true, y_pred, y_prob):
    """
    binary performance
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "PRECISION": precision_score(y_true, y_pred, zero_division=0),
        "RECALL": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "SPECIFICITY": specificity,
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob)
    }
    return metrics

def run_inference(model, dataloader, args):
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_prob_list = [] # class 1

    with torch.no_grad():
        for x1, x2, x3, y in dataloader:
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)

            model_output = model(x1, x2, x3, device=device,
                                 first_self_query_dim=args.first_self_query_dim,
                                 deep_self=True,
                                 deep_self_query_dim=args.deep_self_query_dim,
                                 deep_cross_query_dim=args.deep_cross_query_dim)

            # logits = score_class_1 - score_class_0
            logits = model_output[:, 1] - model_output[:, 0]

            probs = torch.sigmoid(logits)

            preds = (logits > 0).long()

            y_true_list.extend(y.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            y_prob_list.extend(probs.cpu().numpy())

    return np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)

def main():
    parser = argparse.ArgumentParser(description='ProtSATT Inference')

    parser.add_argument('--datadir', type=str, default='\datasets\EColi\\')
    parser.add_argument('--model_dir', type=str, required=False,
                        help='Directory containing the .pth files',
                        default=r'\EColi\2labels\lr0.004_firstSelf0_deepSelf0.5_cross0\save_models')

    parser.add_argument('--model_name', type=str, default='ProtSATT')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=0)
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
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)

    # 1. load data
    print(f"Loading data from {args.datadir}...")
    x_dir_path3 = os.path.join(args.datadir, "x_esm2_dataset.csv")
    x_dir_path2 = os.path.join(args.datadir, "x_protT5_dataset.csv")
    x_dir_path1 = os.path.join(args.datadir, "x_unirep_dataset.csv")
    y_dir_path = os.path.join(args.datadir, "y_dataset.csv")

    try:
        x_dataset1 = np.loadtxt(x_dir_path1, delimiter=",", dtype="float")
        x_dataset2 = np.loadtxt(x_dir_path2, delimiter=",", dtype="float")
        x_dataset3 = np.loadtxt(x_dir_path3, delimiter=",", dtype="float")
        y_dataset = np.loadtxt(y_dir_path, delimiter=",", dtype="float")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # initial results
    all_fold_metrics = []

    # 2. 10 folds
    Cross_Fold = 10
    print(f"Starting inference for {Cross_Fold} folds...")

    for i in range(1, Cross_Fold + 1):
        print(f"\nProcessing Fold {i}...")

        # 2.1 initial model
        if args.model_name == 'ProtSATT':
            model = ProtSATT(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep,
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
            ).to(device)
        else:
            raise ValueError(f"Model {args.model_name} not implemented in inference script yet.")

        # 2.2 load weights
        weights_path = os.path.join(args.model_dir, f'best_acc{i}.pth')
        if not os.path.exists(weights_path):
            print(f"Warning: Checkpoint not found at {weights_path}, skipping Fold {i}")
            continue

        model.load_state_dict(torch.load(weights_path, map_location=device))

        # 2.3 get testset
        _, _, _, _, x_test1, x_test2, x_test3, y_test = split_train_test_fold(
            x_dataset1, x_dataset2, x_dataset3, y_dataset, Cross_Fold, i
        )

        dataset_test = MyDataset_3input(x1=x_test1, x2=x_test2, x3=x_test3, y=y_test)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        # 2.4 inference
        y_true, y_pred, y_prob = run_inference(model, dataloader_test, args)

        # 2.5 calculate performance
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics['Fold'] = i
        all_fold_metrics.append(metrics)

        print(f"Fold {i}: ACC={metrics['ACC']:.4f}, AUC={metrics['ROC_AUC']:.4f}")

    if not all_fold_metrics:
        print("No results generated.")
        return

    # 3. save metrics
    df_results = pd.DataFrame(all_fold_metrics)

    cols = ['Fold'] + [c for c in df_results.columns if c != 'Fold']
    df_results = df_results[cols]


    metrics_only = df_results.drop(columns=['Fold'])
    mean_metrics = metrics_only.mean()
    std_metrics = metrics_only.std()

    print("\n========== Final Results (Mean ± Std) ==========")
    for col in mean_metrics.index:
        print(f"{col}: {mean_metrics[col]:.4f} ± {std_metrics[col]:.4f}")

    df_save = df_results.copy()
    df_save.loc['Mean'] = mean_metrics
    df_save.loc['Std'] = std_metrics
    df_save.at['Mean', 'Fold'] = 'Average'
    df_save.at['Std', 'Fold'] = 'Std Dev'

    csv_path = os.path.join(args.model_dir, 'inference_metrics_summary.csv')
    df_save.to_csv(csv_path, index=False)
    print(f"\nDetailed CSV saved to: {csv_path}")

    txt_path = os.path.join(args.model_dir, 'inference_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("ProtSATT EColi 10-Fold Inference Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model Path: {args.model_dir}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Metric Summary (Mean ± Std):\n")
        f.write("-" * 30 + "\n")
        for col in mean_metrics.index:
            line = f"{col:<15}: {mean_metrics[col]:.4f} ± {std_metrics[col]:.4f}\n"
            f.write(line)

    print(f"Summary Report saved to: {txt_path}")

if __name__ == '__main__':
    main()