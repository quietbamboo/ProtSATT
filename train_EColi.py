import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)

from Model import ProtSATT

def parse_args():
    parser = argparse.ArgumentParser(description="ProtSATT")
    parser.add_argument('--num_classes', type=int, default=3, choices=[2, 3])
    parser.add_argument('--sim_threshold', type=str, default='Sim_40')
    parser.add_argument('--scheduler', type=str, default='LambdaLR') # ReduceLROnPlateau LambdaLR
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=150) # LambdaLR
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--lr_patience', type=int, default=20) # 5 ReduceLROnPlateau
    parser.add_argument('--lr_factor', type=float, default=0.5) # 0.5 ReduceLROnPlateau
    parser.add_argument('--min_lr', type=float, default=1e-6) # 1e-6 ReduceLROnPlateau
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-2) # 1e-3
    parser.add_argument('--seed', type=int, default=2025) # 42
    parser.add_argument('--data_dir', type=str, default=r'D:\Project\Python\conclusion\datasets\EColi\E_Coli_3labels_seqID')
    parser.add_argument('--save_dir', type=str, default=r'D:\Project\Python\conclusion\EColi\2_3_lables_experiment\checkpoints_2labels_lr1e-2_LambdaLR_monitorACC')

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
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiFeatureDataset(Dataset):
    def __init__(self, x_esm, x_unirep, x_prott5, labels):
        self.x_esm = torch.tensor(x_esm, dtype=torch.float32)
        self.x_unirep = torch.tensor(x_unirep, dtype=torch.float32)
        self.x_prott5 = torch.tensor(x_prott5, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x_esm[idx], self.x_unirep[idx], self.x_prott5[idx], self.labels[idx]

def load_and_align_data(args):
    print(f"📦 loading {args.num_classes} dataset...")

    cluster_csv_path = os.path.join(args.data_dir, 'EColi_homology_clusters_unified.csv')
    cluster_df = pd.read_csv(cluster_csv_path)

    cluster_df = cluster_df.dropna(subset=['Sequence_ID']).copy()
    def safe_extract_label(seq_id):
        try:
            return int(str(seq_id).split('_')[-1])
        except ValueError:
            return -1
    cluster_df['Label'] = cluster_df['Sequence_ID'].apply(safe_extract_label)
    if (cluster_df['Label'] == -1).sum() > 0:
        cluster_df = cluster_df[cluster_df['Label'] != -1].copy()

    if args.num_classes == 2:
        df_filtered = cluster_df[cluster_df['Label'].isin([0, 2])].copy()
        df_filtered['Label'] = df_filtered['Label'].replace({2: 1})
    else:
        df_filtered = cluster_df.copy()

    print(f"-> filtered remaining: {len(df_filtered)}")

    groups = df_filtered[args.sim_threshold].values
    labels = df_filtered['Label'].values
    target_seq_ids = df_filtered['Sequence_ID'].values

    print("⏳ aligning Sequence_ID...")

    def load_and_match_features(csv_filename, target_ids):
        filepath = os.path.join(args.data_dir, csv_filename)
        df_feat = pd.read_csv(filepath, header=None)

        df_feat.rename(columns={0: 'Sequence_ID'}, inplace=True)
        df_feat.set_index('Sequence_ID', inplace=True)

        try:
            aligned_df = df_feat.loc[target_ids]
        except KeyError as e:
            raise KeyError(f"❌ Error： {csv_filename} can't find sequence ID。\nmsg: {e}")

        return aligned_df.values

    X_esm = load_and_match_features('x_EColi_esm2_dataset_with_IDs.csv', target_seq_ids)
    X_unirep = load_and_match_features('x_EColi_unirep_dataset_with_IDs.csv', target_seq_ids)
    X_prott5 = load_and_match_features('x_EColi_protT5_dataset_with_IDs.csv', target_seq_ids)

    print(f"✅ feature ID aligned - ESM2: {X_esm.shape}, UniRep: {X_unirep.shape}, ProtT5: {X_prott5.shape}")
    return X_esm, X_unirep, X_prott5, labels, groups

def evaluate(model, dataloader, criterion, device, num_classes, args):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_esm, x_unirep, x_prott5, y in dataloader:
            x_esm, x_unirep, x_prott5, y = x_esm.to(device), x_unirep.to(device), x_prott5.to(device), y.to(device)

            # logits = model(x_esm, x_unirep, x_prott5)

            logits = model(x_esm, x_unirep, x_prott5, device=device,
                                 first_self_query_dim=args.first_self_query_dim,
                                 deep_self=True,
                                 deep_self_query_dim=args.deep_self_query_dim,
                                 deep_cross_query_dim=args.deep_cross_query_dim)

            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_loss /= len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    metrics = {
        'Loss': total_loss,
        'Accuracy': acc,
        'MCC': mcc
    }

    if num_classes == 2:
        metrics['Precision'] = precision_score(all_labels, all_preds, zero_division=0)
        metrics['Recall'] = recall_score(all_labels, all_preds, zero_division=0)
        metrics['F1'] = f1_score(all_labels, all_preds, zero_division=0)

        pos_probs = [p[1] for p in all_probs]
        metrics['ROC_AUC'] = roc_auc_score(all_labels, pos_probs)
        metrics['PR_AUC'] = average_precision_score(all_labels, pos_probs)

        monitor_auc = metrics['ROC_AUC']
    else:
        metrics['Macro_Precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['Macro_Recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['Macro_F1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        metrics['Macro_ROC_AUC'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        monitor_auc = metrics['Macro_ROC_AUC']

    return monitor_auc, metrics

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    run_name = f"ProtSATT_{args.num_classes}Class_{args.sim_threshold}"
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"=== Starting ProtSATT {args.num_classes}-Class Training on {device} ===")
    X_esm, X_unirep, X_prott5, labels, groups = load_and_align_data(args)

    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    outer_fold_results = []

    for fold, (train_val_idx, test_idx) in enumerate(sgkf_outer.split(X_esm, labels, groups)):
        print(f"\n" + "="*50)
        print(f"🚀 Outer Fold {fold+1}/10")

        X_tv_esm, X_tv_uni, X_tv_pro = X_esm[train_val_idx], X_unirep[train_val_idx], X_prott5[train_val_idx]
        y_tv, groups_tv = labels[train_val_idx], groups[train_val_idx]

        sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + fold)
        inner_train_sub_idx, inner_val_sub_idx = next(sgkf_inner.split(X_tv_esm, y_tv, groups_tv))

        train_idx = train_val_idx[inner_train_sub_idx]
        val_idx = train_val_idx[inner_val_sub_idx]

        train_groups, val_groups, test_groups = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
        assert train_groups.isdisjoint(val_groups) and train_groups.isdisjoint(test_groups) and val_groups.isdisjoint(test_groups), "⚠️ 严重数据泄露！"

        train_dataset = MultiFeatureDataset(X_esm[train_idx], X_unirep[train_idx], X_prott5[train_idx], labels[train_idx])
        val_dataset = MultiFeatureDataset(X_esm[val_idx], X_unirep[val_idx], X_prott5[val_idx], labels[val_idx])
        test_dataset = MultiFeatureDataset(X_esm[test_idx], X_unirep[test_idx], X_prott5[test_idx], labels[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # model = ProtSATT(num_classes=args.num_classes).to(device)
        model = ProtSATT(
            dropout=args.dropout,
            first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
            self_deep=args.self_deep,
            deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
            deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
            out_scores=args.out_scores,
        ).to(device)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.03)

        criterion = nn.CrossEntropyLoss()

        scheduler = None
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
            )
        elif args.scheduler == 'LambdaLR':
            def lambda_lr(s):
                warm_up = args.warmup
                s += 1
                return (32 ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0

        model_save_path = os.path.join(save_path, f"best_model_fold_{fold+1}.pth")

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for x_esm_b, x_unirep_b, x_prott5_b, y_b in train_loader:
                x_esm_b, x_unirep_b, x_prott5_b, y_b = x_esm_b.to(device), x_unirep_b.to(device), x_prott5_b.to(device), y_b.to(device)

                optimizer.zero_grad()
                # logits = model(x_esm_b, x_unirep_b, x_prott5_b)
                logits = model(x_esm_b, x_unirep_b, x_prott5_b, device=device,
                                     first_self_query_dim=args.first_self_query_dim,
                                     deep_self=True,
                                     deep_self_query_dim=args.deep_self_query_dim,
                                     deep_cross_query_dim=args.deep_cross_query_dim)

                loss = criterion(logits, y_b)
                total_loss += loss.item() * y_b.size(0)
                loss.backward()
                optimizer.step()
                if args.scheduler == 'LambdaLR':
                    scheduler.step()
            print(f"   Train Epoch {epoch+1:03d} [] | Train Loss: {total_loss / len(train_loader.dataset)}")
            # 验证评估 (利用字典解包避免混乱)
            val_monitor_auc, val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes, args)

            if args.scheduler == 'ReduceLROnPlateau':
                # scheduler.step(val_monitor_auc)
                scheduler.step(val_metrics['Accuracy'])

            current_lr = optimizer.param_groups[0]['lr']

            if val_metrics['Accuracy'] > best_val_auc:
                best_val_auc = val_metrics['Accuracy']
            # if val_monitor_auc > best_val_auc:
            #     best_val_auc = val_monitor_auc
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)
                patience_counter = 0
                print(f"   Val Epoch {epoch+1:03d} [*] | Val ACC: {val_metrics['Accuracy']:.4f} | Val AUC: {val_monitor_auc:.4f} | Val MCC: {val_metrics['MCC']:.4f} | LR: {current_lr:.2e} | LOSS: {val_metrics['Loss']:.4f}")
            else:
                patience_counter += 1
                print(f"   Val Epoch {epoch+1:03d} [ ] | Val ACC: {val_metrics['Accuracy']:.4f} | Val AUC: {val_monitor_auc:.4f} | Val MCC: {val_metrics['MCC']:.4f} | LR: {current_lr:.2e} | Patience: {patience_counter}/{args.patience} | LOSS: {val_metrics['Loss']:.4f}")

            if patience_counter >= args.patience:
                print(f"   🛑 early patience triggered！{args.patience} epoch didn't improve。")
                break

        print(f"   ✨ Inner-Val 最佳点: Epoch {best_epoch+1} (AUC = {best_val_auc:.4f})")

        model.load_state_dict(torch.load(model_save_path))
        test_monitor_auc, test_metrics = evaluate(model, test_loader, criterion, device, args.num_classes, args)

        if args.num_classes == 2:
            print(f"   🏆 Outer-Test 成绩 -> ROC_AUC: {test_metrics['ROC_AUC']:.4f} | PR_AUC: {test_metrics['PR_AUC']:.4f} | MCC: {test_metrics['MCC']:.4f} | F1: {test_metrics['F1']:.4f} | ACC: {test_metrics['Accuracy']:.4f}")
        else:
            print(f"   🏆 Outer-Test 成绩 -> Macro_AUC: {test_metrics['Macro_ROC_AUC']:.4f} | MCC: {test_metrics['MCC']:.4f} | Macro_F1: {test_metrics['Macro_F1']:.4f} | ACC: {test_metrics['Accuracy']:.4f}")

        test_metrics['Fold'] = fold + 1
        outer_fold_results.append(test_metrics)

    df_res = pd.DataFrame(outer_fold_results)

    cols = ['Fold'] + [c for c in df_res.columns if c != 'Fold']
    df_res = df_res[cols]

    mean_res = df_res.drop(columns=['Fold']).mean()
    std_res = df_res.drop(columns=['Fold']).std()

    report_path = os.path.join(save_path, "Rigorous_10Fold_Summary.txt")
    with open(report_path, 'w') as f:
        f.write(f"========== ProtSATT {args.num_classes}-Class Performance ({args.sim_threshold}) ==========\n\n")

        f.write("--- Fold Details ---\n")
        f.write(df_res.to_string(index=False) + "\n\n")

        f.write("--- Overall Average ---\n")
        for metric in mean_res.index:
            line = f"{metric:<15}: {mean_res[metric]:.4f} ± {std_res[metric]:.4f}\n"
            print(line.strip())
            f.write(line)

    csv_path = os.path.join(save_path, "Rigorous_10Fold_Summary.csv")
    df_res.to_csv(csv_path, index=False)

    print(f"\n✅ saved: {save_path}")

if __name__ == "__main__":
    main()