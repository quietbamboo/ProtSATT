import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from Bio.PDB import PDBParser
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_geometric.utils import add_self_loops
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, roc_auc_score, average_precision_score
)
import warnings
warnings.filterwarnings("ignore")

# Import your custom model
from model import FGNNSol  

# ==========================================
# 1. Configuration and Device Setup
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class InferArgs:
    def __init__(self, task_type='binary'):
        """
        task_type: 'binary' (Classes 0, 2) or 'multi' (Classes 0, 1, 2)
        """
        self.task_type = task_type
        self.num_classes = 2 if self.task_type == 'binary' else 3
        
        # Hyperparameters MUST match the training configuration exactly
        self.node_dim = 1280 + 184      
        self.edge_in_channels = 450 
        self.hidden_channels = 256
        self.extra_feat_dim = 16
        self.gcn_num_layers = 4
        self.gat_num_layers = 2
        self.dropout = 0.0 # Disable dropout during inference for deterministic results
        
        # Paths: Please update these to your actual weight and data paths
        self.model_weight_path = f"/root/autodl-fs/dwc/ablation/FGNNSol/results_{self.num_classes}labels_lr5e-05/best_fgnnsol_fold1.pth"
        self.cache_dir = "/root/autodl-fs/dwc/ablation/MTPSol/EColi" 
        self.pdb_dir = "/root/autodl-fs/dwc/EColi/pdb/" 

# ==========================================
# 2. Model Initialization and Loading
# ==========================================
def load_trained_model(args):
    model = FGNNSol(
        node_dim=args.node_dim,
        edge_input_dim=args.edge_in_channels,
        hidden_dim=args.hidden_channels,
        global_dim=args.extra_feat_dim,
        gcn_num_layers=args.gcn_num_layers,
        gat_num_layers=args.gat_num_layers,
        dropout=args.dropout,
        device=device
    )
    # Adjust the final fully connected layer dimension based on task type
    if hasattr(model, 'FC_2'):
        in_dim = model.FC_2.in_features
        model.FC_2 = nn.Linear(in_dim, args.num_classes)
        
    print(f"[*] Loading weights from: {args.model_weight_path}")
    model.load_state_dict(torch.load(args.model_weight_path, map_location=device))
    model.to(device)
    model.eval() # Switch to evaluation mode
    return model

# ==========================================
# 3. Data Processing (Identical to Training)
# ==========================================
def extract_pdb_coords(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb_file)
    coords = []
    
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ': # Standard amino acids only
                    try:
                        n = res['N'].get_coord() if 'N' in res else [0,0,0]
                        ca = res['CA'].get_coord() if 'CA' in res else [0,0,0]
                        c = res['C'].get_coord() if 'C' in res else [0,0,0]
                        o = res['O'].get_coord() if 'O' in res else [0,0,0]
                        
                        sc_atoms = [a.get_coord() for a in res.get_atoms() if a.get_name() not in ['N', 'CA', 'C', 'O', 'H']]
                        r = np.mean(sc_atoms, axis=0) if len(sc_atoms) > 0 else ca
                        
                        coords.append([n, ca, c, o, r])
                    except:
                        coords.append([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]])
    return torch.tensor(coords, dtype=torch.float32)

def build_single_pyg_data(seq_emb, pdb_path, label=0):
    """Constructs a PyG Data object for a single protein sample."""
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
    X = extract_pdb_coords(pdb_path) 
    L = min(X.shape[0], seq_emb.shape[0])
    
    X = X[:L]
    seq_emb = seq_emb[:L]
    
    ca_coords = X[:, 1, :]
    edge_index = radius_graph(ca_coords, r=8.0, loop=False)
    edge_index, _ = add_self_loops(edge_index, num_nodes=L)
    
    data = Data(
        X=X, 
        node_feat=seq_emb.to(torch.float32), 
        edge_index=edge_index,
        extra_feat=torch.zeros((1, 16), dtype=torch.float32), 
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=L   
    )
    return data

# ==========================================
# 4. Core Inference Functions
# ==========================================
def predict_single_sample(model, seq_emb, pdb_path, args):
    """Performs inference on a single sequence."""
    data = build_single_pyg_data(seq_emb, pdb_path)
    data = data.to(device)
    
    with torch.no_grad():
        logits = model(data)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = torch.argmax(logits, dim=1).item()
        
    # Parse prediction results based on task type
    if args.task_type == 'binary':
        # Binary label mapping: 0 -> Low (Original 0), 1 -> High (Original 2)
        label_map_inverse = {0: 'Low (0)', 1: 'High (2)'}
        result = {
            'predicted_class': label_map_inverse[pred_idx],
            'prob_low': probs[0],
            'prob_high': probs[1]
        }
    else:
        # Multi-class label mapping: 0 -> Low, 1 -> Intermediate, 2 -> High
        label_map_inverse = {0: 'Low (0)', 1: 'Intermediate (1)', 2: 'High (2)'}
        result = {
            'predicted_class': label_map_inverse[pred_idx],
            'prob_low': probs[0],
            'prob_intermediate': probs[1],
            'prob_high': probs[2]
        }
        
    return result

def evaluate_dataset(model, dataloader, num_classes):
    """Evaluates the model on an entire DataLoader, computing all metrics."""
    all_y, all_probs, all_preds = [], [], []
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            logits = model(data)
            
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_y.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_y = np.array(all_y)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    try:
        if num_classes == 2:
            pos_probs = all_probs[:, 1]
            roc_auc = roc_auc_score(all_y, pos_probs)
            pr_auc = average_precision_score(all_y, pos_probs)
        else:
            roc_auc = roc_auc_score(all_y, all_probs, multi_class='ovr')
            pr_auc = average_precision_score(pd.get_dummies(all_y).values, all_probs, average='macro')
    except ValueError:
        roc_auc, pr_auc = 0.0, 0.0

    metrics = {
        'acc': accuracy_score(all_y, all_preds),
        'precision': precision_score(all_y, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_y, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_y, all_preds, average='macro'),
        'mcc': matthews_corrcoef(all_y, all_preds),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    return metrics

# ==========================================
# 5. Main Execution Example
# ==========================================
if __name__ == "__main__":
    # ==========================
    # Step 1: Define Task Type
    # ==========================
    TASK_TYPE = 'binary'  # Change to 'multi' to test the 3-class model
    args = InferArgs(task_type=TASK_TYPE)
    
    # Load the trained model
    model = load_trained_model(args)
    
    print("\n" + "="*40)
    print(f"[*] Initializing {TASK_TYPE.upper()} Classification Inference...")
    print("="*40)

    # Load feature data (identical to training phase)
    raw_seq_feats = torch.load(os.path.join(args.cache_dir, "sequence_feature.pt"))
    raw_labels = torch.load(os.path.join(args.cache_dir, "label_feature.pt"))
    
    # ==========================
    # Demo A: Single Sample Inference
    # ==========================
    print("\n--- Demo A: Single Sample Prediction ---")
    # Using the first sample as an example
    sample_idx = 0 
    sample_seq_emb = raw_seq_feats[sample_idx]
    sample_raw_label = int(raw_labels[sample_idx])
    sample_pdb_path = os.path.join(args.pdb_dir, f"EColi{sample_idx}_{sample_raw_label}.pdb")
    
    try:
        res = predict_single_sample(model, sample_seq_emb, sample_pdb_path, args)
        print(f"Processing PDB file: {sample_pdb_path}")
        print(f"Ground truth original label: {sample_raw_label}")
        print(f"Model prediction: {res['predicted_class']}")
        print(f"Class probability distribution: {res}")
    except Exception as e:
        print(f"Single sample inference failed: {e}")


    # ==========================
    # Demo B: Batch Dataset Evaluation
    # ==========================
    print("\n--- Demo B: Building Subset and Batch Evaluation ---")
    # Reusing the dataset filtering logic
    filtered_seq_feats, filtered_raw_labels, original_indices = [], [], []
    if args.task_type == 'binary':
        for i, lbl in enumerate(raw_labels):
            if lbl in [0, 2]:
                filtered_seq_feats.append(raw_seq_feats[i])
                filtered_raw_labels.append(lbl)
                original_indices.append(i)
    else:
        filtered_seq_feats = raw_seq_feats
        filtered_raw_labels = raw_labels
        original_indices = list(range(len(raw_labels)))

    unique_labels = sorted(list(set(filtered_raw_labels)))
    label_map = {val: idx for idx, val in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in filtered_raw_labels])
    
    # For demonstration purposes, we take the first 100 samples as a "Test Set"
    test_ds = []
    print("Building PyG Data objects for the test subset...")
    for i in range(min(100, len(filtered_raw_labels))):
        pdb_path = os.path.join(args.pdb_dir, f"EColi{original_indices[i]}_{int(filtered_raw_labels[i])}.pdb")
        if os.path.exists(pdb_path):
            data = build_single_pyg_data(filtered_seq_feats[i], pdb_path, label=mapped_labels[i])
            test_ds.append(data)
            
    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        test_metrics = evaluate_dataset(model, test_loader, args.num_classes)
        print("\n[*] Batch Evaluation Results (Metrics):")
        for k, v in test_metrics.items():
            print(f"    {k.upper()}: {v:.4f}")
    else:
        print("[!] Failed to build test dataset. No valid PDB files found.")