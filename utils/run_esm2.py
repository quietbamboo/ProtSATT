import numpy as np
import pandas as pd
import torch
import pickle, math
import os, sys
from tqdm import tqdm
from esm.pretrained import esm2_t33_650M_UR50D
import json
import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
current_dir = os.path.dirname(sys.argv[0])

def construct_seq_dict(fasta_path):
    seq_dict = {}
    f = open(fasta_path)
    seq_name = 'init'
    for line in f:
        if line.startswith('>'):
            seq_name = line.replace('>','').replace('\n','')
        else:
            seq_dict[seq_name]=line.replace('\n','')
    f.close()
    return seq_dict

def cal_esm2_to_json(fasta_path, save_esm2_path):
    model, alphabet = esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = model.to(device)

    res_dict = {}
    res_dict_json = {}
    seq_dict = construct_seq_dict(fasta_path)
    seq_dict_len = len(seq_dict)
    for index, seq_name in enumerate(seq_dict):
        start_dt = datetime.datetime.now()
        print('======>', index+1, '/', seq_dict_len)
        batch_labels, batch_strs, batch_tokens = batch_converter(((seq_name, seq_dict[seq_name]),) )
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        token_representations = results["representations"][33]
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        esm2_vector = [token_representations[i, 1: tokens_len - 1].mean(0).cpu().detach().numpy() for i, tokens_len in
                       enumerate(batch_lens)]
        res_dict[seq_name]=esm2_vector[0]
        res_dict_json[seq_name]=esm2_vector[0].tolist()
        end_dt = datetime.datetime.now()
        time_cost = (end_dt - start_dt).seconds
        print("time cost:", time_cost, "s", " ------> remain time:", int((seq_dict_len-index-1)*time_cost/60), "min")

    with open(save_esm2_path, "w") as fp:
        json.dump(res_dict_json, fp)
    return res_dict

if __name__ == '__main__':
    fasta_path = rf'/datasets/test_dataset.fasta'
    save_esm2_path = rf'/datasets/test_dataset_esm2.json'
    cal_esm2_to_json(fasta_path, save_esm2_path)
