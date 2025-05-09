import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTHONHASHSEED'] = '68'

import argparse
import numpy as np
import pandas as pd
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader, Subset
import utils
from Model import *
import torch
from common import MyDataset_3input, split_train_test_fold
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn import preprocessing
# matplotlib.use('TkAgg')
# from keras.utils import np_utils
# from sklearn.metrics import classification_report

_print_freq = 50
def seed_torch(seed=68):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 配置CuDNN确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_torch()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    elif (model_name == 'multi_layer_attention_no_self'):
        model = multi_layer_attention_no_self(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).double().to(device)
    elif (model_name == 'multi_layer_attention_no_cross'):
        model = multi_layer_attention_no_cross(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).double().to(device)
    elif (model_name == 'multi_layer_attention_2input'):
        model = multi_layer_attention_2input(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).double().to(device)
    elif (model_name == 'multi_layer_attention_1input'):
        model = multi_layer_attention_1input(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).double().to(device)
    else:
        model = ProtSATT(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores
        ).double().to(device)
    return model

def train(model, dataloader, optim, loss_fn, scheduler, args, e):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    y_pred = list()
    y_actual = list()
    header = 'Training Epoch: [{}]'.format(e)
    for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

        # y = np_utils.to_categorical(y, 2)
        # x torch.Size([300, 1280])
        model_output = model(x1, x2, x3, device=device,
                            first_self_query_dim=args.first_self_query_dim, 
                            deep_self=False, 
                            deep_self_query_dim=args.deep_self_query_dim, 
                            deep_cross_query_dim=args.deep_cross_query_dim)
        loss = loss_fn(model_output, y)

        y_pred.extend(model_output.cpu().detach().numpy())
        y_actual.extend(y.float().detach().cpu().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler is not None and args.scheduler == 'LambdaLR' or args.scheduler == 'Cos':
            scheduler.step()
        metric_logger.update(loss=loss)

    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, y_pred, y_actual

def evaluate_metrics(model, dataloader, loss_fn, args, e):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    predictions_test = list()
    y_actual = list()
    loss_mse = list()
    y_pred = list()
    header = 'Evaluation Epoch: [{}]'.format(e)
    with torch.no_grad():
        for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            # y = np_utils.to_categorical(y, 2)
            model_output = model(x1, x2, x3, device=device,
                            first_self_query_dim=args.first_self_query_dim, 
                            deep_self=False, 
                            deep_self_query_dim=args.deep_self_query_dim, 
                            deep_cross_query_dim=args.deep_cross_query_dim)
            loss = loss_fn(model_output, y)

            y_pred.extend(model_output.cpu().detach().numpy())
            y_actual.extend(y.float().detach().cpu().numpy())
            metric_logger.update(loss=loss)
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, y_pred, y_actual

def compute_indicate(y_pred, y_actual):
    pred_correct = sum([1 for x, y in zip(y_pred, y_actual) if (x>=0.5 and y>=0.5) or (x<0.5 and y<0.5)])
    pred_correct_1 = sum([1 for x, y in zip(y_pred, y_actual) if (x>=0.5 and y>=0.5)])
    pred_correct_0 = sum([1 for x, y in zip(y_pred, y_actual) if (x<0.5 and y<0.5)])
    pred_1 = sum([1 for x, y in zip(y_pred, y_actual) if (x>=0.5)])
    pred_0 = sum([1 for x, y in zip(y_pred, y_actual) if (x<0.5)])
    actual_num = len(y_actual)
    actual_num_1 = sum([1 for x, y in zip(y_pred, y_actual) if y>=0.5])
    actual_num_0 = sum([1 for x, y in zip(y_pred, y_actual) if y<0.5])

    # correct_2 = sum([1 for x, y in zip(y_pred, y_actual) if x==y and x==2])
    Accuracy = 0 if actual_num==0 else pred_correct/actual_num
    Recall = 0 if actual_num_1==0 else pred_correct_1/actual_num_1
    Precision = 0 if pred_1==0 else pred_correct_1/pred_1
    return Accuracy, Recall, Precision

def main(lr=0.0006, first_self_residual_coef=0, deep_self_residual_coef=0, deep_cross_residual_coef=1):
    parser = argparse.ArgumentParser(description='SOLUABLE_REGRESSION')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--datadir', type=str, default='/datasets/eSOL/')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--optim', type=str, default='AdamW', choices=('SGD', 'Adam', 'AdamW'))
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--weight_decay', type=float, default=0.003)
    parser.add_argument('--warmup', type=int, default=150)
    parser.add_argument('--scheduler', type=str, default='LambdaLR', choices=('None', 'StepLR', 'LambdaLR', 'Polynomial', 'CosineAnnealingLR', 'cosine'))
    parser.add_argument('--letter_emb_size', type=int, default=16)

    parser.add_argument('--dropout', type=float, default=0.2)
    
    parser.add_argument('--first_self_query_dim', type=int, default=32)
    parser.add_argument('--first_self_return_dim', type=int, default=512)
    parser.add_argument('--first_self_num_head', type=int, default=1)
    parser.add_argument('--first_self_dropout', type=int, default=0.15)
    parser.add_argument('--first_self_residual_coef', type=float, default=first_self_residual_coef)

    parser.add_argument('--self_deep', type=int, default=1)
    parser.add_argument('--deep_self_query_dim', type=int, default=16)
    parser.add_argument('--deep_self_return_dim', type=int, default=128)
    parser.add_argument('--deep_self_num_head', type=int, default=1)
    parser.add_argument('--deep_self_dropout', type=float, default=0.15)
    parser.add_argument('--deep_self_residual_coef', type=float, default=deep_self_residual_coef)

    parser.add_argument('--deep_cross_query_dim', type=int, default=8)
    parser.add_argument('--deep_cross_return_dim', type=int, default=32)
    parser.add_argument('--deep_cross_num_head', type=int, default=1)
    parser.add_argument('--deep_cross_dropout', type=int, default=0.15)
    parser.add_argument('--deep_cross_residual_coef', type=float, default=deep_cross_residual_coef)

    parser.add_argument('--out_scores', type=int, default=1)

    parser.add_argument('--K_fold_train', type=int, default=1) # 1 is fold on
    parser.add_argument('--Cross_Fold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    seed = args.seed

    x_dir_path1 = args.datadir+"x_eSol_train_esm2_dataset.csv" # esm2
    x_dir_path2 = args.datadir+"x_eSol_train_protT5_dataset.csv" # protT5
    x_dir_path3 = args.datadir+"x_eSol_train_unirep_dataset.csv" # unirep
    y_dir_path = args.datadir+"y_eSol_train_dataset.csv"
    x_train1 = np.loadtxt(x_dir_path1, delimiter=",", dtype="float")
    x_train2 = np.loadtxt(x_dir_path2, delimiter=",", dtype="float")
    x_train3 = np.loadtxt(x_dir_path3, delimiter=",", dtype="float")
    y_train = np.loadtxt(y_dir_path, delimiter=",", dtype="float")

    model_name = 'ProtSATT'
    # model_name = 'multi_layer_attention_no_self'
    # model_name = 'multi_layer_attention_no_cross'
    # model_name = 'multi_layer_attention_2input'
    # model_name = 'multi_layer_attention_1input'
    if args.K_fold_train == 1:
        Cross_Fold=args.Cross_Fold
        res_dir = rf'/results/eSOL_fold'
    else:
        Cross_Fold=1
        res_dir = rf'/results/eSOL/{model_name}_one_self/activation_residue/swish/lr{args.lr}_epoch{args.epoch}_warmup{args.warmup}_letterEmbSize{args.letter_emb_size}_scheduler{args.scheduler}_weightDecay{args.weight_decay}_dropout{args.dropout}_firstSelf{args.first_self_residual_coef}_deepSelf{args.deep_self_residual_coef}_cross{args.deep_cross_residual_coef}_A2000_'
        # res_dir = rf'/mnt/dwc/MPEPE/soluable/paper/results/eSOL/{model_name}_one_self/residue_ablation/firstSelf{args.first_self_residual_coef}_deepSelf{args.deep_self_residual_coef}_cross{args.deep_cross_residual_coef}_'

    os.makedirs(res_dir, mode=0o777, exist_ok=True)
    res_model_dir = rf'{res_dir}/save_models'
    os.makedirs(res_model_dir, mode=0o777, exist_ok=True)
    
    all_fold_loss_list=[]
    all_fold_r2_list=[]

    dataset = MyDataset_3input(x1=x_train1, x2=x_train2, x3=x_train3, y=y_train)

    kf = KFold(n_splits=args.Cross_Fold, random_state=666, shuffle=True)
    for i, (train_ids, val_ids) in enumerate(kf.split(dataset)):
        def lambda_lr(s):
            warm_up = args.warmup
            s += 1
            return (args.letter_emb_size ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

        def polynomial(current_step):
            num_warmup_steps = args.warmup
            num_training_steps = args.epoch
            lr_init = args.lr
            lr_end = lr_init / 10
            power = 5
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step > num_training_steps:
                return lr_end / lr_init  # as LambdaLR multiplies by lr_init
            else:
                lr_range = lr_init - lr_end
                decay_steps = num_training_steps - num_warmup_steps
                pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
                decay = lr_range * pct_remaining ** power + lr_end
                return decay / lr_init  # as LambdaLR multiplies by lr_init

        def cosine(current_epoch):
            max_epoch = args.epoch
            lr_min=0
            lr_max=args.lr
            warmup_epoch = args.warmup
            if current_epoch < warmup_epoch:
                return (lr_max * current_epoch / warmup_epoch)
            else:
                return (lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2)

    # for i in range(1, Cross_Fold+1):
        # i=1
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        dataloader_train = DataLoader(train_subsampler, batch_size=args.batch_size, shuffle=True)
        dataloader_val = DataLoader(val_subsampler, batch_size=args.batch_size)
    
        #Model
        model = choose_model(model_name, args)
        if args.optim == 'Adam' and args.scheduler != 'LambdaLR':
            optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        elif args.optim == 'Adam' and args.scheduler == 'LambdaLR':
            optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        elif args.optim == 'AdamW' and args.scheduler == 'LambdaLR':
            optim = AdamW(model.parameters(), lr=args.lr, betas=(0.88, 0.98), weight_decay=args.weight_decay)
        else:
            optim = SGD(model.parameters(), lr=args.lr, momentum=0.9)

        if args.scheduler == 'StepLR':
            scheduler = StepLR(optim, step_size=30, gamma=0.5)
        elif args.scheduler == 'LambdaLR':
            scheduler = LambdaLR(optim, lambda_lr)
        elif args.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optim, T_max=5000)
        elif args.scheduler == 'cosine':
            scheduler = LambdaLR(optim, cosine)
        else:
            scheduler = None
        # loss_fn = torch.nn.L1Loss()
        loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.NLLLoss()

        best_train_mse = 999.0
        best_val_mse = 999.0
        best_r2 = .0
        best_r2_epoch = 0
        best_train_epoch = 0
        best_val_epoch = 0
        val_r2_list = []

        result = []
        epoch_list = []
        train_loss_list = []
        val_loss_list = []

        best_r2_loss = 999
        best_val_y_pred_in_loss = []
        best_val_y_actual_in_loss = []
        best_val_y_pred_in_r2 = []
        best_val_y_actual_in_r2 = []

        best_val_correlation_in_loss = 0
        for e in range(args.epoch):
            train_mse_loss, train_y_pred, train_y_actual = train(model, dataloader_train, optim, loss_fn, scheduler, args, e)
            epoch_list.append(e)
            train_loss_list.append(train_mse_loss)
            print("Train loss MSE: %s", train_mse_loss)
            if scheduler is not None and args.scheduler == 'StepLR':
                scheduler.step()
            # train mse
            if best_train_mse > train_mse_loss:
                best_train_mse = train_mse_loss
                best_train_epoch = e

            # val mse
            val_loss, val_y_pred, val_y_actual = evaluate_metrics(model, dataloader_val, loss_fn, args, e)
            val_loss_list.append(val_loss)
            print("Val loss MSE: %s", val_loss)
            Accuracy, Recall, Precision = compute_indicate(val_y_pred, val_y_actual)
            if best_val_mse > val_loss:
                best_val_mse = val_loss
                best_val_epoch = e
                best_val_y_pred_in_loss = val_y_pred
                best_val_y_actual_in_loss = val_y_actual
                best_val_correlation_in_loss = r2_score(val_y_actual, val_y_pred)
                # torch.save(model.state_dict(), rf'{res_model_dir}/best_loss{i}.pth')
            
            # val r2 
            correlation = r2_score(val_y_actual, val_y_pred)
            if best_r2 < correlation:
                best_r2 = correlation
                best_r2_epoch = e
                best_r2_loss = val_loss
                best_val_y_pred_in_r2 = val_y_pred
                best_val_y_actual_in_r2 = val_y_actual   
                # torch.save(model.state_dict(), rf'{res_model_dir}/best_r2{i}.pth')
            val_r2_list.append(correlation)
            print("best_r2: %s", best_r2, " best_r2_epoch: %s", best_r2_epoch)


        # plot loss
        plt.plot(epoch_list, train_loss_list, label='train_loss')
        plt.plot(epoch_list, val_loss_list, label='val_loss')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("metric figure")
        plt.savefig(rf'{res_dir}/loss_fold{i}.png')
        plt.close()
        print(rf"========{i}=========")
        print(f"best_train_mse: {best_train_mse}; epoch: {best_train_epoch}")
        print(f'r2_in_best_loss: {best_val_correlation_in_loss}; epoch: {best_val_epoch}; val_loss: {best_val_mse}')
        print(f'best_r2: {best_r2}; epoch: {best_r2_epoch}; val_loss: {best_r2_loss}')

        # plot actual_pred scatter
        plt.scatter(best_val_y_actual_in_loss, best_val_y_pred_in_loss)
        # plt.plot([min(best_val_y_actual_in_loss), max(best_val_y_actual_in_loss)], [min(best_val_y_actual_in_loss), max(best_val_y_actual_in_loss)], 'r--')
        plt.plot([0, 1], [0, 1], 'r--')
        # plt.text(200, 200, 'R2='+ ('%.3f'%r2score), fontdict={'family': 'serif', 'size': 16, 'color': 'black'}, ha='center', va='center')
        plt.text(-5, 2, 'correlation='+ ('%.3f'%best_val_correlation_in_loss))
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Diagonal Plot - Actual vs. Predicted')
        plt.savefig(rf'{res_dir}/r2_in_best_loss{i}.png')
        plt.close()
        
        plt.scatter(best_val_y_actual_in_r2, best_val_y_pred_in_r2)
        # plt.plot([min(best_val_y_actual_in_r2), max(best_val_y_actual_in_r2)], [min(best_val_y_actual_in_r2), max(best_val_y_actual_in_r2)], 'r--')
        plt.plot([0, 1], [0, 1], 'r--')
        # plt.text(200, 200, 'R2='+ ('%.3f'%r2score), fontdict={'family': 'serif', 'size': 16, 'color': 'black'}, ha='center', va='center')
        plt.text(-5, 2, 'correlation='+ ('%.3f'%best_r2))
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Diagonal Plot - Actual vs. Predicted')
        plt.savefig(rf'{res_dir}/best_r2{i}.png')
        plt.close()
        
        output_list = []
        output_list.append(f'==========fold{i}============\n')
        output_list.append(f"best_train_mse: {best_train_mse}; epoch: {best_train_epoch}\n")
        output_list.append(f'==============val===============\n')
        output_list.append(f'r2_in_best_loss: {best_val_correlation_in_loss}; epoch: {best_val_epoch}; val_loss: {best_val_mse}\n')
        output_list.append(f'==============val===============\n')
        output_list.append(f'best_r2: {best_r2}; epoch: {best_r2_epoch}; val_loss: {best_r2_loss}\n\n')
        with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
            f.writelines(output_list)


        all_fold_loss_list.append(best_val_mse)
        all_fold_r2_list.append(best_r2)
    if args.K_fold_train == 1:
        print(all_fold_r2_list, '===', np.mean(all_fold_r2_list))
        with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
            f.writelines('all_fold_loss_list: '+str(all_fold_loss_list)+'\n')
            f.writelines('mean_loss: '+str(np.mean(all_fold_loss_list))+'\n')
            f.writelines('all_fold_r2_list: '+str(all_fold_r2_list)+'\n')
            f.writelines('mean_r2: '+str(np.mean(all_fold_r2_list))+'\n')

if __name__ == '__main__':
    main(0.0006)