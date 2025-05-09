import os
seed = 68
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYTHONHASHSEED'] = str(seed)
import argparse
import numpy as np
import pandas as pd
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from scipy import stats
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
import utils
from Model import *
import torch
from common import MyDataset_3input, split_train_test_Tc
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn import preprocessing
# matplotlib.use('TkAgg')
# from keras.utils import np_utils
# from sklearn.metrics import classification_report

os.environ['KMP_DUPLICATE_LIB_OK']='True'

_print_freq = 50

def seed_torch(seed=68):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def MinMaxNomalize(data):
    max_data = max(data)
    min_data = min(data)
    denominator = max_data-min_data
    # 归一化到 [1 1]
    return list(map(lambda x: (x-min_data)/denominator, data))
    # 归一化到 [-1，1]
    # return list(map(lambda x: ((x-min_data)/denominator)-0.5*2, data))

def plot_scatter(actual, pred, pic_name):
    # 绘制actual_pred散点图
    plt.scatter(actual, pred)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    # plt.text(200, 200, 'R2='+ ('%.3f'%r2score), fontdict={'family': 'serif', 'size': 16, 'color': 'black'}, ha='center', va='center')
    plt.text(-5, 2, 'correlation='+ ('%.3f'%best_val_correlation_in_loss))
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Diagonal Plot - Actual vs. Predicted')
    plt.savefig(pic_name)
    plt.close()

def choose_model(model_name):
    if model_name == 'ProtSATT':
        model = ProtSATT(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores
        ).to(device)
    elif (model_name == 'multi_layer_attention_no_self'):
        model = multi_layer_attention_no_self(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).to(device)
    elif (model_name == 'multi_layer_attention_no_cross'):
        model = multi_layer_attention_no_cross(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).to(device)
    elif (model_name == 'multi_layer_attention_2input'):
        model = multi_layer_attention_2input(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).to(device)
    elif (model_name == 'multi_layer_attention_1input'):
        model = multi_layer_attention_1input(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores,
        ).to(device)
    else:
        model = ProtSATT(
                dropout=args.dropout,
                first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                self_deep=args.self_deep, 
                deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                out_scores=args.out_scores
        ).to(device)
    return model

def train(model, dataloader, optim, loss_fn, scheduler):
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
        loss = loss_fn(model_output, y.float())

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

def evaluate_metrics(model, dataloader):
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
            loss = loss_fn(model_output, y.float())

            y_pred.extend(model_output.cpu().detach().numpy())
            y_actual.extend(y.float().detach().cpu().numpy())
            metric_logger.update(loss=loss)
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, y_pred, y_actual

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tc-Riboswitches')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--datadir', type=str, default='/dataset/Tc-Riboswitches/')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--optim', type=str, default='AdamW', choices=('SGD', 'Adam', 'AdamW'))
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--weight_decay', type=float, default=0.003)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--scheduler', type=str, default='LambdaLR', choices=('None', 'StepLR', 'LambdaLR', 'Polynomial', 'CosineAnnealingLR', 'cosine'))
    parser.add_argument('--letter_emb_size', type=int, default=16)

    parser.add_argument('--dropout', type=float, default=0.00001)
    
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
    parser.add_argument('--deep_cross_residual_coef', type=float, default=0.4)

    parser.add_argument('--out_scores', type=int, default=1)
    args = parser.parse_args()

    x_dir_path3 = args.datadir+"x_Tc_esm2_dataset.csv" # esm2
    x_dir_path2 = args.datadir+"x_Tc_protT5_dataset.csv" # protT5
    x_dir_path1 = args.datadir+"x_Tc_unirep_dataset.csv" # unirep
    y_dir_path = args.datadir+"y_Tc_dataset.csv"

    x_dataset1 = np.loadtxt(x_dir_path1, delimiter=",", dtype="float")
    x_dataset2 = np.loadtxt(x_dir_path2, delimiter=",", dtype="float")
    x_dataset3 = np.loadtxt(x_dir_path3, delimiter=",", dtype="float")
    y_dataset = np.loadtxt(y_dir_path, delimiter=",", dtype="float")

    model_name = 'ProtSATT'
    # model_name = 'multi_layer_attention_no_self'
    # model_name = 'multi_layer_attention_no_cross'
    # model_name = 'multi_layer_attention_2input'
    # model_name = 'multi_layer_attention_1input'
    res_dir = rf'/results/TC_regression/{model_name}_one_Self/residue_ablation/firstSelf{args.first_self_residual_coef}_deepSelf{args.deep_self_residual_coef}_cross{args.deep_cross_residual_coef}'

    os.makedirs(res_dir, mode=0o777, exist_ok=True)
    res_model_dir = rf'{res_dir}/save_models'
    os.makedirs(res_model_dir, mode=0o777, exist_ok=True)

    all_fold_loss_list=[]
    # kf = KFold(n_splits=5, random_state=42, shuffle=True)
    # for i, (train, val) in enumerate(kf.split(y_dataset)):
    for i in range(1):
        # i=1
        #Model
        model = choose_model(model_name)

        # Optimizers
        def lambda_lr(s):
            warm_up = args.warmup
            s += 1
            return (args.letter_emb_size ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

        if args.optim == 'Adam' and args.scheduler != 'LambdaLR':
            optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        elif args.optim == 'Adam' and args.scheduler == 'LambdaLR':
            optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        elif args.optim == 'AdamW' and args.scheduler == 'LambdaLR':
            optim = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        else:
            optim = SGD(model.parameters(), lr=args.lr, momentum=0.9)

        if args.scheduler == 'StepLR':
            scheduler = StepLR(optim, step_size=30, gamma=0.5)
        elif args.scheduler == 'LambdaLR':
            scheduler = LambdaLR(optim, lambda_lr)
        else:
            scheduler = None
        # loss_fn = torch.nn.L1Loss()
        loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.NLLLoss()

        best_train_mse = 999.0
        best_val_mse = 999.0
        best_spearmanr = .0
        best_spearmanr_epoch = 0
        patience = 0
        start_epoch = 0
        best_train_epoch = 0
        best_val_epoch = 0
        val_spearmanr_list = []

        x_train1, x_train2, x_train3, y_train, x_val1, x_val2, x_val3, y_val, x_test1, x_test2, x_test3, y_test = split_train_test_Tc(x_dataset1, x_dataset2, x_dataset3, y_dataset)

        y_train = torch.FloatTensor(MinMaxNomalize(y_train))
        y_val = torch.FloatTensor(MinMaxNomalize(y_val))
        y_test = torch.FloatTensor(MinMaxNomalize(y_test))

        dataset_train = MyDataset_3input(x1=x_train1, x2=x_train2, x3=x_train3, y=y_train)
        dataset_val = MyDataset_3input(x1=x_val1, x2=x_val2, x3=x_val3, y=y_val)
        dataset_test = MyDataset_3input(x1=x_test1, x2=x_test2, x3=x_test3, y=y_test)

        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      sampler=train_sampler,
                                      pin_memory=True,
                                      num_workers=args.workers)
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=args.batch_size,
                                    sampler=val_sampler,
                                    pin_memory=True)
        dataloader_test = DataLoader(dataset_test,
                                    batch_size=args.batch_size,
                                    sampler=test_sampler,
                                    pin_memory=True)

        epoch_list = []
        train_loss_list = []
        val_loss_list = []
        for e in range(args.epoch):
            train_mse_loss, train_y_pred, train_y_actual = train(model, dataloader_train, optim, loss_fn, scheduler)
            epoch_list.append(e+1)
            train_loss_list.append(train_mse_loss)
            # wandb.log({"Train loss MSE": train_mse_loss, "epoch": e})
            # _logger.info("Train loss MSE: %s", train_mse_loss)
            print("Train loss MSE: %s", train_mse_loss)
            if scheduler is not None and args.scheduler == 'StepLR':
                scheduler.step()
            
            if best_train_mse > train_mse_loss:
                best_train_mse = train_mse_loss
                best_train_epoch = e
                best_train_y_pred = train_y_pred
                best_train_y_actual = train_y_actual

            val_loss, val_y_pred, val_y_actual = evaluate_metrics(model, dataloader_val)
            val_loss_list.append(val_loss)
            print("Val loss MSE: %s", val_loss)

            if best_val_mse > val_loss:
                best_val_mse = val_loss
                best_val_epoch = e
                best_val_y_pred_in_loss = val_y_pred
                best_val_y_actual_in_loss = val_y_actual
                best_val_correlation_in_loss, best_val_p_value_in_loss = spearmanr(val_y_pred, val_y_actual)
                torch.save(model.state_dict(), rf'{res_model_dir}/best_loss{i}.pth')

            correlation, p_value = spearmanr(val_y_actual, val_y_pred)
            if best_spearmanr < correlation:
                best_spearmanr = correlation
                best_spearmanr_epoch = e
                best_spearmanr_loss = val_loss
                best_val_y_pred_in_spearmanr = val_y_pred
                best_val_y_actual_in_spearmanr = val_y_actual   
                torch.save(model.state_dict(), rf'{res_model_dir}/best_spearmanr{i}.pth')
            val_spearmanr_list.append(correlation)
            print("best_spearmanr: %s", best_spearmanr, " best_spearmanr_epoch: %s", best_spearmanr_epoch)

        # 绘制loss曲线图
        plt.plot(epoch_list, train_loss_list, label='train_loss')
        plt.plot(epoch_list, val_loss_list, label='val_loss')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("metric figure")
        plt.savefig(rf'{res_dir}/loss_fold{i}.png')
        plt.close()
        print(rf"========{i}=========")
        print("best_train_mse", best_train_mse)
        print("best_train_epoch", best_train_epoch)

        # r2score = r2_score(best_val_y_actual, best_val_y_pred)
        # print('r2_score: %.2f' % r2score)
        # print(best_val_y_actual)
        # print(best_val_y_pred)
        # correlation, p_value = spearmanr(best_val_y_actual, best_val_y_pred)
        # print(f"spearman: {correlation:.3f}")
        # print(f"p值: {p_value:.3f}")
        # print(f'best_spearmanr: {best_spearmanr}; epoch: {best_spearmanr_epoch}; val_loss: {best_spearmanr_loss}')
        print(f'spearmanr_in_best_loss: {best_val_correlation_in_loss}; epoch: {best_val_epoch}; val_loss: {best_val_mse}')
        print(f'best_spearmanr: {best_spearmanr}; epoch: {best_spearmanr_epoch}; val_loss: {best_spearmanr_loss}')
        
        # 绘制actual_pred散点图
        plot_scatter(best_val_y_actual_in_loss, best_val_y_pred_in_loss, rf'{res_dir}/spearmanr_in_best_loss{i}.png')    
        plot_scatter(best_val_y_actual_in_spearmanr, best_val_y_pred_in_spearmanr, rf'{res_dir}/best_spearmanr{i}.png')

        # del model
        output_list = []
        output_list.append(f'==========fold{i}============\n')
        output_list.append('best_train_mse: '+str(best_train_mse)+'\n')
        output_list.append('best_train_epoch: '+str(best_train_epoch)+'\n')
        output_list.append(f'===========val==================\n')
        output_list.append(f'spearmanr_in_best_loss: {best_val_correlation_in_loss}; epoch: {best_val_epoch}; val_loss: {best_val_mse}\n')
        output_list.append(f'===========val==================\n')
        output_list.append(f'best_spearmanr: {best_spearmanr}; epoch: {best_spearmanr_epoch}; val_loss: {best_spearmanr_loss}\n')

        with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
            f.writelines(output_list)
        all_fold_loss_list.append(best_val_mse)

    
    # print(np.mean(all_fold_loss_list))
    # with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
    #     f.writelines('all_fold_loss_list: '+str(all_fold_loss_list)+'\n')
    #     f.writelines('mean_loss: '+str(np.mean(all_fold_loss_list))+'\n')
        # test
        model = choose_model(model_name)
        state_dict_best_loss = torch.load(f'{res_model_dir}/best_loss{i}.pth')
        model.load_state_dict(state_dict_best_loss)
        test_loss, test_y_pred, test_y_actual = evaluate_metrics(model, dataloader_test)
        correlation, p_value = spearmanr(test_y_pred, test_y_actual)
        plot_scatter(test_y_actual, test_y_pred, rf'{res_dir}/test_in_best_loss.png')
        with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
            output_list = []
            output_list.append('\n========test best loss model in val=========\n')
            output_list.append('test spearmanr in best val loss: '+str(correlation)+'\n')
            output_list.append('loss: '+str(test_loss)+'\n')
            f.writelines(output_list)
        
        model = choose_model(model_name)
        state_dict_best_searmanr = torch.load(f'{res_model_dir}/best_spearmanr{i}.pth')
        model.load_state_dict(state_dict_best_searmanr)
        test_loss, test_y_pred, test_y_actual = evaluate_metrics(model, dataloader_test)
        correlation, p_value = spearmanr(test_y_pred, test_y_actual)
        plot_scatter(test_y_actual, test_y_pred, rf'{res_dir}/test_in_best_spearmanr.png')
        with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
            output_list = []
            output_list.append('\n========test best spearmanr model in val=========\n')
            output_list.append('test spearmanr in best val spearmnar: '+str(correlation)+'\n')
            output_list.append('loss: '+str(test_loss)+'\n\n')
            f.writelines(output_list)
        
        # with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
        #     f.write(str(val_spearmanr_list))