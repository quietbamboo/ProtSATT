import os
seed=68
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYTHONHASHSEED'] = str(seed)
import argparse
import numpy as np
import pandas as pd
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
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
BEST_EPOCH = 377

def seed_torch(seed=68):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed)

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

def predict(model, dataloader, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    y_actual = list()
    y_pred = list()
    header = 'Evaluation Epoch: Predict'
    with torch.no_grad():
        for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            # y = np_utils.to_categorical(y, 2)
            model_output = model(x1, x2, x3, device=device,
                            first_self_query_dim=args.first_self_query_dim, 
                            deep_self=False, 
                            deep_self_query_dim=args.deep_self_query_dim, 
                            deep_cross_query_dim=args.deep_cross_query_dim)

            y_pred.extend(model_output.cpu().detach().numpy())
            y_actual.extend(y.float().detach().cpu().numpy())

    # actual_pred scatter
    plt.scatter(y_actual, y_pred)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Diagonal Plot - Actual vs. Predicted')
    plt.savefig(rf'./results/predict.png')
    plt.close()

    R2 = r2_score(y_actual, y_pred)
    # AUC
    auc_value = roc_auc_score([1 if y>=0.5 else 0 for y in y_actual], y_pred)
    
    # correct = np.sum(y_pred == y_actual)
    pred_correct = sum([1 for x, y in zip(y_pred, y_actual) if (x>=0.5 and y>=0.5) or (x<0.5 and y<0.5)])
    pred_correct_1 = sum([1 for x, y in zip(y_pred, y_actual) if (x>=0.5 and y>=0.5)])
    pred_correct_0 = sum([1 for x, y in zip(y_pred, y_actual) if (x<0.5 and y<0.5)])
    pred_1 = sum([1 for x, y in zip(y_pred, y_actual) if (x>=0.5)])
    pred_0 = sum([1 for x, y in zip(y_pred, y_actual) if (x<0.5)])

    actual_num = len(y_actual)
    actual_num_1 = sum([1 for x, y in zip(y_pred, y_actual) if y>=0.5])
    actual_num_0 = sum([1 for x, y in zip(y_pred, y_actual) if y<0.5])

    # correct_2 = sum([1 for x, y in zip(y_pred, y_actual) if x==y and x==2])
    Accuracy = pred_correct/actual_num
    Recall = pred_correct_1/actual_num_1
    Precision = pred_correct_1/pred_1
    F1 = 2*(Precision*Recall)/(Precision+Recall)
    print("test accuracy ", pred_correct,"/", actual_num,  Accuracy)
    print("0=== ", pred_correct_0,"/",actual_num_0, pred_correct_0/actual_num_0)
    print("1=== Recall ", pred_correct_1,"/",actual_num_1, Recall)
    print("Precision ", pred_correct_1,"/",pred_1, Precision)
    print("F1 ", F1)
    print("AUC ", auc_value) # 输出AUC值
    print("R2 ", R2)
    # print("2===========", correct_2,"/",y_actual.count(2.), correct_1/y_actual.count(2.))
    # print(utils.accuracy(y_pred, y_actual))
    return Accuracy, Recall, Precision, F1, auc_value, R2

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
    args = parser.parse_args()

    x_dir_path1 = args.datadir+"x_eSol_train_esm2_dataset.csv" # esm2
    x_dir_path2 = args.datadir+"x_eSol_train_protT5_dataset.csv" # protT5
    x_dir_path3 = args.datadir+"x_eSol_train_unirep_dataset.csv" # unirep
    y_dir_path = args.datadir+"y_eSol_train_dataset.csv"
    x_train1 = np.loadtxt(x_dir_path1, delimiter=",", dtype="float")
    x_train2 = np.loadtxt(x_dir_path2, delimiter=",", dtype="float")
    x_train3 = np.loadtxt(x_dir_path3, delimiter=",", dtype="float")
    y_train = np.loadtxt(y_dir_path, delimiter=",", dtype="float")

    x_dir_path1 = args.datadir+"x_eSol_test_esm2_dataset.csv" # esm2
    x_dir_path2 = args.datadir+"x_eSol_test_protT5_dataset.csv" # protT5
    x_dir_path3 = args.datadir+"x_eSol_test_unirep_dataset.csv" # unirep
    y_dir_path = args.datadir+"y_eSol_test_dataset.csv"
    x_test1 = np.loadtxt(x_dir_path1, delimiter=",", dtype="float")
    x_test2 = np.loadtxt(x_dir_path2, delimiter=",", dtype="float")
    x_test3 = np.loadtxt(x_dir_path3, delimiter=",", dtype="float")
    y_test = np.loadtxt(y_dir_path, delimiter=",", dtype="float")

    model_name = 'ProtSATT'
    # model_name = 'multi_layer_attention_no_self'
    # model_name = 'multi_layer_attention_no_cross'
    # model_name = 'multi_layer_attention_2input'
    # model_name = 'multi_layer_attention_1input'

    res_dir = rf'/results/eSOL/{model_name}_one_self/firstSelf{args.first_self_residual_coef}_deepSelf{args.deep_self_residual_coef}_cross{args.deep_cross_residual_coef}'

    os.makedirs(res_dir, mode=0o777, exist_ok=True)
    res_model_dir = rf'{res_dir}/save_models'
    os.makedirs(res_model_dir, mode=0o777, exist_ok=True)

    #Model
    model = choose_model(model_name, args)
    # Optimizers
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
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr

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
    start_epoch = 0
    best_train_epoch = 0

    # ====================================================
    dataset_train = MyDataset_3input(x1=x_train1, x2=x_train2, x3=x_train3, y=y_train)
    dataset_test = MyDataset_3input(x1=x_test1, x2=x_test2, x3=x_test3, y=y_test)

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  pin_memory=True,
                                  num_workers=args.workers)
    dataloader_test = DataLoader(dataset_test,
                                batch_size=args.batch_size,
                                sampler=test_sampler,
                                pin_memory=True)

    epoch_list = []
    train_loss_list = []
    for e in range(args.epoch):
        train_mse_loss, train_y_pred, train_y_actual = train(model, dataloader_train, optim, loss_fn, scheduler, args, e)
        epoch_list.append(e+1)
        train_loss_list.append(train_mse_loss)
        print("Train loss MSE: %s", train_mse_loss)
        if scheduler is not None and args.scheduler == 'StepLR':
            scheduler.step()
        if best_train_mse > train_mse_loss:
            best_train_mse = train_mse_loss
            # best_train_epoch = e
            # best_train_y_pred = train_y_pred
            # best_train_y_actual = train_y_actual
        if e==BEST_EPOCH:
            torch.save(model.state_dict(), rf'{res_model_dir}/best_epoch_{BEST_EPOCH}.pth')

    # test
    model = choose_model(model_name, args)
    state_dict_best_loss = torch.load(f'{res_model_dir}/best_epoch_{BEST_EPOCH}.pth')
    model.load_state_dict(state_dict_best_loss)
    Accuracy, Recall, Precision, F1, auc_value, R2 = predict(model, dataloader_test, args)
    with open(rf'{res_save_dir}/test_output_res.txt', 'a') as f:
        output_list = []
        output_list.append(f'\n\n{res_model_dir}/best_epoch_{BEST_EPOCH}.pth')
        output_list.append('\n======== model in test=========\n')
        output_list.append('Accuracy: '+str(Accuracy)+'\n')
        output_list.append('Recall: '+str(Recall)+'\n')
        output_list.append('Precision: '+str(Precision)+'\n')
        output_list.append('F1: '+str(F1)+'\n')
        output_list.append('AUC: '+str(auc_value)+'\n')
        output_list.append('R2: '+str(R2)+'\n')
        f.writelines(output_list)

if __name__ == '__main__':
    main(0.0006)