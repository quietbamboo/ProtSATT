import os
seed = 68
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYTHONHASHSEED'] = str(seed)
import argparse
import numpy as np
import pandas as pd
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
import utils
from Model import *
import torch
from common import MyDataset_3input, split_train_test_fold
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import KFold
# matplotlib.use('TkAgg')
# from keras.utils import np_utils
# from sklearn.metrics import classification_report

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

def train(model, dataloader, optim, loss_fn, scheduler, args, e):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    y_pred = list()
    y_test = list()
    header = 'Training Epoch: [{}]'.format(e)
    for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        # y = np_utils.to_categorical(y, 2)
        # x torch.Size([300, 1280])
        model_output = model(x1, x2, x3, device=device, 
                            first_self_query_dim=args.first_self_query_dim, 
                            deep_self=True, 
                            deep_self_query_dim=args.deep_self_query_dim, 
                            deep_cross_query_dim=args.deep_cross_query_dim)
        loss = loss_fn(model_output, y.long())

        y_pred.extend(np.argmax(model_output.cpu().detach().numpy(), axis = 1).astype(np.float32))
        y_test.extend(y.float().detach().cpu().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler is not None and args.scheduler == 'LambdaLR' or args.scheduler == 'Cos':
            scheduler.step()

        metric_logger.update(loss=loss)

    correct = sum([1 for x, y in zip(y_pred, y_test) if x==y])
    correct_0 = sum([1 for x, y in zip(y_pred, y_test) if x==y and x==0])
    correct_1 = sum([1 for x, y in zip(y_pred, y_test) if x==y and x==1])

    print("train", correct,"/",len(y_test), correct/len(y_test))
    print("0===========", correct_0,"/",y_test.count(0.), correct_0/y_test.count(0.))
    print("1===========", correct_1,"/",y_test.count(1.), correct_1/y_test.count(1.))

    metric_logger.  synchronize_between_processes()
    return metric_logger.loss.global_avg, correct_0/y_test.count(0.), correct_1/y_test.count(1.), correct/len(y_test)

def evaluate_metrics(model, dataloader, loss_fn, args, e):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    predictions_test = list()
    y_test = list()
    loss_mse = list()
    y_pred = list()

    header = 'Evaluation Epoch: [{}]'.format(e)
    with torch.no_grad():
        for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            # y = np_utils.to_categorical(y, 2)
            model_output = model(x1, x2, x3, device=device,
                            first_self_query_dim=args.first_self_query_dim, 
                            deep_self=True, 
                            deep_self_query_dim=args.deep_self_query_dim, 
                            deep_cross_query_dim=args.deep_cross_query_dim)
            loss = loss_fn(model_output, y.long())

            y_pred.extend(np.argmax(model_output.cpu().detach().numpy(), axis = 1).astype(np.float32))
            y_test.extend(y.float().detach().cpu().numpy())

    # correct = np.sum(y_pred == y_test)
    correct = sum([1 for x, y in zip(y_pred, y_test) if x==y])
    correct_0 = sum([1 for x, y in zip(y_pred, y_test) if x==y and x==0])
    correct_1 = sum([1 for x, y in zip(y_pred, y_test) if x==y and x==1])
    print("test", correct,"/",len(y_test), correct/len(y_test))
    print("0===========", correct_0,"/",y_test.count(0.), correct_0/y_test.count(0.))
    print("1===========", correct_1,"/",y_test.count(1.), correct_1/y_test.count(1.))
    # print(utils.accuracy(y_pred, y_test))
    return utils.accuracy(y_pred, y_test), utils.f1_score(y_pred, y_test), correct_0/y_test.count(0.), correct_1/y_test.count(1.), loss

def main(lr):
    parser = argparse.ArgumentParser(description='SOLUABLE')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--datadir', type=str, default='/dataset/EColi/')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--optim', type=str, default='AdamW', choices=('SGD', 'Adam', 'AdamW'))
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--weight_decay', type=float, default=0.0003)
    parser.add_argument('--warmup', type=int, default=150)
    parser.add_argument('--scheduler', type=str, default='LambdaLR', choices=('None', 'StepLR', 'LambdaLR'))
    parser.add_argument('--letter_emb_size', type=int, default=32)

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

    x_dir_path3 = args.datadir+"x_esm2_dataset.csv" # esm2
    x_dir_path2 = args.datadir+"x_protT5_dataset.csv" # protT5
    x_dir_path1 = args.datadir+"x_unirep_dataset.csv" # unirep
    y_dir_path = args.datadir+"y_dataset.csv"

    x_dataset1 = np.loadtxt(x_dir_path1, delimiter=",", dtype="float")
    x_dataset2 = np.loadtxt(x_dir_path2, delimiter=",", dtype="float")
    x_dataset3 = np.loadtxt(x_dir_path3, delimiter=",", dtype="float")
    y_dataset = np.loadtxt(y_dir_path, delimiter=",", dtype="float")
    seed = args.seed

    model_name = 'ProtSATT'
    # model_name = 'multi_layer_attention_no_self'
    # model_name = 'multi_layer_attention_no_cross'
    # model_name = 'multi_layer_attention_2input'
    res_dir = rf'/results/E_Coli/{model_name}/residue_ablation/firstSelf{args.first_self_residual_coef}_deepSelf{args.deep_self_residual_coef}_cross{args.deep_cross_residual_coef}'
    os.makedirs(res_dir, mode=0o777, exist_ok=True)
    res_model_dir = rf'{res_dir}/save_models'
    os.makedirs(res_model_dir, mode=0o777, exist_ok=True)

    test_acc_list=[]
    # kf = KFold(n_splits=5, random_state=42, shuffle=True)
    # for i, (train, val) in enumerate(kf.split(y_dataset)):
    Cross_Fold=10
    for i in range(1,Cross_Fold+1):
        # i=1
        #Model
        if model_name == 'ProtSATT':
            model = ProtSATT(
                    dropout=args.dropout,
                    first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
                    self_deep=args.self_deep, 
                    deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
                    deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
                    out_scores=args.out_scores,
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

        # loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.NLLLoss()
        patience = 0
        start_epoch = 0

        best_train_acc_0 = 0
        best_train_acc_1 = 0
        best_train_acc_0_in_f1 = 0
        best_train_acc_1_in_f1 = 0

        best_test_acc = 0
        best_acc_epoch = 0
        best_test_acc_0 = 0
        best_test_acc_1 = 0

        best_test_f1_score = 0
        best_f1_epoch = 0
        best_acc_in_best_f1 = 0
        best_test_acc_0_in_f1 = 0
        best_test_acc_1_in_f1 = 0

        x_train1, x_train2, x_train3, y_train, x_test1, x_test2, x_test3, y_test = split_train_test_fold(x_dataset1, x_dataset2, x_dataset3, y_dataset, Cross_Fold, i)

        dataset_train = MyDataset_3input(x1=x_train1, x2=x_train2, x3=x_train3, y=y_train)
        dataset_val = MyDataset_3input(x1=x_test1, x2=x_test2, x3=x_test3, y=y_test)

        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      sampler=train_sampler,
                                      pin_memory=True,
                                      num_workers=args.workers)
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=args.batch_size,
                                    sampler=val_sampler,
                                    pin_memory=True)

        epoch_list = []
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        for e in range(args.epoch):
            train_mse_loss, acc_train_temp_0, acc_train_temp_1, train_acc = train(model, dataloader_train, optim, loss_fn, scheduler, args, e)
            epoch_list.append(e+1)
            train_loss_list.append(train_mse_loss)
            train_acc_list.append(train_acc)
            print("Train loss MSE: %s", train_mse_loss)
            if scheduler is not None and args.scheduler == 'StepLR':
                scheduler.step()

            val_acc_temp, f1_score_temp, acc_temp_0, acc_temp_1, val_loss = evaluate_metrics(model, dataloader_val, loss_fn, args, e)
            print(val_loss)
            val_loss_list.append(val_loss.cpu())
            val_acc_list.append(val_acc_temp)
            if val_acc_temp>best_test_acc:
                best_train_acc_0 = acc_train_temp_0
                best_train_acc_1 = acc_train_temp_1
                best_test_acc = val_acc_temp
                best_acc_epoch = e
                best_test_acc_0 = acc_temp_0
                best_test_acc_1 = acc_temp_1
                torch.save(model.state_dict(), f'{res_model_dir}/best_acc{i}.pth')
            if f1_score_temp > best_test_f1_score:
                best_train_acc_0_in_f1 = acc_train_temp_0
                best_train_acc_1_in_f1 = acc_train_temp_1
                best_test_f1_score = f1_score_temp
                best_f1_epoch = e
                best_acc_in_best_f1 = val_acc_temp
                best_test_acc_0_in_f1 = acc_temp_0
                best_test_acc_1_in_f1 = acc_temp_1
                # 绘制loss曲线图
        plt.plot(epoch_list, train_loss_list, label='train_loss')
        plt.plot(epoch_list, val_loss_list, label='val_loss')
        plt.plot(epoch_list, train_acc_list, label='train_acc')
        plt.plot(epoch_list, val_acc_list, label='val_acc')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("metric figure")
        # plt.savefig(rf'{res_dir}/loss_fold{i}.png')
        plt.close()
        print("=================")
        print("best_train_acc_0", best_train_acc_0)
        print("best_train_acc_1", best_train_acc_1)
        print("best_train_acc_0_in_f1", best_train_acc_0_in_f1)
        print("best_train_acc_1_in_f1", best_train_acc_1_in_f1)
        print("=================")
        print("best_test_acc", best_test_acc)
        print("best_epoch", best_acc_epoch)
        print("best_test_acc_0", best_test_acc_0)
        print("best_test_acc_1", best_test_acc_1)
        print("=================")
        print("best_test_f1_score", best_test_f1_score)
        print("best_acc_in_best_f1", best_acc_in_best_f1)
        print("best_test_acc_0_in_f1", best_test_acc_0_in_f1)
        print("best_test_acc_1_in_f1", best_test_acc_1_in_f1)
        print("best_f1_epoch", best_f1_epoch)
        # del model
        output_list = []
        count_test_0 = y_test.tolist().count(0.)
        count_test_1 = y_test.tolist().count(1.)
        output_list.append(f'==========fold{i}============\n')
        output_list.append('y_test.count(0.): '+str(count_test_0)+'\n')
        output_list.append('y_test.count(1.): '+str(count_test_1)+'\n')
        output_list.append('=================\n')
        output_list.append('best_train_acc_0: '+str(best_train_acc_0)+'\n')
        output_list.append('best_train_acc_1: '+str(best_train_acc_1)+'\n')
        output_list.append('best_train_acc_0_in_f1: '+str(best_train_acc_0_in_f1)+'\n')
        output_list.append('best_train_acc_1_in_f1: '+str(best_train_acc_1_in_f1)+'\n')
        output_list.append('=================\n')
        output_list.append('best_test_acc: '+str(best_test_acc)+'\n')
        output_list.append('best_epoch: '+str(best_acc_epoch)+'\n')
        output_list.append('best_test_acc_0: '+str(best_test_acc_0)+'\n')
        output_list.append('best_test_acc_1: '+str(best_test_acc_1)+'\n')
        output_list.append('=================\n')
        output_list.append('best_test_f1_score: '+str(best_test_f1_score)+'\n')
        output_list.append('best_acc_in_best_f1: '+str(best_acc_in_best_f1)+'\n')
        output_list.append('best_test_acc_0_in_f1: '+str(best_test_acc_0_in_f1)+'\n')
        output_list.append('best_test_acc_1_in_f1: '+str(best_test_acc_1_in_f1)+'\n')
        output_list.append('best_f1_epoch: '+str(best_f1_epoch)+'\n\n')
        with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
            f.writelines(output_list)

        test_acc_list.append(best_test_acc)
    print(np.mean(test_acc_list))
    with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
        f.writelines('mean_acc: '+str(np.mean(test_acc_list)))

if __name__ == '__main__':
    main(0.001)
