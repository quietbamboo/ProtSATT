import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
import utils
from Model import ProtSATT
import torch
from common import MyDataset_3input
import os
from sklearn.metrics import r2_score, roc_auc_score

os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(68)
np.random.seed(68)
_print_freq = 50
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate_metrics(model, dataloader, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    predictions_test = list()
    y_actual = list()
    loss_mse = list()
    y_pred = list()
    header = 'Evaluation: Predict'
    with torch.no_grad():
        for it, (x1, x2, x3, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
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

def evaluate_metrics_cls(model, dataloader, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    predictions_test = list()
    y_actual = list()
    loss_mse = list()
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
            loss = loss_fn(model_output, y)

            y_pred.extend(model_output.cpu().detach().numpy())
            y_actual.extend(y.float().detach().cpu().numpy())
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
    # print("2===========", correct_2,"/",y_actual.count(2.), correct_1/y_actual.count(2.))
    # print(utils.accuracy(y_pred, y_actual))
    return Accuracy, Recall, Precision, F1, auc_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict_regress')
    parser.add_argument('--datadir', type=str, default='/datasets/eSOL/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)

    parser.add_argument('--dropout', type=float, default=0.2)
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

    parser.add_argument('--out_scores', type=int, default=1)
    args = parser.parse_args()

    # SC_test
    # x_dir_path3 = args.datadir+"x_S.cerevisiae_test_esm2_dataset.csv" # esm2
    # x_dir_path2 = args.datadir+"x_S.cerevisiae_test_protT5_dataset.csv" # protT5
    # x_dir_path1 = args.datadir+"x_S.cerevisiae_test_unirep_dataset.csv" # unirep
    # y_dir_path = args.datadir+"y_S.cerevisiae_test_esm2_dataset.csv"

    # eSOL_test
    x_dir_path3 = args.datadir+"x_eSol_test_esm2_dataset.csv" # esm2
    x_dir_path2 = args.datadir+"x_eSol_test_protT5_dataset.csv" # protT5
    x_dir_path1 = args.datadir+"x_eSol_test_unirep_dataset.csv" # unirep
    y_dir_path = args.datadir+"y_eSol_test_esm2_dataset.csv"

    x_test1 = np.loadtxt(x_dir_path1, delimiter=",", dtype="float")
    x_test2 = np.loadtxt(x_dir_path2, delimiter=",", dtype="float")
    x_test3 = np.loadtxt(x_dir_path3, delimiter=",", dtype="float")
    y_test = np.loadtxt(y_dir_path, delimiter=",", dtype="float")

    model_name = 'ProtSATT'
    res_model_dir = rf'/results/eSOL/multi_layer_attention_one_self/residue_ablation/lr0.0006_epoch500_warmup150_letterEmbSize16_schedulerLambdaLR_weightDecay0.003_dropout0.2_firstSelf0_deepSelf0_cross1_A2000_/save_models/best_epoch_478_1.pth'
    res_save_dir = rf'/results/eSOL_predict/eSOL_cls'
    os.makedirs(res_save_dir, mode=0o777, exist_ok=True)

    loss_fn = torch.nn.MSELoss()
    dataset_test = MyDataset_3input(x1=x_test1, x2=x_test2, x3=x_test3, y=y_test)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test,
                                batch_size=args.batch_size,
                                sampler=test_sampler,
                                pin_memory=True)

    # test
    model = ProtSATT(
            dropout=args.dropout,
            first_self_query_dim=args.first_self_query_dim, first_self_return_dim=args.first_self_return_dim, first_self_num_head=args.first_self_num_head, first_self_dropout=args.first_self_dropout, first_self_residual_coef=args.first_self_residual_coef,
            self_deep=args.self_deep, 
            deep_self_query_dim=args.deep_self_query_dim, deep_self_return_dim=args.deep_self_return_dim, deep_self_num_head=args.deep_self_num_head, deep_self_dropout=args.deep_self_dropout, deep_self_residual_coef=args.deep_self_residual_coef,
            deep_cross_query_dim=args.deep_cross_query_dim, deep_cross_return_dim=args.deep_cross_return_dim, deep_cross_num_head=args.deep_cross_num_head, deep_cross_dropout=args.deep_cross_dropout, deep_cross_residual_coef=args.deep_cross_residual_coef,
            out_scores=args.out_scores
            ).double().to(device)
    state_dict_best_loss = torch.load(f'{res_model_dir}')
    model.load_state_dict(state_dict_best_loss)

    # SC_test
    # test_loss, test_y_pred, test_y_actual = evaluate_metrics(model, dataloader_test, args)
    # # correlation, p_value = spearmanr(test_y_pred, test_y_actual)
    # correlation = r2_score(test_y_actual, test_y_pred)
    # with open(rf'{res_save_dir}/predict_output_res.txt', 'a') as f:
    #     output_list = []
    #     output_list.append(f'\n\n{res_model_dir}')
    #     output_list.append('\n======== model in test=========\n')
    #     output_list.append('test r2: '+str(correlation)+'\n')
    #     output_list.append('loss: '+str(test_loss)+'\n')
    #     f.writelines(output_list)

    # eSOL_test
    Accuracy, Recall, Precision, F1, auc_value = evaluate_metrics_cls(model, dataloader_test, args)
    with open(rf'{res_save_dir}/predict_output_res.txt', 'a') as f:
        output_list = []
        output_list.append(f'\n\n{res_model_dir}')
        output_list.append('\n======== model in test=========\n')
        output_list.append('Accuracy: '+str(Accuracy)+'\n')
        output_list.append('Recall: '+str(Recall)+'\n')
        output_list.append('Precision: '+str(Precision)+'\n')
        output_list.append('F1: '+str(F1)+'\n')
        output_list.append('AUC: '+str(auc_value)+'\n')
        f.writelines(output_list)


    # with open(rf'{res_dir}/train_output_res.txt', 'a') as f:
    #     f.write(str(val_spearmanr_list))