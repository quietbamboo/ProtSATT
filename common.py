from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class MyDataset_1input(Dataset):
    def __init__(self, x1, y):
        self.x1 = x1
        self.y = y

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.y[idx]

class MyDataset_2input(Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

class MyDataset_3input(Dataset):
    def __init__(self, x1, x2, x3, y):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y = y

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.y[idx]

class MyDataset_4input(Dataset):
    def __init__(self, x1, x2, x3, x4, y):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.y = y

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.x4[idx], self.y[idx]


# 数据集划分
def split_train_test(x_dataset1, x_dataset2, y_dataset, test_ratio, seed):
    if len(x_dataset1) == len(y_dataset):
        #设置随机数种子，保证每次生成的结果都是一样的
        # 42
        np.random.seed(seed)
        #permutation随机生成0-len(data)随机序列
        shuffled_indices = np.random.permutation(len(x_dataset1))
        #test_ratio为测试集所占的半分比
        test_set_size = int(len(x_dataset1) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        #iloc选择参数序列中所对应的行
        return torch.FloatTensor(x_dataset1[train_indices]), \
               torch.FloatTensor(x_dataset2[train_indices]), \
               torch.FloatTensor(y_dataset[train_indices]), \
               torch.FloatTensor(x_dataset1[test_indices]), \
               torch.FloatTensor(x_dataset2[test_indices]), \
               torch.FloatTensor(y_dataset[test_indices])
    else:
        raise ValueError('length of x and y is not same')

# 数据集划分-交叉验证
# num_fold 第几折
def split_train_test_fold(x_dataset1, x_dataset2, x_dataset3, y_dataset, Cross_Fold, num_fold):
    if len(x_dataset1) == len(y_dataset):
        # 每个fold的大小
        fold_size = round(len(x_dataset1)/Cross_Fold)
        x_dataset1 = list(x_dataset1)
        x_dataset2 = list(x_dataset2)
        x_dataset3 = list(x_dataset3)
        y_dataset = list(y_dataset)
        if(num_fold!=10):
            x_dataset1_test = x_dataset1[fold_size*(num_fold-1):fold_size*num_fold]
            x_dataset1_train = x_dataset1[0:fold_size*(num_fold-1)]
            x_dataset1_train.extend(x_dataset1[fold_size*num_fold:])
            x_dataset2_test = x_dataset2[fold_size*(num_fold-1):fold_size*num_fold]
            x_dataset2_train = x_dataset2[0:fold_size*(num_fold-1)]
            x_dataset2_train.extend(x_dataset2[fold_size*num_fold:])
            x_dataset3_test = x_dataset3[fold_size*(num_fold-1):fold_size*num_fold]
            x_dataset3_train = x_dataset3[0:fold_size*(num_fold-1)]
            x_dataset3_train.extend(x_dataset3[fold_size*num_fold:])

            y_dataset_test = y_dataset[fold_size*(num_fold-1):fold_size*num_fold]
            y_dataset_train = y_dataset[0:fold_size*(num_fold-1)]
            y_dataset_train.extend(y_dataset[fold_size*num_fold:])
        else:
            x_dataset1_test = x_dataset1[fold_size*(num_fold-1):]
            x_dataset1_train = x_dataset1[0:fold_size*(num_fold-1)]
            x_dataset2_test = x_dataset2[fold_size*(num_fold-1):]
            x_dataset2_train = x_dataset2[0:fold_size*(num_fold-1)]
            x_dataset3_test = x_dataset3[fold_size*(num_fold-1):]
            x_dataset3_train = x_dataset3[0:fold_size*(num_fold-1)]

            y_dataset_test = y_dataset[fold_size*(num_fold-1):]
            y_dataset_train = y_dataset[0:fold_size*(num_fold-1)]

        #iloc选择参数序列中所对应的行
        return torch.FloatTensor(np.array(x_dataset1_train)), \
               torch.FloatTensor(np.array(x_dataset2_train)), \
               torch.FloatTensor(np.array(x_dataset3_train)), \
               torch.FloatTensor(np.array(y_dataset_train)), \
               torch.FloatTensor(np.array(x_dataset1_test)), \
               torch.FloatTensor(np.array(x_dataset2_test)), \
               torch.FloatTensor(np.array(x_dataset3_test)), \
               torch.FloatTensor(np.array(y_dataset_test))
        # return x_dataset1_train, \
        #        x_dataset2_train, \
        #        x_dataset3_train, \
        #        y_dataset_train, \
        #        x_dataset1_test, \
        #        x_dataset2_test, \
        #        x_dataset3_test, \
        #        y_dataset_test
    else:
        raise ValueError('length of x and y is not same')

# 数据集划分
def split_train_test_Tc(x_dataset1, x_dataset2, x_dataset3, y_dataset, test_length=54):
    if len(x_dataset1) == len(y_dataset):
        # 每个fold的大小
        fold_size = round(len(x_dataset1)/10)
        x_dataset1 = list(x_dataset1)
        x_dataset2 = list(x_dataset2)
        x_dataset3 = list(x_dataset3)
        y_dataset = list(y_dataset)
 
        x_dataset1_test = x_dataset1[-test_length:]
        x_dataset1_val = x_dataset1[-(test_length*2):-test_length]
        x_dataset1_train = x_dataset1[0:-(test_length*2)]

        x_dataset2_test = x_dataset2[-test_length:]
        x_dataset2_val = x_dataset2[-(test_length*2):-test_length]
        x_dataset2_train = x_dataset2[0:-(test_length*2)]

        x_dataset3_test = x_dataset3[-test_length:]
        x_dataset3_val = x_dataset3[-(test_length*2):-test_length]    
        x_dataset3_train = x_dataset3[0:-(test_length*2)]

        y_dataset_test = y_dataset[-test_length:]
        y_dataset_val = y_dataset[-(test_length*2):-test_length]
        y_dataset_train = y_dataset[0:-(test_length*2)]

        #iloc选择参数序列中所对应的行
        return torch.FloatTensor(np.array(x_dataset1_train)), \
               torch.FloatTensor(np.array(x_dataset2_train)), \
               torch.FloatTensor(np.array(x_dataset3_train)), \
               torch.FloatTensor(np.array(y_dataset_train)), \
               torch.FloatTensor(np.array(x_dataset1_val)), \
               torch.FloatTensor(np.array(x_dataset2_val)), \
               torch.FloatTensor(np.array(x_dataset3_val)), \
               torch.FloatTensor(np.array(y_dataset_val)), \
               torch.FloatTensor(np.array(x_dataset1_test)), \
               torch.FloatTensor(np.array(x_dataset2_test)), \
               torch.FloatTensor(np.array(x_dataset3_test)), \
               torch.FloatTensor(np.array(y_dataset_test))
        # return x_dataset1_train, \
        #        x_dataset2_train, \
        #        x_dataset3_train, \
        #        y_dataset_train, \
        #        x_dataset1_val, \
        #        x_dataset2_val, \
        #        x_dataset3_val, \
        #        y_dataset_val, \
        #        x_dataset1_test, \
        #        x_dataset2_test, \
        #        x_dataset3_test, \
        #        y_dataset_test
    else:
        raise ValueError('length of x and y is not same')


def train_MinMaxNomalize(data):
    # 创建一个二维张量
    # x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # 沿着行的方向计算最小值和最大值
    min_vals, _ = torch.min(data, dim=1, keepdim=True)
    max_vals, _ = torch.max(data, dim=1, keepdim=True)
    
    # 最小-最大缩放，将x的范围缩放到[0, 1]
    scaled_x = (data - min_vals) / (max_vals - min_vals)
    
    print(scaled_x)
    return scaled_x

def transform_to_tensor(pd_series):
    return torch.from_numpy(np.array(pd_series.to_list()))

