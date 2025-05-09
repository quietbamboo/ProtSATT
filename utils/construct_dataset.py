import numpy as np
import pandas as pd
import json
import h5py

def get_gene_class():
    seq = {}
    f = open('/datasets/test_dataset.fasta')
    for line in f:
        if line.startswith('>'):
            # gene_name = line.replace('>','').split('_')[0]
            evaluation_scores = line.replace('>','').replace('\n','').split('_')[1]
            seq[line.replace('>','').replace('\n','')] = evaluation_scores
            # print('seq', seq)
        # else:
        #     seq[name]+=line.replace('\n','').strip()
    f.close()
    return seq
# ================== unirep ======================
def load_unirep(path):
    unirep_data = np.load(path+'/test_dataset_unirep.npz', allow_pickle=True)
    # gene = np.load(npz_file, allow_pickle=True)
    # gene_name_list = gene.files
    # 4281条数据 (key：gene_name；value：class)
    gene_name2score_dict = get_gene_class()
    # 数据
    x_dataset = []
    # 标签
    y_dataset = []
    for key, value in gene_name2score_dict.items():
        x_dataset.append(unirep_data[key].item()['avg'])
        if float(value)>=0.5:
            y_dataset.append(1)
        else:
            y_dataset.append(0)

    return np.array(x_dataset), np.array(y_dataset)

# ================== esm2 ======================
def load_json_file(path):
    with open(path+"/test_dataset_esm2.json", 'r') as f:
        result = json.load(f)
        return result

def load_esm2(path):
    protein_dict = load_json_file(path)
    # 数据
    x_dataset = []
    # 标签
    y_dataset = []
    gene_name2score_dict = get_gene_class()
    for key, value in gene_name2score_dict.items():
        x_dataset.append(protein_dict[key])
        y_dataset.append(value)
    return np.array(x_dataset), np.array(y_dataset)

# ================== protT5 ====================
def load_protT5(path):
    path = path+'/test_dataset_protT5.h5'
    embedding_matrix = h5py.File(path, 'r')
    protein_keys = list(embedding_matrix.keys())
    # print(protein_keys)
    embedding_dict = dict()
    for key in protein_keys:
        name = key
        # name = key.split('_')[0]+'_'+key.split('_')[1]
        # print(key)
        # name = key.split('_')[0]+'_'+key.split('_')[1]+'.'+key.split('_')[2]
        embedding_dict[name] = np.array(embedding_matrix[key])
    # 数据
    x_dataset = []
    # 标签
    y_dataset = []
    gene_name2score_dict = get_gene_class()
    for key, value in gene_name2score_dict.items():
        protT5 = embedding_dict[key.replace('.','_')]
        label = value
        x_dataset.append(protT5)
        y_dataset.append(label)
    return np.array(x_dataset), np.array(y_dataset)

# =================== construct csv ===================
def construct_csv(x_dataset, y_dataset, path, file_name):
    # 构建数据集
    print(len(x_dataset))
    x_dataset_csv = pd.DataFrame(x_dataset)
    x_dataset_csv.to_csv(path+f'/x_{file_name}_dataset.csv', header=None, index=None)
    y_dataset_csv = pd.DataFrame(y_dataset)
    y_dataset_csv.to_csv(path+f'/y_{file_name}_dataset.csv', header=None, index=None)

if __name__ == '__main__':
    path = rf'../datasets/'
    save_path = '../datasets'
    save_name = 'test'
    unirep_x, unirep_y = load_unirep(path)
    print(unirep_x.shape)
    print(unirep_y.shape)
    construct_csv(unirep_x, unirep_y, save_path, rf'{save_name}_unirep')

    esm2_x, esm2_y = load_esm2(path)
    print(esm2_x.shape)
    print(esm2_y.shape)
    construct_csv(esm2_x, esm2_y, save_path, rf'{save_name}_esm2')

    protT5_x, protT5_y = load_protT5(path)
    print(protT5_x.shape)
    print(protT5_y.shape)
    construct_csv(protT5_x, protT5_y, save_path, rf'{save_name}_protT5')
