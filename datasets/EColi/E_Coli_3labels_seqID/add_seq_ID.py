import pandas as pd
import os

def merge_fasta_ids_with_features(fasta_path, csv_path, output_path):
    print(f"🔍 正在解析 FASTA 文件: {fasta_path}")

    fasta_ids = []
    # 提取 FASTA 中的序列 ID
    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(">"):
                # 剥离 '>' 号和换行符 (\n 或 \r\n)
                seq_id = line.strip()[1:]
                fasta_ids.append(seq_id)

    print(f"✅ 共提取到 {len(fasta_ids)} 个序列 ID (示例: {fasta_ids[0]})")

    print(f"⏳ 正在加载 ESM2 特征矩阵: {csv_path} (请稍候...)")
    # 加载特征 CSV (注意：因为原文件全是特征，没有列名，必须用 header=None)
    df = pd.read_csv(csv_path, header=None)
    print(f"✅ 特征矩阵加载成功，形状: {df.shape}")

    # 【核心安全校验】确保 FASTA 序列数与 CSV 行数绝对一致！
    if len(fasta_ids) != len(df):
        raise ValueError(f"❌ 致命错误: FASTA 数量 ({len(fasta_ids)}) 与 CSV 行数 ({len(df)}) 不匹配！")

    # 将提取出的 ID 作为第一列 (索引 0) 插入到 DataFrame 中
    # 列名命名为 'Sequence_ID'
    df.insert(0, 'Sequence_ID', fasta_ids)

    print(f"💾 正在保存带有 ID 的新特征矩阵至: {output_path}")
    # 保存结果。如果您希望保留列名 'Sequence_ID' 及 0,1,2... 的特征列名，使用 header=True
    # 如果您希望它还是一个纯净的数据矩阵，只有第一列变成字符串，建议使用 header=False
    df.to_csv(output_path, index=False, header=False)

    print("🎉 合并完美完成！")

if __name__ == "__main__":
    # 请根据您的实际路径修改以下三个变量
    feature = ['esm2', 'protT5', 'unirep']
    FASTA_FILE = r"I:\A_Graguation\ProtSATT\ACS\conclusion\datasets\EColi\EColi_dataset_amino_acid.fasta"
    INPUT_CSV = rf"I:\A_Graguation\ProtSATT\ACS\conclusion\datasets\EColi\E_Coli_3labels\x_EColi_{feature[2]}_dataset.csv"  # 您原本的纯数字特征文件
    OUTPUT_CSV = rf"I:\A_Graguation\ProtSATT\ACS\conclusion\datasets\EColi\E_Coli_3labels_seqID\x_EColi_{feature[2]}_dataset_with_IDs.csv"  # 融合后生成的新文件

    merge_fasta_ids_with_features(FASTA_FILE, INPUT_CSV, OUTPUT_CSV)