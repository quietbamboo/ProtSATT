# ProtSATT
ProtSATT: An Advanced Protein Solubility Predictor Based on Attention Mechanism

## Requirements

To set up the necessary environment for ProtSATT, follow these steps:

1. **Create and Activate Conda Environment**

```
conda create -n ProtSATT python=3.10.12
conda activate ProtSATT
```

2. **Install Python Packages**

```
pip install keras==3.2.1
pip install Keras-Applications==1.0.2
pip install Keras-Preprocessing==1.0.1
pip install scikit-learn==1.3.0
pip install scipy==1.11.2
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118
```

## Usage

### Feature extraction

**The UniRep repository**, available at https://github.com/churchlab/UniRep, contains the source code, documentation, and examples for using UniRep in your projects.
Due to the installation of the `tape_proteins` toolkit, you can conveniently utilize UniRep as follows:

```
tape-embed unirep input.fasta output.npz babbler-1900 --tokenizer unirep
```

**The ESM-2 repository**, available at https://github.com/facebookresearch/esm.
When you need to use ESM-2 feature vectors, you can use the `utils/run_esm2.py` program to convert the `.fasta` format file to the `.json` format.
**The ProtTrans repository**, available at https://github.com/agemagician/ProtTrans.
When you need to use ProtT5 feature vectors, you can use the `utils/run_protT5.py` program to convert the `.fasta` format file to the `.json` format.
Before using it, you must download all files from https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main.

Once you have obtained the embedding files from UniRep, ESM-2, and ProtT5, use the `utils/construct_dataset.py` file to build the dataset and generate a CSV file. 

### Train and test

- Run the `train_eColi.py` file to train and validate the E. coli dataset. 
- Run the `train_eSOL_fold.py` file to optimize the hyperparameters of the eSOL dataset. 
- Run the `train_eSOL.py` file to train the eSOL dataset. 
- Run the `train_Tc.py` file to train and test the TR dataset.
- Run the `predict.py` file to predict target.



