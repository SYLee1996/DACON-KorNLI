import os 
import json
import time
import copy 
import click
import random 
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
logging.set_verbosity_error()

from KoNLI_UTILS import NLIDataset, predict

import warnings
warnings.filterwarnings(action='ignore')

@click.command()
@click.option('--model_name', type=click.STRING, required=True)
@click.option('--model_save', type=click.STRING, required=True)
@click.option('--device', type=click.STRING, required=True)

def main(model_name, model_save, device):
    
    model_save_folder = './RESULTS/'
    model_save = model_save 
    model_name = model_name
    device = device
    
    text = "_".join(model_save.split("_")[:-9])
    learning_rate = float(model_save.split("_")[-9])

    max_seq_len = int(model_save.split("_")[-7])
    n_fold = int(model_save.split("_")[-6])
    batch_size = int(model_save.split("_")[-5])
    weight_decay = float(model_save.split("_")[-4])
    drop_out = float(model_save.split("_")[-3])
    patience = int(model_save.split("_")[-2])
    num_labels = 3
    seed = 10    

    if model_name == 'klue/roberta-base':
        library_name = 'AutoModelForSequenceClassification'
        
    elif model_name == 'klue/roberta-large':
        library_name = 'AutoModelForSequenceClassification'
        
    elif model_name == 'Huffon/klue-roberta-base-nli':
        library_name = 'AutoModelForSequenceClassification'
        
    elif model_name == 'skt/kogpt2-base-v2':
        library_name = 'GPT2ForSequenceClassification'    
        
    Model_library = eval(library_name)

    # -------------------------------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    # -------------------------------------------------------------------------------------------
    train = pd.read_csv("train_data.csv")
    test = pd.read_csv("test_data.csv")
    sub = pd.read_csv("sample_submission.csv")

    le = LabelEncoder()
    train['label_num'] = le.fit_transform(train['label'])

    label_idx = dict(zip(list(le.classes_), le.transform(list(le.classes_))))
    idx_label = {value: key for key, value in label_idx.items()}
    # -------------------------------------------------------------------------------------------
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]=device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    # -------------------------------------------------------------------------------------------
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Test
    models = []
    for fold in range(n_fold):
        model = Model_library.from_pretrained(model_name, num_labels=num_labels).to(device) 
        model = nn.DataParallel(model).to(device)
        model_dict = torch.load(model_save_folder+model_save+str(fold)+".pt")
        model.module.load_state_dict(model_dict) if torch.cuda.device_count() > 1 else model.load_state_dict(model_dict)
        models.append(model)
            
    # Test 
    test_set = NLIDataset(data=test, 
                          max_seq_len=max_seq_len, 
                          tokenizer=tokenizer,
                          model_name=model_name,
                          mode='test')
    Test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=16, shuffle=False)

    preds = predict(models,Test_loader,device,model_name)
    preds = np.array([idx_label[int(val)] for val in preds])

    sub["label"] = preds
    sub.to_csv(model_save_folder+"{}.csv".format(model_save), index=False)
    print(model_save + " is saved!")
    
if __name__ == '__main__':
    main()
