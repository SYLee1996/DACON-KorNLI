import os 
import json
import time
import copy 
import click
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import logging, AutoTokenizer, AutoModelForSequenceClassification, GPT2ForSequenceClassification
logging.set_verbosity_error()

from KoNLI_UTILS import NLIDataset, SmoothCrossEntropyLoss, EarlyStopping, calc_accuracy, predict

import warnings
warnings.filterwarnings(action='ignore')

@click.command()
@click.option('--epochs', type=click.STRING, required=True)
@click.option('--batch_size', type=click.STRING, required=True)
@click.option('--max_seq_len', type=click.STRING, required=True)
@click.option('--drop_out', type=click.STRING, required=True)
@click.option('--weight_decay', type=click.STRING, required=True)
@click.option('--n_fold', type=click.STRING, required=True)
@click.option('--patience', type=click.STRING, required=True)
@click.option('--learning_rate', type=click.STRING, required=True)
@click.option('--device', type=click.STRING, required=True)
@click.option('--text', type=click.STRING, required=True)
@click.option('--model_name', type=click.STRING, required=True)
@click.option('--load_dataset', type=click.STRING, required=False)
@click.option('--kornlu_num', type=click.STRING, required=False)
@click.option('--load_dataset', type=click.STRING, required=False)
@click.option('--label_smoothing', type=click.STRING, required=False)

def main(device, epochs, max_seq_len, batch_size, drop_out, weight_decay, n_fold, patience, learning_rate, text, model_name, load_dataset, kornlu_num, label_smoothing):

    seed = 10
    num_labels = 3
    if kornlu_num is not None:
        kornlu_num = int(kornlu_num)
    n_fold = int(n_fold)
    epochs = int(epochs)
    patience = int(patience)
    max_seq_len = int(max_seq_len)
    batch_size = int(batch_size)
    weight_decay = float(weight_decay)
    drop_out = float(drop_out)
    label_smoothing = float(label_smoothing)
    learning_rate = float(learning_rate)
    max_grad_norm = 10
        
    device = device
    text = text
    model_name = model_name
    if model_name == 'klue/roberta-base':
        library_name = 'AutoModelForSequenceClassification'
        
    elif model_name == 'klue/roberta-large':
        library_name = 'AutoModelForSequenceClassification'
        
    elif model_name == 'klue/roberta-small':
        library_name = 'AutoModelForSequenceClassification'

    Model_library = eval(library_name)
    
    model_save_folder = './RESULTS/'
    model_save = '{}_{}_{}_{}_{}_{}_{}_{}_{}_fold'.format(text,
                                                        kornlu_num,
                                                        learning_rate,
                                                        max_seq_len,
                                                        n_fold,
                                                        batch_size,
                                                        weight_decay,
                                                        drop_out,
                                                        patience)
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
    if load_dataset is not None:
        train = pd.read_csv(load_dataset)
        print("Loaded data is using.\n")
        
    else:
        AUG_data = pd.read_csv('Augmented_data.csv')
        json_train_path = 'klue-nli-v1.1_train.json'
        json_test_path = 'klue-nli-v1.1_dev.json'
        multinli_path = 'https://github.com/kakaobrain/KorNLUDatasets/raw/master/KorNLI/multinli.train.ko.tsv'
        snli_path = 'https://github.com/kakaobrain/KorNLUDatasets/raw/master/KorNLI/snli_1.0_train.ko.tsv'
        xnli_dev_path = 'https://github.com/kakaobrain/KorNLUDatasets/raw/master/KorNLI/xnli.dev.ko.tsv'
        xnli_test_path = 'https://github.com/kakaobrain/KorNLUDatasets/raw/master/KorNLI/xnli.test.ko.tsv'
            
        with open(json_train_path, 'r', encoding="utf-8") as f:
            json_train = json.load(f)
        with open(json_test_path, 'r', encoding="utf-8") as f:
            json_test = json.load(f)
            
        json_train_df = pd.DataFrame(json_train)[['premise','hypothesis','gold_label']]
        json_test_df = pd.DataFrame(json_test)[['premise','hypothesis','gold_label']]    
        multinli_data = pd.read_csv(multinli_path, sep='\t', error_bad_lines=False).sample(kornlu_num) # on_bad_lines='skip'
        snli_data = pd.read_csv(snli_path, delimiter='\t', error_bad_lines=False).sample(kornlu_num)
        xnli_dev_data = pd.read_csv(xnli_dev_path, delimiter='\t', error_bad_lines=False)
        xnli_test_data  = pd.read_csv(xnli_test_path, delimiter='\t', error_bad_lines=False)
        
        json_train_df.rename(columns = {'gold_label' : 'label'}, inplace = True)
        json_test_df.rename(columns = {'gold_label' : 'label'}, inplace = True)
        multinli_data.rename(columns = {'gold_label' : 'label', 'sentence1' : 'premise', 'sentence2' : 'hypothesis'}, inplace = True)
        snli_data.rename(columns = {'gold_label' : 'label', 'sentence1' : 'premise', 'sentence2' : 'hypothesis'}, inplace = True)
        xnli_test_data.rename(columns = {'gold_label' : 'label', 'sentence1' : 'premise', 'sentence2' : 'hypothesis'}, inplace = True)
        xnli_dev_data.rename(columns = {'gold_label' : 'label', 'sentence1' : 'premise', 'sentence2' : 'hypothesis'}, inplace = True)    
        
        df1 = pd.concat([json_train_df, json_test_df]).reset_index(drop=True)
        df2 = pd.concat([multinli_data,
                        snli_data,
                        xnli_test_data.loc[:kornlu_num],
                        xnli_dev_data[:kornlu_num]]).reset_index(drop=True)
        
        
        df_grp1 = df1.groupby(df1.columns.tolist()) # 전체 열 비교
        df_di1 = df_grp1.groups # 딕셔너리로 만들기 
        idx_T1 = [x[0] for x in df_di1.values() if len(x) == 1] # 중복X 인덱스 검토
        idx_F1 = [x[0] for x in df_di1.values() if not len(x) == 1] # 중복O 인덱스 검토
        df_concated1 = pd.concat([df1.loc[idx_T1,:], df1.loc[idx_F1,:]])
        df_concated1 = df_concated1.dropna(how='any') # Null 값이 존재하는 행 제거
        train = df_concated1.reset_index(drop=True)

        le = LabelEncoder()
        train['label_num'] = le.fit_transform(train['label'])
    
        if kornlu_num != 0:
            df_grp2 = df2.groupby(df2.columns.tolist()) # 전체 열 비교
            df_di2 = df_grp2.groups # 딕셔너리로 만들기 
            idx_T2 = [x[0] for x in df_di2.values() if len(x) == 1] # 중복X 인덱스 검토
            idx_F2 = [x[0] for x in df_di2.values() if not len(x) == 1] # 중복O 인덱스 검토
            df_concated2 = pd.concat([df2.loc[idx_T2,:], df2.loc[idx_F2,:]])
            df_concated2 = df_concated2.dropna(how='any') # Null 값이 존재하는 행 제거

            df_concated2['premise'] = df_concated2['premise'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9 a-z A-Z]', '', regex=True)
            df_concated2['hypothesis'] = df_concated2['hypothesis'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9 a-z A-Z]', '', regex=True)

            df_concated2 = df_concated2[df_concated2['premise'].apply(lambda x: len(x)>=19)] # 'premise' 글자 수 18 미만인 데이터 제거 
            df_concated2 = df_concated2[df_concated2['premise'].apply(lambda x: len(x)<=90)] # 'premise' 글자 수 89 초과인 데이터 제거
            df_concated2 = df_concated2[df_concated2['hypothesis'].apply(lambda x: len(x)>=5)] # 'hypothesis' 글자 수 5 미만인 데이터 제거
            df_concated2 = df_concated2[df_concated2['hypothesis'].apply(lambda x: len(x)<=103)] # 'hypothesis' 글자 수 103 초과인 데이터 제거
            df_concated2['label_num'] = le.transform(df_concated2['label'])
                
            KAKAO_list = [df_concated2.iloc[i] for i in range(len(df_concated2))] 
                    
        print("KLUE data + {} is using.\n".format(kornlu_num))
    
    # -------------------------------------------------------------------------------------------
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]=device
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    # -------------------------------------------------------------------------------------------
    
    AUG_data = AUG_data.rename(columns={"back_premise": "premise", "back_hypothesis": "hypothesis"})
    AUG_data['label_num'] = le.transform(AUG_data['label'])
    
    AUG_list = [AUG_data.iloc[i] for i in range(len(AUG_data))]
    # -------------------------------------------------------------------------------------------

    # KFold
    fold=0
    folds = []
    k_acc_plot, k_val_acc_plot = [], []    
    
    if model_name == "skt/kogpt2-base-v2":
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                    pad_token='<pad>', mask_token='<mask>', sep_token='</s>')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for train_idx, valid_idx in kf.split(train):
        folds.append((train_idx, valid_idx))
        train_idx, valid_idx = folds[fold]

        Train_set = [train.iloc[i] for i in train_idx]
        Valid_set = [train.iloc[i] for i in valid_idx]
        
        if kornlu_num != 0:
            Train_set = Train_set + AUG_list + KAKAO_list
        else:
            Train_set = Train_set + AUG_list
        
        # Train
        train_set = NLIDataset(data=Train_set, 
                               max_seq_len=max_seq_len,    
                               tokenizer=tokenizer,
                               model_name=model_name)
        Train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True)
        
        # Validation 
        valid_set = NLIDataset(data=Valid_set, 
                               max_seq_len=max_seq_len, 
                               tokenizer=tokenizer,
                               model_name=model_name)
        Valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=16, shuffle=True)
            
        model = Model_library.from_pretrained(model_name, num_labels=num_labels).to(device) 
        model = nn.DataParallel(model).to(device)
        
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_out
            
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
            ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        loss_fn = SmoothCrossEntropyLoss(smoothing=label_smoothing)
        early_stopping = EarlyStopping(patience=patience, mode='max')
        
        best=0.5
        acc_plot, val_acc_plot = [], []
        for e in range(epochs):
            start=time.time()
            train_acc = 0.0
            valid_acc = 0.0
            
            model.train()
            for batch_id, batch in tqdm(enumerate(Train_loader), total=len(Train_loader)):
                optimizer.zero_grad()
                ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
                segment_ids = torch.tensor(batch['token_type_ids'], dtype=torch.long, device=device)
                atts = torch.tensor(batch['attention_mask'], dtype=torch.float, device=device)
                labels = torch.tensor(batch['labels'], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast():    
                    pred = model(input_ids=ids, token_type_ids=segment_ids, attention_mask=atts)[0] 
                    pred = pred.type(torch.FloatTensor).to(device)
                loss = loss_fn(pred, labels)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                train_acc += calc_accuracy(pred, labels)
            train_acc = train_acc/(batch_id+1)
            acc_plot.append(train_acc)
            scheduler.step()
            
            model.eval()
            for batch_id, batch in tqdm(enumerate(Valid_loader), total=len(Valid_loader)):
                with torch.no_grad():
                    ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
                    segment_ids = torch.tensor(batch['token_type_ids'], dtype=torch.long, device=device)
                    atts = torch.tensor(batch['attention_mask'], dtype=torch.float, device=device)
                    labels = torch.tensor(batch['labels'], dtype=torch.long, device=device)
                    pred = model(input_ids=ids, token_type_ids=segment_ids, attention_mask=atts)[0]
                    pred = pred.type(torch.FloatTensor).to(device)
                valid_acc += calc_accuracy(pred, labels)
            valid_acc = valid_acc/(batch_id+1)
            val_acc_plot.append(valid_acc)
            
            print_best = 0    
            if valid_acc>=best:
                difference = valid_acc - best
                best = valid_acc 
                best_idx = e+1
                model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
                best_model_wts = copy.deepcopy(model_state_dict)
                
                # load and save best model weights
                model.module.load_state_dict(best_model_wts)
                torch.save(model_state_dict, model_save_folder+model_save+str(fold)+".pt")
                print_best = '==> best model saved - %d epoch / %.5f    difference %.5f'%(best_idx, best, difference)
                
            TIME = time.time() - start
            print(f'fold : {fold+1}/{n_fold}    epoch : {e+1}/{epochs}    time : {TIME:.0f}/{TIME*(epochs-e-1):.0f} ')
            print(f'TRAIN acc : {train_acc:.5f} ')
            print(f'VALID acc : {valid_acc:.5f}    best : {best:.5f}')
            print('\n') if type(print_best)==int else print(print_best,'\n')
            
            if early_stopping.step(torch.tensor(valid_acc)):
                break
            
        fold+=1    
        k_acc_plot.append(max(acc_plot))
        k_val_acc_plot.append(max(val_acc_plot))
        
    print("Train Loss: ",np.mean(k_acc_plot),", Valid Loss: ",np.mean(k_val_acc_plot))
    print(model_save + ' model is saved!')
    del train; del train_set; del Train_loader; del valid_set; del Valid_loader; 
    torch.cuda.empty_cache()  

if __name__ == '__main__':
    main()
