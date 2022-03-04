import os 
import re 
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.modules.loss import _WeightedLoss

from googletrans import Translator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def chrome_setting(path):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(path, options=chrome_options)
    return driver

def google_translation(dataset, raw_text_list, sk, tk):
    raw_trans_list = []
    for idx in tqdm(raw_text_list):
        translator = Translator()
        translator.raise_Exception = True
        trans = translator.translate(dataset.iloc[idx], src=sk, dest=tk)
        time.sleep(1.5)
        raw_trans_list.append(trans.text)
    
    return raw_trans_list

def papago_translation(text_data, sk, tk, driver, save_name=None, index_list=None, mode='save'):
    target_present = EC.presence_of_element_located((By.XPATH, '//*[@id="txtTarget"]'))
    trans_list = []
    
    if index_list is not None:
        final_index = index_list
    else:
        final_index = range(len(text_data))
    
    for idx in tqdm(final_index): 
        try:
            driver.get('https://papago.naver.com/?sk='+sk+'&tk='+tk+'&st='+text_data.iloc[idx])
            time.sleep(1.5)
            element=WebDriverWait(driver, 10).until(target_present)
            backtrans = element.text 

            if (backtrans=='')|(backtrans==' '):
                element=WebDriverWait(driver, 10).until(target_present)
                backtrans = element.text 
                time.sleep(0.1)
                trans_list.append(backtrans)
            else:
                trans_list.append(backtrans)
        except:
            trans_list.append('')
    
        if mode == 'save':
            if idx%100==0:
                np.save(save_name+'_{}_{}.npy'.format(0,(final_index[-1] + 1)),trans_list)
    
    driver.close()
    driver.quit()  
    os.system('killall chrome') 

    if mode == 'save':    
        np.save(save_name+'_{}_{}.npy'.format(0,(final_index[-1] + 1)),trans_list)
        print(save_name+'_{}_to_{} is translated!'.format(sk, tk))
    else:    
        return trans_list
    
    
def nan_list_retranslation(raw_array_df, train, col_name, sk, tk, path):
    
    print('nan_list re-translation.')
    # 번역이 안 된 문장이 존재하는 경우 재번역  
    raw_array_df[col_name].replace('', np.nan, inplace=True)
    raw_array_df[col_name].replace(' ', np.nan, inplace=True)
    nan_list = raw_array_df[raw_array_df[col_name].isnull()].index

    if len(nan_list) != 0:
        count = 0
        
        while len(nan_list) != 0:
            driver = chrome_setting(path)
            if count < 2:
                re_trans_list = papago_translation(train[col_name], sk, tk, driver, index_list=nan_list, mode='retry') # 원본 데이터 재번역
                
            elif count >= 2:
                re_trans_list = google_translation(train[col_name], nan_list, sk, tk) # google-translator로 재번역
            
            for idx, value in zip(nan_list, re_trans_list): 
                raw_array_df[col_name].iloc[idx] = value   
            
            driver.quit()
            raw_array_df[col_name].replace('', np.nan, inplace=True)
            raw_array_df[col_name].replace(' ', np.nan, inplace=True)

            nan_list = raw_array_df[raw_array_df[col_name].isnull()].index
            count+=1
            if count == 4:
                break
        os.system('killall chrome')
        
    return raw_array_df, nan_list


def hangul_list_retranslation(raw_array_df, train, col_name, sk, tk, path):
    
    print('hangul_list re-translation.')
    # sk -> tk 번역된 문장에 sk가 존재하는 경우 재번역 
    hangul_ind=[]
    for i in range(0,len(raw_array_df)):
        temp=re.findall('[a-zA-Z]',str(raw_array_df[col_name][i]))
        if len(temp)!=0:
            hangul_ind.append(i)

    if len(hangul_ind) != 0:
        count = 0
        
        while len(hangul_ind) != 0:
            driver = chrome_setting(path)
            
            if count < 2:
                re_trans_list = papago_translation(train[col_name], sk, tk, driver, index_list=hangul_ind, mode='retry') # 원본 데이터 재번역
                            
            elif count >= 2:
                re_trans_list = google_translation(train[col_name], hangul_ind, sk, tk) # google-translator로 재번역

            for idx, value in zip(hangul_ind, re_trans_list): 
                raw_array_df[col_name].iloc[idx] = value  
            
            driver.quit()
            hangul_ind=[]
            for i in range(0,len(raw_array_df)):
                temp=re.findall('[a-zA-Z]',str(raw_array_df[col_name][i]))
                if len(temp)!=0:
                    hangul_ind.append(i)
            count+=1
            if count == 4:
                break
            
        os.system('killall chrome')
        
    return raw_array_df, hangul_ind


def hangul_word_translation(raw_array_df, col_name, sk, tk, path):
    print('hangul_word re-translation.')
    hangul_ind=[]
    for i in range(0,len(raw_array_df)):
        temp=re.findall('[a-zA-Z]',str(raw_array_df[col_name][i]))
        if len(temp)!=0:
            hangul_ind.append(i)
                    
    if len(hangul_ind) != 0:
        count = 0
        while len(hangul_ind) != 0:
            
            if count < 1:
                for idx in tqdm(hangul_ind):
                    dictt = {}
                    words_raw = re.sub('[^A-Z a-z]', ' ', raw_array_df[col_name].iloc[idx])
                    words = words_raw.split("  ")
                    words = [x.strip() for x in words if x.strip()]
                    
                    transResult_list = []
                    for text in words: 
                        driver = chrome_setting(path)
                        driver.get('https://papago.naver.com/?sk=' + sk + '&tk='+tk+'&hn=0&st=')
                        
                        time.sleep(1)
                        driver.find_element_by_xpath('//*[@id="sourceEditArea"]/label').send_keys(text)
                        driver.find_element_by_xpath('//*[@id="btnTranslate"]').click()
                        time.sleep(1.5)
                        transResult = driver.find_element_by_xpath('//*[@id="txtTarget"]').text
                        time.sleep(1)
                        transResult_list.append(transResult)
                        driver.quit()
                        os.system('killall chrome')
                        
                    dictt['word'] = words
                    dictt['translated_word'] = transResult_list

                    for i in range(len(dictt['word'])):
                        raw_array_df[col_name].iloc[idx] = raw_array_df[col_name].iloc[idx].replace(dictt['word'][i],dictt['translated_word'][i])
                
            elif count >= 1:
                for idx in tqdm(hangul_ind):
                    dictt = {}
                    words_raw = re.sub('[^A-Z a-z]', ' ', raw_array_df[col_name].iloc[idx])
                    words = words_raw.split("  ")
                    words = [x.strip() for x in words if x.strip()]
                    
                    transResult_list = []
                    for text in words: 
                        translator = Translator()
                        translator.raise_Exception = True
                        trans = translator.translate(text, src=sk, dest=tk)
                        time.sleep(1.5)
                        transResult_list.append(trans.text)
                        os.system('killall chrome')
                    dictt['word'] = words
                    dictt['translated_word'] = transResult_list 
                    
                    for i in range(len(dictt['word'])):
                        raw_array_df[col_name].iloc[idx] = raw_array_df[col_name].iloc[idx].replace(dictt['word'][i],dictt['translated_word'][i])
                            
            hangul_ind=[]
            for i in range(0,len(raw_array_df)):
                temp=re.findall('[a-zA-Z]',str(raw_array_df[col_name][i]))
                if len(temp)!=0:
                    hangul_ind.append(i)
                    
            count+=1
            if count >= 2:
                break
            
        os.system('killall chrome')
    return raw_array_df


def len_retranslation(raw_array_df, train, col_name, sk, tk, path):
# Attempt to re-translate a translated sentence if the translated sentence has a ratio of less than 0.5 to the length of an existing sentence
    print('len rate re-translation.')    
    retrans_ind=[]
    for i in range(0,len(raw_array_df)):
        if len(raw_array_df[col_name][i])/len(df_concated1[col_name][i])<=0.5:
            retrans_ind.append(i)
            
    retrans_ind=list(set(retrans_ind))
    count = 0
    
    while len(retrans_ind) != 0:
        driver = chrome_setting(path)
        raw_trans_list = google_translation(train[col_name], retrans_ind, sk, tk)
        
        for idx, value in zip(retrans_ind, raw_trans_list): 
            raw_array_df[col_name].iloc[idx] = value  
            
        retrans_ind=[]
        for i in range(0,len(raw_array_df)):
            if len(raw_array_df[col_name][i])/len(train[col_name][i])<=0.5:
                retrans_ind.append(i)
        
        retrans_ind=list(set(retrans_ind))   
        driver.quit()
        
        count+=1
        if count >= 2:
            break
        
        os.system('killall chrome')
    return raw_array_df


def pytorch_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class NLIDataset(Dataset):
    def __init__(self, data, tokenizer, model_name, max_seq_len=128, mode='train'):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mode = mode
        self.model_name = model_name
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        if self.mode == 'train':
            record = self.data[index]
            encoding_result = self.tokenizer.encode_plus(record['premise'], record['hypothesis'], 
                                                    max_length=self.max_seq_len, 
                                                    pad_to_max_length=True,
                                                    truncation=True)
            if self.model_name == "skt/kogpt2-base-v2":
                return {'input_ids': np.array(encoding_result['input_ids'], dtype=int),
                        'attention_mask': np.array(encoding_result['attention_mask'], dtype=int),
                        'token_type_ids': 0,
                        'labels': np.array(record['label_num'], dtype=int)}
            else:
                return {'input_ids': np.array(encoding_result['input_ids'], dtype=int),
                        'attention_mask': np.array(encoding_result['attention_mask'], dtype=int),
                        'token_type_ids': np.array(encoding_result['token_type_ids'], dtype=int),
                        'labels': np.array(record['label_num'], dtype=int)}
        else:
            record = self.data.iloc[index]
            encoding_result = self.tokenizer.encode_plus(record['premise'], record['hypothesis'], 
                                                    max_length=self.max_seq_len, 
                                                    pad_to_max_length=True,
                                                    truncation=True)
            if self.model_name == "skt/kogpt2-base-v2":
                return {'input_ids': np.array(encoding_result['input_ids'], dtype=int),
                        'token_type_ids': 0,
                        'attention_mask': np.array(encoding_result['attention_mask'], dtype=int)}
            else:
                return {'input_ids': np.array(encoding_result['input_ids'], dtype=int),
                        'attention_mask': np.array(encoding_result['attention_mask'], dtype=int),
                        'token_type_ids': np.array(encoding_result['token_type_ids'], dtype=int)}


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


def predict(models,dataset,device,model_name):
    results = []
    tqdm_dataset = tqdm(enumerate(dataset), total=len(dataset))
    for batch, batch_item in tqdm_dataset:
        ids = torch.tensor(batch_item['input_ids'], dtype=torch.long, device=device)
        segment_ids = torch.tensor(batch_item['token_type_ids'], dtype=torch.long, device=device)
        atts = torch.tensor(batch_item['attention_mask'], dtype=torch.float, device=device)
        
        for fold,model in enumerate(models):
            model.eval()
            with torch.no_grad():
                if fold == 0:
                    if model_name == 'Huffon/klue-roberta-base-nli':
                        pred = model(input_ids=ids)[0]
                    elif model_name == "skt/kogpt2-base-v2":
                        pred = model(input_ids=ids, attention_mask=atts)[0]
                    else:  
                        pred = model(input_ids=ids, token_type_ids=segment_ids, attention_mask=atts)[0] 
                else:
                    if model_name == 'Huffon/klue-roberta-base-nli':
                        pred = pred + model(input_ids=ids)[0]
                    elif model_name == "skt/kogpt2-base-v2":
                        pred = model(input_ids=ids, attention_mask=atts)[0]
                    else:  
                        pred = pred + model(input_ids=ids, token_type_ids=segment_ids, attention_mask=atts)[0] 
        pred = 0.2*pred
        pred = torch.tensor(torch.argmax(pred, axis=-1), dtype=torch.int32).cpu().numpy()
        results.extend(pred)
    return results