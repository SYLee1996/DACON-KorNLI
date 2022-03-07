# DACON-KorNLI

## Private 5th | 0.89615 | Roberta-large+Backtrans(SBERT, cosine similarity)
+ 주최 및 주관: 데이콘 
+ 링크: https://dacon.io/competitions/official/235875/overview/description
----
## Edit log
+ googletanslator를 사용할 때 필요한 chromedriver 업로드(22.03.07)
+ Backtranslation 과정의 set_setting의 'path'값을 './chromedriver' 수정(22.03.07)
+ Docker image 수정 - 일부 오래된 버전의 라이브러리가 존재하여 conda 설치 후 라이브러리 업데이트. tag 끝에 -conda 붙어있는게 수정된 환경입니다.(22.03.07)

----
## Process
+ 데이터는 KLUE, KorNLU를 사용했으며 KorNLU 데이터의 경우 전체 데이터 중 일부(50000개)를 학습에 사용했습니다.
+ KLUE 데이터에 대해서만 (한->영->한) BackTranslation augmentation을 진행했습니다.
+ BackTranslation의 경우 번역 데이터의 품질이 좋지못해 원본 데이터와 증강 데이터의 '코사인 유사도'를 구해 유사도가 0.9 이상인 데이터만 선택적으로 학습에 사용했습니다.
  + 원본 데이터와 증강 데이터 사이의 유사도를 구하기위해 SentenceBERT(SBERT)를 이용했으며, pre-trained 'sentence-roberta-base' 모델을 사용했습니다.    
   
</br>


+ 일차적으로 papago를 사용하여 번역을 진행한 후, 번역이 안되는 경우 부가적으로 google translator를 이용하여 번역을 진행했습니다.
+ 총 4step을 거쳐 번역을 시도했습니다. 
   + 1-step: 번역이 진행됨에 따라 공백 또는 번역이 이루어지지 않는 데이터를 nan값으로 처리 후 재번역     
     ###### ex) ' ' or ''   ->   'gocheok sky dome is dome of korea' 재번역
   + 2-step: 번역이 진행됨에 따라 일부분만 번역된 경우 또한 재번역  
     ###### ex) 'gocheok sky dome은 korea의 경기장이다.'   ->   'gocheok sky dome is dome of korea' 재번역
   + 3-step: 재번역에도 일부분 번역이 안되는 단어의 경우 문장에서 분리 후 번역기로 해당 단어만 번역   
     ###### ex) 'gocheok sky dome은 korea의 경기장이다.'   ->   'gocheok sky dome', 'korea' 단어만 번역
   + 4-step: '한글' -> '영어'로 번역 시, 번역된 문장이 기존 문장의 길이에 대한 비율 0.5 이하이면 재번역
</br>

+ klue_roberta_large를 사용했으며, 5-fold를 이용하여 각 80%의 train data로 학습시킨 모델로 test셋에 대해 soft voting ensemble을 진행했습니다.
    + 학습 데이터: KLUE(80%) + KorNLU + Back Translation
    + 검증 데이터: KLUE(20%)

Optimizer는 AdamW를 사용했고, automatic mixed precision, LabelSmoothing 적용 및 각 fold별 EarlyStopping을 적용했습니다.

---- 
## Environment 
+ Ubuntu 18.04
+ 사용한 Docker image는 Docker Hub에 첨부하며, 두 버전의 환경을 제공합니다.
  + https://hub.docker.com/r/lsy2026/kor-nli/tags
  + (cuda10.2, cudnn7, ubuntu18.04), (cuda11.2.0, cudnn8, ubuntu18.04)
  
  
## Libraries
+ python==3.6.9
+ pandas==1.1.5
+ numpy==1.19.5
+ json==2.0.9
+ tqdm==4.63.0
+ sklearn==0.24.2
+ googletrans==3.0.0
+ selenium==3.141.0
+ torch==1.10.2+cu102
+ transformers==4.16.2

---- 

## Usage
+ ipynb 파일을 이용하는 경우, 'Private-5th_0.89615_Lee.ipynb' 파일에 augmentation 및 train셀이 포함되어 실행시키면 됩니다.
+ py 파일을 이용하여 터미널에서 사용하는 경우, 우선적으로 'BackTranslation+SBERT.ipynb' 파일에서 augmentation 후 학습을 진행해야 합니다.
+ Backtrans augmentation에서 시간이 오래걸리기 때문에 업로드 된 'Augmented_data.csv' 파일을 사용하여 augmentation을 건너 뛸 수 있습니다.


### Terminal Command Example for train
```
!python3 KoNLI_MAIN.py \
--epochs 100 \
--max_seq_len 103 \
--n_fold 5 \
--batch_size 128 \
--weight_decay 0.0001 \
--drop_out 0.1 \
--patience 5 \
--learning_rate 5e-5 \
--label_smoothing 0.2 \
\
\
--device '0,1,2,3' \
--model_name 'klue/roberta-large' \
--text '_FINAL_DATA' \
--kornlu_num 50000
```

Result: 
  
         
        KLUE data + 50000 is using.

        Device: cuda
        GPU activate --> Count of using GPUs: 4
        100%|█████████████████████████████████████████| 914/914 [10:23<00:00,  1.47it/s]
        100%|███████████████████████████████████████████| 44/44 [00:17<00:00,  2.50it/s]
        fold : 1/5    epoch : 1/100    time : 645/63830 
        TRAIN acc : 0.78067 
        VALID acc : 0.92430    best : 0.92430
        ==> best model saved - 1 epoch / 0.92430    difference 0.42430 

        100%|█████████████████████████████████████████| 914/914 [10:12<00:00,  1.49it/s]
        100%|███████████████████████████████████████████| 44/44 [00:17<00:00,  2.45it/s]
        fold : 1/5    epoch : 2/100    time : 635/62230 
        TRAIN acc : 0.87331 
        VALID acc : 0.92637    best : 0.92637
        ==> best model saved - 2 epoch / 0.92637    difference 0.00207 

        100%|█████████████████████████████████████████| 914/914 [10:12<00:00,  1.49it/s]
        100%|███████████████████████████████████████████| 44/44 [00:17<00:00,  2.45it/s]
        fold : 1/5    epoch : 3/100    time : 633/61400 
        TRAIN acc : 0.91737 
        VALID acc : 0.91797    best : 0.92637

        ...


        Train Loss:  0.9854809775555955 , Valid Loss:  0.9371210875199362
        _FINAL_DATA_50000_5e-05_103_5_128_0.0001_0.1_5_fold model is saved!

### Terminal Command Example for inference
```
!python3 KoNLI_INFERENCE.py \
--model_save '_FINAL_DATA_50000_5e-05_103_5_128_0.0001_0.1_5_fold' \
--model_name 'klue/roberta-large' \
--device '0,1,2,3'
```


Result: 

      Device: cuda
      GPU activate --> Count of using GPUs: 4
      100%|███████████████████████████████████████████| 14/14 [00:15<00:00,  1.14s/it]
      _FINAL_DATA_50000_5e-05_103_5_128_0.0001_0.1_5_fold is saved!
