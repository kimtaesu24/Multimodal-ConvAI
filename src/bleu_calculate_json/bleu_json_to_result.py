import os
import json
import pandas as pd 
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings("ignore")

json_folder_path = './json_data/'

mean_df = pd.DataFrame(columns=['filename', 'mean_Bleu_1', 'mean_Bleu_2', 'mean_Bleu_3', 'mean_Bleu_4'])
mean_idx = 0 

os.makedirs(json_folder_path+'/bleu_result/', exist_ok=True)
for json_file in os.listdir(json_folder_path):
    if json_file[-4:] == 'json':
        with open(json_folder_path+json_file, "r") as json_data_file:
            data = json.load(json_data_file)

        idx = 0
        df = pd.DataFrame(columns=['ref', 'ans', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'])
        
        for value in data:
            ref = [value['ref'].replace('!','').replace('.','').split()]
            ans = value['a'].replace('!','').replace('.','').split()

            bleu_1 = format(sentence_bleu(ref, ans, weights=(1, 0, 0, 0)), '.8f')
            bleu_2 = format(sentence_bleu(ref, ans, weights=(0.5, 0.5, 0, 0)), '.8f')
            bleu_3 = format(sentence_bleu(ref, ans, weights=(1/3, 1/3, 1/3, 0)), '.8f')
            bleu_4 = format(sentence_bleu(ref, ans, weights=(0.25, 0.25, 0.25, 0.25)), '.8f')

            df.loc[idx] = [value['ref'].replace('!','').replace('.',''), 
                        value['a'].replace('!','').replace('.','')
                        , float(bleu_1), float(bleu_2), float(bleu_3), float(bleu_4)]
            idx +=1
        
        
        df.to_csv(json_folder_path+'/bleu_result/'+json_file+'.csv')
        mean_df.loc[mean_idx] = [json_file,
                        df['bleu-1'].mean(),
                        df['bleu-2'].mean(),
                        df['bleu-3'].mean(),
                        df['bleu-4'].mean()]
        mean_idx += 1
mean_df.to_csv(json_folder_path+'bleu_result/bleu_mean.csv')