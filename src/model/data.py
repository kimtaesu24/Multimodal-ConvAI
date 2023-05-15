import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MyDataset(Dataset):
    def __init__(self, data_path, device, mode='train',max_length=30):
        self.data_path = data_path
        self.audio_feature_path = self.data_path + 'audio_feature/'+ mode + '/'
        self.device = device
        if mode == 'train':
            self.FA = pd.read_csv(self.data_path + 'train_FA.csv')
            self.fer = pd.read_csv(self.data_path + 'train_fer.csv')
        else:
            self.FA = pd.read_csv(self.data_path + 'valid_FA.csv')
            self.fer = pd.read_csv(self.data_path + 'valid_fer.csv')
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return max(self.FA['Dialogue_ID'])

    def __getitem__(self, idx):
        try:
            # print(idx)
            FA = self.FA[self.FA['Dialogue_ID']==idx]
            word = FA['word'].apply(eval).values.tolist()  # string-> list / get values / to 2d list
            for i in range(len(word)):
                word[i] = " ".join(word[i]).lower()  # list element to string
                
            # label = word.pop()  # get label
            # print('word: ', word)
            
            start = FA['start'].apply(eval).values.tolist()[:-1]  # string-> list / get values / to 2d list / slice label
            # print('start: ', start)
            
            end = FA['end'].apply(eval).values.tolist()[:-1]  # string-> list / get values / to 2d list / slice label
            # print('end: ', end)
            
            # utt_list = FA['Utterance_ID'].values.tolist()[:-1]
            # print(utt_list)
            
            fer = self.fer[self.fer['Dialogue_ID']==idx]
            T = fer['T_list'].apply(eval).values.tolist()[:-1]  # string-> list / get values / to 2d list / slice label
            # print('T: ', T)
            
            tokens = self.tokenizer(word[0] + self.tokenizer.eos_token,
                                    # padding='max_length',
                                    # max_length=self.max_length,
                                    # truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt'
                                    )
            
            #for utt_n in utt_list:
            waveform = torch.load(self.audio_feature_path+'dia{}_utt0.pt'.format(idx), map_location=self.device)
            
            label_token = self.tokenizer(word[1] + self.tokenizer.eos_token,
                                # padding='max_length',
                                # max_length=self.max_length,
                                # truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                                )
            
            inputs = [torch.tensor(start[0]).to(self.device),
                    torch.tensor(end[0]).to(self.device), 
                    torch.tensor(T[0]).to(self.device), 
                    tokens.to(self.device), 
                    waveform]
            labels = label_token.to(self.device)
        except Exception as e:
            #print(f"An error occurred while processing the data at index {idx}: {e}")
            return self.__getitem__(idx + 1) 
        
        return inputs, labels