import numpy as np
import pandas as pd
import torch
import ast
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def pad(inputs, max_length):
    tmp = [0 for i in range(max_length)]
    if len(inputs) > max_length:
        tmp[:len(inputs)] = inputs[:max_length]  # truncation
    else:
        tmp[:len(inputs)] = inputs  # padding
    return tmp

class Arch1_Dataset(Dataset):
    def __init__(self, data_path, device, mode='train',max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = '!'
        self.tokenizer.padding_side = 'left'
        
        self.label_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.label_tokenizer.pad_token = '!'
        
        self.data_path = data_path
        self.audio_feature_path = self.data_path + 'audio_feature/'+ mode
        self.device = device
        
        if mode == 'train':
            self.FA = pd.read_csv(self.data_path + 'train_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'train_fer_matched.csv')
        else:
            self.FA = pd.read_csv(self.data_path + 'valid_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'valid_fer_matched.csv')
            
        # self.timestamp_padding = max(len(i) for i in self.FA['start'].apply(eval))  # 69
        self.T_padding = max(len(i) for i in self.fer['T_list'].apply(eval))  # 459
        self.max_length = max_length
        self.manual_index = 0

    def __len__(self):
        length = 0
        for idx in range(len(self.FA) - 1):
            if (self.FA['Utterance_ID'][idx] == (self.FA['Utterance_ID'][idx+1] -1)):
                length += 1
        return length

    def __getitem__(self, idx):
        if idx == 0:
            self.manual_index = 0  # initialize
            
        idx += self.manual_index
        if self.FA['Dialogue_ID'][idx] != self.FA['Dialogue_ID'][idx+1]:
            self.manual_index += 1
            idx += 1
            # print('----------------------------------------')
        while(self.FA['Utterance_ID'][idx] != (self.FA['Utterance_ID'][idx+1] - 1)):
            self.manual_index += 1
            idx += 1
        # print(idx)
        
        context = ' '.join(ast.literal_eval(self.FA['word'][idx])).lower() + '.'
        response = ' '.join(ast.literal_eval(self.FA['word'][idx+1])).lower() + '.'
        # print('context: ', context)
        # print('response: ', response)
        
        start = ast.literal_eval(self.FA['start'][idx])
        start = pad(start, self.max_length)
        end = ast.literal_eval(self.FA['end'][idx])
        end = pad(end, self.max_length)
        
        T = ast.literal_eval(self.fer['T_list'][idx])
        if len(T) < self.T_padding:  # padding
            T = pad(T, self.T_padding)
        
        tokens = self.tokenizer(context + self.tokenizer.eos_token,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                                )
        labels = self.label_tokenizer(response + self.tokenizer.eos_token,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                                )
        
        waveform = torch.load(self.audio_feature_path+'/dia{}_utt{}_16000.pt'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx]), map_location=self.device)
        audio_feature = torch.mean(waveform, dim=1)
        
        inputs = [torch.tensor(start).to(self.device),
                  torch.tensor(end).to(self.device), 
                  torch.tensor(T).to(self.device), 
                  tokens.to(self.device),
                  audio_feature,
                  ]
        
        labels = labels.to(self.device)
        
        return inputs, labels