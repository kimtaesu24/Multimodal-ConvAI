import numpy as np
import pandas as pd
import torch
import ast
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def pad(inputs, max_length):
    tmp = [0 for i in range(max_length)]
    tmp[:len(inputs)] = inputs
    return tmp

class ARCH1_Dataset(Dataset):
    def __init__(self, data_path, device, mode='train',max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = 'left'
        self.data_path = data_path
        self.audio_feature_path = self.data_path + 'audio_feature/'+ mode
        self.device = device
        
        if mode == 'train':
            self.FA = pd.read_csv(self.data_path + 'train_FA.csv')
            self.fer = pd.read_csv(self.data_path + 'train_fer.csv')
        else:
            self.FA = pd.read_csv(self.data_path + 'valid_FA.csv')
            self.fer = pd.read_csv(self.data_path + 'valid_fer.csv')
            
        self.timestamp_padding = max(len(i) for i in self.FA['start'].apply(eval))  # 69
        self.T_padding = max(len(i) for i in self.fer['T_list'].apply(eval))  # 459
        self.max_length = max_length
        self.manual_index = 0

    def __len__(self):
        return len(self.FA['Dialogue_ID']) - max(self.FA['Dialogue_ID'])

    def __getitem__(self, idx):
        if idx == 0:
            self.manual_index = 0  # initialize
            
        idx += self.manual_index
        if self.FA['Dialogue_ID'][idx] != self.FA['Dialogue_ID'][idx+1]:
            self.manual_index += 1
            idx += 1
            # print('----------------------------------------')
        # print(idx)
        
        context = ' '.join(ast.literal_eval(self.FA['word'][idx])).lower()
        response = ' '.join(ast.literal_eval(self.FA['word'][idx+1])).lower()
        # print('context: ', context)
        # print('response: ', response)
        
        start = ast.literal_eval(self.FA['start'][idx])
        end = ast.literal_eval(self.FA['end'][idx])
        if len(start) < self.timestamp_padding:  # padding
            start = pad(start, self.timestamp_padding)
            end = pad(end, self.timestamp_padding)
        
        T = ast.literal_eval(self.fer['T_list'][idx])
        if len(T) < self.T_padding:  # padding
            T = pad(T, self.T_padding)
        
        tokens = self.tokenizer(context + self.tokenizer.eos_token,
                                # padding='max_length',
                                # max_length=self.max_length,
                                # truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                                )
        labels = self.tokenizer(response + self.tokenizer.eos_token,
                                # padding='max_length',
                                # max_length=self.max_length,
                                # truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                                )
        
        waveform = torch.load(self.audio_feature_path+'/dia{}_utt{}_16000.pt'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx]), map_location=self.device)
        
        inputs = [torch.tensor(start).to(self.device),
                  torch.tensor(end).to(self.device), 
                  torch.tensor(T).to(self.device), 
                  tokens.to(self.device),
                  waveform,
                #   torch.tensor(self.FA['Dialogue_ID'][idx]).to(self.device),
                #   torch.tensor(self.FA['Utterance_ID'][idx]).to(self.device),
                  ]
        
        labels = labels.to(self.device)
        
        return inputs, labels