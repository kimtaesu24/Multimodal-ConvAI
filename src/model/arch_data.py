import numpy as np
import pandas as pd
import torch
import ast
from . import modules
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MyArch_Dataset(Dataset):
    def __init__(self, data_path, device, mode='train', max_length=30, FA=True, LM=True, audio_padding=50, fps=24):
        self.data_path = data_path
        self.device = device
        self.audio_feature_path = self.data_path + 'audio_feature/'+ mode
        self.single_file_path = self.data_path + mode
        self.max_length = max_length
        self.forced_align = FA
        self.landmark_append = LM
        self.audio_padding = audio_padding
        self.fps = fps
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = '!'
        self.tokenizer.bos_token = '#'
        self.tokenizer.padding_side = 'left'
        
        self.label_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.label_tokenizer.pad_token = '!'
        self.tokenizer.bos_token = '#'
        
        if mode == 'train':
            self.FA = pd.read_csv(self.data_path + 'train_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'train_fer_matched.csv')
            self.emotion = pd.read_csv(self.data_path + 'train_emotion_matched.csv')
            # self.emotion = pd.read_csv(self.data_path + 'train_landmark.csv')
        else:
            self.FA = pd.read_csv(self.data_path + 'valid_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'valid_fer_matched.csv')
            self.emotion = pd.read_csv(self.data_path + 'valid_emotion_matched.csv')
            # self.emotion = pd.read_csv(self.data_path + 'valid_landmark.csv')
            
        self.T_padding = max(len(i) for i in self.fer['T_list'].apply(eval))  # 459
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
        if self.FA['Dialogue_ID'][idx] != self.FA['Dialogue_ID'][idx+1]:  # next dialogue appear
            self.manual_index += 1
            idx += 1
            
        while(self.FA['Utterance_ID'][idx] != (self.FA['Utterance_ID'][idx+1] - 1)):  # empty uttrance appear
            self.manual_index += 1
            idx += 1
        
        context = ' '.join(ast.literal_eval(self.FA['word'][idx])).lower() + '.'
        response = ' '.join(ast.literal_eval(self.FA['word'][idx+1])).lower() + '.'
        # print('context: ', context)
        # print('response: ', response)
        
        start = ast.literal_eval(self.FA['start'][idx])
        start = modules.pad(start, self.max_length)
        end = ast.literal_eval(self.FA['end'][idx])
        end = modules.pad(end, self.max_length)
        
        T = ast.literal_eval(self.fer['T_list'][idx])
        if len(T) < self.T_padding:  # padding
            T = modules.pad(T, self.T_padding)
        
        tokens = self.tokenizer(context + self.tokenizer.eos_token,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                                )

        waveform = torch.load(self.audio_feature_path+'/dia{}_utt{}_16000.pt'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx]), map_location=self.device)
        if self.forced_align:
            audio_path = self.single_file_path+'/dia{0}/utt{1}/dia{0}_utt{1}_16000.wav'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx])
            audio_feature = modules.audio_word_align(waveform, audio_path, start, end, self.audio_padding)
        else:
            audio_feature = torch.mean(waveform, dim=1)
        
        if self.landmark_append:
            landmarks = modules.get_landmark(self.single_file_path+'/dia{0}/utt{1}/'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx]), start, self.fps)
        else:
            landmarks = torch.tensor([])
            
        tokens_labels = self.label_tokenizer(response + self.tokenizer.eos_token,
                                            padding='max_length',
                                            max_length=self.max_length,
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors='pt'
                                            )
        
        inputs = [torch.tensor(start).to(self.device),
                  torch.tensor(end).to(self.device), 
                  torch.tensor(T).to(self.device), 
                  tokens.to(self.device),
                  audio_feature,
                  landmarks.to(self.device),
                  ]
        
        labels = [tokens_labels.to(self.device),
                  torch.tensor(self.emotion['Emotion'][idx]).to(self.device)]
        
        return inputs, labels