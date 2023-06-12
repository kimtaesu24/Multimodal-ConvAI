import numpy as np
import pandas as pd
import torch
import ast
import json

from . import modules
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MyArch_Dataset(Dataset):
    def __init__(self, data_path, device, hyper_param, param, mode='train'):
        self.data_path = data_path
        self.device = device
        self.audio_feature_path = self.data_path + 'audio_feature/'+ mode
        self.single_file_path = self.data_path + mode
        self.max_length = hyper_param["max_length"]
        self.history_length = hyper_param["history_length"]
        self.audio_padding = hyper_param["audio_pad_size"]
        self.forced_align = param["forced_align"]
        self.landmark_append = param["landmark_append"]
        self.fps = param["fps"]
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')
        # self.tokenizer.pad_token = '!'
        # self.tokenizer.bos_token = '#'
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        
        self.label_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')
        # self.label_tokenizer.pad_token = '!'
        # self.label_tokenizer.bos_token = '#'
        
        if mode == 'train':
            self.FA = pd.read_csv(self.data_path + 'new_train_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'new_train_FER_matched.csv')
            self.emotion = pd.read_csv(self.data_path + 'new_train_emotion_matched.csv')
            self.landmark = pd.read_csv(self.data_path + 'new_train_LM_matched.csv')
        else:
            self.FA = pd.read_csv(self.data_path + 'new_valid_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'new_valid_FER_matched.csv')
            self.emotion = pd.read_csv(self.data_path + 'new_valid_emotion_matched.csv')
            self.landmark = pd.read_csv(self.data_path + 'new_valid_LM_matched.csv')
            
        self.history_path = self.data_path+ mode
        self.T_padding = max(len(i) for i in self.fer['T_list'].apply(eval))  # 459
        self.manual_index = 0

    def __len__(self):
        length = 0
        for idx in range(len(self.FA) - 1):
            if (self.FA['Dialogue_ID'][idx] == self.FA['Dialogue_ID'][idx+1]):  # same dialogue
                if (self.FA['Utterance_ID'][idx] == (self.FA['Utterance_ID'][idx+1] -1)):  # next utterance
                    length += 1
        return length

    def __getitem__(self, idx):
        if idx == 0:
            self.manual_index = 0  # initialize
            
        idx += self.manual_index
        while((self.FA['Dialogue_ID'][idx] != self.FA['Dialogue_ID'][idx+1]) or (self.FA['Utterance_ID'][idx] != (self.FA['Utterance_ID'][idx+1] - 1))):  # next dialogue appear OR empty uttrance appear
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
        
        waveform = torch.load(self.audio_feature_path+'/dia{}_utt{}_16000.pt'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx]))
        if self.forced_align:
            audio_path = self.single_file_path+'/dia{0}/utt{1}/dia{0}_utt{1}_16000.wav'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx])
            audio_feature, waveform_start = modules.audio_word_align(waveform, audio_path, start, end, self.audio_padding)
        else:
            audio_feature = torch.mean(waveform, dim=1)
            waveform_start = None
        
        if self.landmark_append:
            # landmarks = modules.get_landmark(self.single_file_path+'/dia{0}/utt{1}/'.format(self.FA['Dialogue_ID'][idx], self.FA['Utterance_ID'][idx]), start, self.fps)
            landmark_set = torch.tensor(ast.literal_eval(self.landmark['landmark_list'][idx]))
            landmarks = modules.get_aligned_landmark(landmark_set, waveform_start)
        else:
            landmarks = torch.tensor([])
            
        with open(f"{self.history_path}/dia{self.FA['Dialogue_ID'][idx]}/utt{self.FA['Utterance_ID'][idx]}/dia{self.FA['Dialogue_ID'][idx]}_utt{self.FA['Utterance_ID'][idx]}_history.json", "r") as json_file:
            historys = json.load(json_file)
            
        input_historys = ""
        for utt_hist in historys:
            input_historys += utt_hist+self.tokenizer.eos_token
            
        input_historys_tokens = self.tokenizer(input_historys,
                                                padding='max_length',
                                                max_length=self.history_length,
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_tensors='pt'
                                                )
        tokens_labels = self.label_tokenizer(response + self.label_tokenizer.eos_token,
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
                  audio_feature.to(self.device),
                  landmarks.to(self.device),
                  input_historys_tokens.to(self.device),
                  ]
        
        labels = [tokens_labels.to(self.device),
                  torch.tensor(self.emotion['Emotion'][idx]).to(self.device)]
        
        return inputs, labels