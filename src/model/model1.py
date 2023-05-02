#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import time
import librosa

from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from .DAN.demo import FER as DAN
from .Forced_Alignment.FA import get_dic as FA

class MyModel1(torch.nn.Module):
    def __init__(
            self,
            param,
            hyper_param,
    ):
        super(MyModel1, self).__init__()
        '''
        self.embedding_size = hyper_param['embedding_size']
        self.dropout = hyper_param['dropout']
        self.num_layers = num_layers
        '''
        self.fps = param['fps']
        #self.fps = 24
        self.give_weight = param['give_weight']
        #self.give_weight = True
        self.modal_fusion = param['modal_fusion']
        #self.modal_fusion = False
        self.device = param['device']
        #self.device = device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = hyper_param['batch_size']
        #self.batch_size = 1
        self.max_length = hyper_param['max_length']
        #self.max_length = 30
        self.alpha = hyper_param['alpha']
        #self.alpha = 2
        if hyper_param['act'] =='relu':
            self.act = nn.ReLU()
        self.gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.embedding_layer = self.gpt_model.get_input_embeddings()        

        self.MMfusion = nn.Linear(1024+768, 1024, bias=True)
        self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)


    def FER(self, frames):
        '''
        Input: frames of single speaker utterance video
        Output: speaker's facial expression list / timestamp of expression transition
        
        Module: DAN
        '''
        expression_list = []
        T = []
        for frame in frames:
            expression_list.append(DAN(frame))
        for t in range(len(expression_list) - 1):
            if expression_list[t] != expression_list[t+1]:
                T.append(t+1)

        return expression_list, T
    
    
    def forced_alignment(self, audio_path, transcript):
        '''
        Input: raw voice data of single speaker
        Output: dictionary of word and timestamp 
                    = {'I': [0.0, 0.05],
                       'am': [0.06, 0.09],
                       ...
                       }
        
        Module: WAV2VEC 
        '''
        word_timestamp = FA(audio_path, transcript)
        return word_timestamp
    
    
    def weighted_word(self, T, start, end, tokens):
        '''
        Input: timestamp of expression transition / raw voice data of single speaker
        
        Module: huggingface tokenizer
        
        Goal: give weight to specific word with attention mask
        '''      
        # give weight to text when transition occur
        if self.give_weight:
            for t in T:
                for i, (audio_start, audio_end) in enumerate(zip(start, end)):
                    if audio_start < (t / self.fps) < audio_end:
                        tokens['attention_mask'][0][i] = 1 * self.alpha
        return tokens
    
    def get_audio_feature(self, SPEECH_FILE):
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        audio_input, _ = librosa.load(SPEECH_FILE, sr=16_000)  # 48000
        
        # Preprocess the audio input
        inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)

        # Pass the input through the model to get the audio features
        with torch.no_grad():
            outputs = model(inputs.input_values)

        # Extract the audio features from the output
        features = outputs.last_hidden_state
        
        return features
    
    
    def multimodal_fusion(self, inputs_embeds, audio_feature):
        '''
        Input:  text embedding / audio feature
        Output: multimodal fused embedding
        '''
        audio_feature = audio_feature.repeat(self.max_length,1)  # [1,768] -> [30,768]
        embedding = self.MMfusion(torch.cat((inputs_embeds, audio_feature), dim=1))
        return self.act(embedding)



    def forward(self, inputs, labels):
        '''
        inputs: start time, end time, T, tokens= single sentence, waveform= audio feature
        labels: responsive sentence
        '''
        start = inputs[0]
        end = inputs[1]
        T = inputs[2]
        tokens = inputs[3]
        waveform = inputs[4]
        
        if self.batch_size == 1:
            start = torch.squeeze(start, dim=0)
            end = torch.squeeze(end, dim=0)
            T = torch.squeeze(T, dim=0)
            tokens['attention_mask'] = torch.squeeze(tokens['attention_mask'], dim=0)
            tokens['input_ids'] = torch.squeeze(tokens['input_ids'], dim=0)
            waveform = torch.squeeze(waveform, dim=0)
        
        # ==== step 1. Give weight to word ====
        tokens = self.weighted_word(T, start, end, tokens)
        
        # ==== step 2. Extract audio feature ====
        audio_feature = torch.mean(waveform, dim=1)
        
        # ==== step 3. Generate next sentence ====
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids'][0]]  # get the embedding layer weights
        
        if self.modal_fusion:
            inputs_embeds = self.multimodal_fusion(inputs_embeds, audio_feature)
            
        if self.batch_size == 1:
            inputs_embeds = torch.unsqueeze(inputs_embeds, 0)
            
        output = self.gpt_model(inputs_embeds=inputs_embeds,
                                attention_mask=tokens['attention_mask'],
                                labels=labels
                                )
        return output.loss

    def inference(self, inputs):
        '''
        inputs: [frames, audio]
        labels: [text]
        '''
        frames = inputs[0]
        audio_path = inputs[1]
        tokens = inputs[2]
        transcript= inputs[3]
        waveform = inputs[4]
        
        
        print("==== step 1. Facial Expression Recognition ====")
        step1 = time.time()
        expression_list, T = self.FER(frames)
        print('step 1 takes time: ', time.time()-step1)
        
        
        print("==== step 2. Give weight to word ====")
        step2 = time.time()
        word_timestamp = self.forced_alignment(audio_path, transcript)
        start = word_timestamp[1]
        end = word_timestamp[2]
        tokens = self.weighted_word(T, start, end, tokens)
        print('step 2 takes time: ', time.time()-step2)
        

        print("==== step 3. Extract audio feature ====")
        step3 = time.time()
        # audio_feature = self.get_audio_feature(audio_path)
        audio_feature = torch.mean(waveform, dim=1)
        print('step 3 takes time: ', time.time()-step3)
        
        
        print("==== step 4. Generate next sentence ====")        
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids'][0]]
        
        if self.modal_fusion:
            inputs_embeds = self.multimodal_fusion(inputs_embeds, audio_feature)
            
        if self.batch_size == 1:
            inputs_embeds = torch.unsqueeze(inputs_embeds, 0)
            
        decoder_input_ids = torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long)*self.gpt_model.config.bos_token_id
        output = self.gpt_model.generate(inputs_embeds=inputs_embeds,
                                        attention_mask=tokens['attention_mask'],
                                        decoder_input_ids=decoder_input_ids
                                        )

        return output
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 30
    
    SPEECH_FILE = '/home2/dataset/MELD/audio_feature/train/dia0_utt3.pt'
    
    #word_timestamp = [" ".join(["OH", "MY", "GOD", "HE'S", "LOST", "IT", "HE'S", "TOTALLY", "LOST", "IT"]),[0.1208125, 0.302125, 0.5035625, 0.8056875, 1.0876875, 1.4905625, 1.59125, 1.73225, 2.115, 2.5178125],[0.282, 0.423, 0.765375, 0.9668125, 1.41, 1.551, 1.692, 2.0545625, 2.417125, 2.558125]]
    #T = [12, 14, 15, 16, 18, 22, 41, 47, 82, 84, 85, 86, 87, 88, 89, 91, 102, 103, 104, 106, 109, 110, 111, 112, 114, 116, 118, 121, 140, 142]
    
    word_timestamp = [" ".join(["SO", "LET'S", "TALK", "A", "LITTLE", "BIT", "ABOUT", "YOUR", "DUTIES"]),[0.4014375, 0.602125, 0.8630625, 1.1039375, 1.1641875, 1.364875, 1.5455625, 1.766375, 1.9269375],[0.562, 0.802875, 1.0638125, 1.124, 1.3448125, 1.5255, 1.74625, 1.88675, 2.187875]]
    T = [54, 55, 58, 59, 60, 61, 62, 63, 64]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    param = dict()
    param['device'] = device
    param['fps'] = 24
    param['give_weight'] = True
    param['modal_fusion'] = True
    
    hyper_param = dict()
    hyper_param['act'] = 'relu'
    hyper_param['batch_size'] = 1
    hyper_param['max_length'] = 30
    hyper_param['alpha'] = 2
    hyper_param['embedding_size'] = 1024
    
    print(word_timestamp[0])
    tokens = tokenizer("so let's talk a little bit about your duties",
                       padding='max_length',
                       max_length=max_length,
                       truncation=True,
                       return_attention_mask=True,
                       return_tensors='pt'
                       )
    print(tokens)
    
    waveform = torch.load(SPEECH_FILE)
    #print(torch.mean(waveform, 1).shape)
    
    labels = "What?"
    labels_token = tokenizer(labels,
                             padding='max_length',
                             max_length=max_length,
                             truncation=True,
                             return_tensors='pt'
                             ).input_ids
    print(labels_token)
    
    model = MyModel1(param, hyper_param).to(device)
    inputs = [word_timestamp[1], word_timestamp[2], T, tokens.to(device), waveform]
    loss = model(inputs, labels_token)
    
    print('loss: ', loss.item())

    