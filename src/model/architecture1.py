#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import time
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer

class MyArch1(torch.nn.Module):
    def __init__(
            self,
            param,
            hyper_param,
    ):
        super(MyArch1, self).__init__()
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
        self.dropout = hyper_param['dropout']
        if hyper_param['act'] =='relu':
            self.act = nn.ReLU()
        self.gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding_layer = self.gpt_model.get_input_embeddings()

        self.projection_layer = nn.Linear(768, 768, bias=False)
        self.projection_layer.weight = torch.nn.init.xavier_uniform_(self.projection_layer.weight)
                                                             
        self.MMfusion = nn.Linear(1280+768, 1280, bias=True)
        self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
        # self.feat_drop = nn.Dropout(self.dropout) if self.dropout > 0 else None
        
        self.loss_function = nn.CrossEntropyLoss(ignore_index=50256)


    def FER(self, frames):
        '''
        Input: frames of single speaker utterance video
        Output: speaker's facial expression list / timestamp of expression transition
        
        Module: DAN
        '''
        from .DAN.demo import FER as DAN
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
        Output: 2d array = [['word1', word2', ...], [start_t1, start_t2,...], [end_t1, end_t2, ...]]
        
        Module: WAV2VEC 
        '''
        from .Forced_Alignment.FA import get_dic as FA
        word_timestamp = FA(audio_path, transcript)
        
        return word_timestamp
    
    
    def weighted_word(self, T, start, end, tokens):
        '''
        Input: timestamp of expression transition / raw voice data of single speaker
        Output: 
        
        Goal: give weight to specific word's attention mask
        '''
        # give weight to text when transition occur
        if self.give_weight:
            for mini_batch in range(start.shape[0]):
                for t in T[mini_batch]:
                    if t == 0:
                        break  # padding appear
                    for i, (audio_start, audio_end) in enumerate(zip(start[mini_batch], end[mini_batch])):
                        if i > len(tokens['attention_mask'][mini_batch]):
                            continue  # ignore longger than padding_size
                        if (audio_start == 0) and (audio_end == 0):
                            break  # padding appear
                        if audio_start < (t / self.fps) < audio_end:
                            tokens['attention_mask'][mini_batch][i] = 1 * self.alpha
        return tokens
    
    def get_audio_feature(self, SPEECH_FILE):
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import librosa
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        audio_input, _ = librosa.load(SPEECH_FILE, sr=16_000)
        
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
        audio_feature = torch.unsqueeze(audio_feature, dim=1)
        audio_feature = audio_feature.repeat(1, len(inputs_embeds[0]),1)  # [batch, feature_dim] -> [batch, padding_size, feature_dim]
        x = torch.cat((inputs_embeds, audio_feature), dim=2)
        embedding = self.MMfusion(x)
        return self.act(embedding)


    def forward(self, inputs, labels):
        '''
        inputs: start time, end time, T, tokens= single sentence, waveform= audio feature
        labels: responsive sentence
        '''
        start = inputs[0]   # [batch, start_len]
        end = inputs[1]     # [batch, end_len]
        T = inputs[2]       # [batch, T_len]
        tokens = inputs[3]
        audio_feature = inputs[4]   # [batch, 1, feature_dim]
        
        # preprocess
        audio_feature = torch.squeeze(audio_feature, dim=1)
        tokens['input_ids'] = torch.squeeze(tokens['input_ids'], dim=1)
        tokens['attention_mask'] = torch.squeeze(tokens['attention_mask'], dim=1)
        labels['input_ids'] = torch.squeeze(labels['input_ids'], dim=1)
        labels['attention_mask'] = torch.squeeze(labels['attention_mask'], dim=1)
        
        
        concat_labels = torch.cat([tokens['input_ids'], labels['input_ids']], dim=1)  # [batch,sentence_length]
        
        # ==== step 1. Give weight to word ====
        new_tokens = self.weighted_word(T, start, end, tokens)
        
        # ==== step 2. Extract audio feature ====
        audio_feature = self.projection_layer(audio_feature)
        
        # ==== step 3. Generate next sentence ====
        
        inputs_embeds = self.embedding_layer.weight.data[new_tokens['input_ids']]  # get the embedding layer weights
        labels_embeds = self.embedding_layer.weight.data[labels['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            inputs_embeds = self.multimodal_fusion(inputs_embeds, audio_feature)
        
        concat_inputs = torch.cat([inputs_embeds, labels_embeds], dim=1)  # [batch, sentence_length, word_dimension]
        concat_mask = torch.cat([new_tokens['attention_mask'], labels['attention_mask']], dim=1)  # [1, sentence_length]
        
        output = self.gpt_model(inputs_embeds=concat_inputs,
                                attention_mask=concat_mask,
                                labels=concat_labels
                                )
        
        sft_idx = tokens['input_ids'].shape[-1]
        p_loss = self.loss_function(output.logits[:,sft_idx-1:-1].contiguous().view(-1,50257), labels['input_ids'][:, :].contiguous().view(-1))
        
        return p_loss
        

    def inference(self, inputs, eos_token_id=50256):
        '''
        inputs: [image_list, audio_path, tokens, transcript, waveform], tokenizer.eos_token_id
        outputs: [text]
        ''' 
        frames = inputs[0]
        audio_path = inputs[1]
        tokens = inputs[2]
        transcript= inputs[3]
        waveform = inputs[4]
        
        prompt = tokens.input_ids
        
        step1 = time.time()
        _ , T = self.FER(frames)
        T = torch.unsqueeze(torch.tensor(T), dim=0)  # [batch_size=1, T_len]
        print("==== Step 1. [Facial Expression Recog]\t spent time: {:.4f} ====".format(time.time()-step1))
        
        step2 = time.time()
        word_timestamp = self.forced_alignment(audio_path, transcript)
        start = torch.unsqueeze(torch.tensor(word_timestamp[1]), dim=0)  # [batch_size=1, start_len]
        end = torch.unsqueeze(torch.tensor(word_timestamp[2]), dim=0)  # [batch_size=1, end_len]
        new_tokens = self.weighted_word(T, start, end, tokens)
        print("==== Step 2. [Give weight to word]\t spent time: {:.4f} ====".format(time.time()-step2))
        
        step3 = time.time()
        # waveform = self.get_audio_feature(audio_path)
        audio_feature = self.projection_layer(waveform)

        inputs_embeds = self.gpt_model.transformer.wte(new_tokens['input_ids'])
        
        if self.modal_fusion:
            inputs_embeds = self.multimodal_fusion(inputs_embeds, audio_feature)
        print("==== Step 3. [Audio feature Fusion]\t spent time: {:.4f} ====".format(time.time()-step3))
        
        step4 = time.time()

        output = self.gpt_model.generate(input_ids=prompt,
                                         max_length=100,
                                         pad_token_id=eos_token_id,
                                         inputs_embeds=inputs_embeds,
                                         attention_mask=new_tokens['attention_mask'],
                                         )
        
        print("==== Step 4. [Generate next sentence]\t spent time: {:.4f} ====".format(time.time()-step4))
        
        return output
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
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
    
    model = MyArch1(param, hyper_param).to(device)
    inputs = [word_timestamp[1], word_timestamp[2], T, tokens.to(device), waveform]
    loss = model(inputs, labels_token)
    
    print('loss: ', loss.item())

    