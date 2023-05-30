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
        self.give_weight = param['give_weight']
        self.modal_fusion = param['modal_fusion']
        self.device = param['device']

        self.batch_size = hyper_param['batch_size']
        self.max_length = hyper_param['max_length']
        self.alpha = hyper_param['alpha']
        self.dropout = hyper_param['dropout']
        if hyper_param['act'] =='relu':
            self.act = nn.ReLU()
        self.audio_feature_dimension = 768
        self.gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = '!'
        self.embedding_layer = self.gpt_model.get_input_embeddings()

        self.projection_layer = nn.Linear(self.audio_feature_dimension, self.audio_feature_dimension, bias=False)
        self.projection_layer.weight = torch.nn.init.xavier_uniform_(self.projection_layer.weight)
                                                             
        self.MMfusion = nn.Linear(1280+self.audio_feature_dimension, 1280, bias=True)
        self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
        # self.feat_drop = nn.Dropout(self.dropout) if self.dropout > 0 else None
        
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)


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
                non_zero = torch.count_nonzero(tokens['attention_mask'][mini_batch])
                zeros = tokens['attention_mask'][mini_batch].shape[-1] - non_zero
                # pre_t = 0
                for t in T[mini_batch]:
                    if t == 0:  # padding appear
                        break
                    # if (t - pre_t) >= self.fps:  # at least 1 second
                    for i, (audio_start, audio_end) in enumerate(zip(start[mini_batch], end[mini_batch])):
                        if i > len(tokens['attention_mask'][mini_batch]):  # ignore when longger than padding_size
                            continue
                        if (audio_start == 0) and (audio_end == 0):  # padding appear
                            break
                        if audio_start < (t / self.fps) < audio_end:
                            if tokens['attention_mask'][mini_batch][i+zeros] < self.alpha:  # duplication block
                                tokens['attention_mask'][mini_batch][i+zeros] *= self.alpha 
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
        tokens['input_ids'] = torch.squeeze(tokens['input_ids'], dim=1)  # [batch_size, padding_size]
        tokens['attention_mask'] = torch.squeeze(tokens['attention_mask'], dim=1)
        labels['input_ids'] = torch.squeeze(labels['input_ids'], dim=1)
        labels['attention_mask'] = torch.squeeze(labels['attention_mask'], dim=1)
        
        # ==== step 1. Give weight to word ====
        new_tokens = self.weighted_word(T, start, end, tokens)
        
        # ==== step 2. Extract audio feature ====
        audio_feature = self.projection_layer(audio_feature)
        
        # ==== step 3. Generate next sentence ====  
        inputs_embeds = self.embedding_layer.weight.data[new_tokens['input_ids']]  # get the embedding layer weights
        labels_embeds = self.embedding_layer.weight.data[labels['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            inputs_embeds = self.multimodal_fusion(inputs_embeds, audio_feature)
        
        concat_inputs = torch.cat([inputs_embeds, labels_embeds], dim=1)  # [batch, sentence_length*2, word_dimension]
        concat_mask = torch.cat([new_tokens['attention_mask'], labels['attention_mask']], dim=1)  # [batch, sentence_length*2]
        
        outputs = self.gpt_model(inputs_embeds=concat_inputs,
                                attention_mask=concat_mask,
                                )
        sft_idx = tokens['input_ids'].shape[-1]

        p_loss = self.loss_function(outputs.logits[:,sft_idx-1:-1].contiguous().view(-1,50257), labels['input_ids'][:, :].contiguous().view(-1))
        
        return p_loss
        

    def inference(self, inputs):
        '''
        inputs: [image_list, audio_path, tokens, transcript, waveform], tokenizer.eos_token_id
        outputs: [text]
        ''' 
        frames = inputs[0]
        audio_path = inputs[1]
        tokens = inputs[2].to(self.device)
        transcript= inputs[3]
        waveform = inputs[4].to(self.device)
        
        step1 = time.time()
        if self.give_weight:
            _ , T = self.FER(frames)
            T = torch.unsqueeze(torch.tensor(T), dim=0)  # [batch_size=1, T_len]
        else:
            T = None
        # print("==== Step 1. [Facial Expression Recog]\t spent time: {:.4f} ====".format(time.time()-step1))
        
        step2 = time.time()
        word_timestamp = self.forced_alignment(audio_path, transcript)
        # print(word_timestamp)
        # word_timestamp[1] = [0.4014375, 0.602125, 0.8630625, 1.1039375, 1.1641875, 1.364875, 1.5455625, 1.766375, 1.9269375]
        # word_timestamp[2] =[0.562, 0.802875, 1.0638125, 1.124, 1.3448125, 1.5255, 1.74625, 1.88675, 2.187875]
        start = pad(word_timestamp[1], self.max_length)
        start = torch.unsqueeze(torch.tensor(start), dim=0)  # [batch_size=1, start_len]
        end = pad(word_timestamp[2], self.max_length)
        end = torch.unsqueeze(torch.tensor(end), dim=0)  # [batch_size=1, end_len]
        new_tokens = self.weighted_word(T, start, end, tokens)
        # print("==== Step 2. [Give weight to word]\t spent time: {:.4f} ====".format(time.time()-step2))
        
        step3 = time.time()
        # print(waveform.shape)
        # waveform = torch.load('/home2/dataset/MELD/audio_feature/train/dia0_utt3_16000.pt')
        audio_feature = torch.mean(waveform, dim=1)
        # print(audio_feature.shape)
        audio_feature = self.projection_layer(audio_feature)

        # inputs_embeds = self.gpt_model.transformer.wte(new_tokens['input_ids'])
        inputs_embeds = self.embedding_layer.weight.data[new_tokens['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            inputs_embeds = self.multimodal_fusion(inputs_embeds, audio_feature)
        # print("==== Step 3. [Audio feature Fusion]\t spent time: {:.4f} ====".format(time.time()-step3))
        
        step4 = time.time()

        output = self.gpt_model.generate(max_length=100,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         inputs_embeds=inputs_embeds,
                                         attention_mask=new_tokens['attention_mask'],
                                        #  do_sample=True,
                                        #  top_k=50,
                                        #  top_p=0.90,
                                         )
        
        # print("==== Step 4. [Generate next sentence]\t spent time: {:.4f} ====".format(time.time()-step4))
        
        return output

def pad(inputs, max_length):
    tmp = [0 for i in range(max_length)]
    if len(inputs) > max_length:
        tmp[:len(inputs)] = inputs[:max_length]  # truncation
    else:
        tmp[:len(inputs)] = inputs  # padding
    return tmp