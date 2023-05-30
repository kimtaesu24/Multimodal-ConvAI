#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import time
import wave
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_printoptions(profile="full")

class MyArch2(torch.nn.Module):
    def __init__(
            self,
            param,
            hyper_param,
    ):
        super(MyArch2, self).__init__()
        '''
        self.embedding_size = hyper_param['embedding_size']
        self.dropout = hyper_param['dropout']
        self.num_layers = num_layers
        '''
        self.fps = param['fps']
        self.give_weight = param['give_weight']
        self.give_weight = False
        self.modal_fusion = param['modal_fusion']
        self.modal_fusion = True
        self.multi_task = param['multi_task']
        self.trans_encoder = param['trans_encoder']
        self.device = param['device']

        self.batch_size = hyper_param['batch_size']
        self.max_length = hyper_param['max_length']
        self.alpha = hyper_param['alpha']
        self.dropout = hyper_param['dropout']
        if hyper_param['act'] =='relu':
            self.act = nn.ReLU()
        self.audio_feature_dimension = 768
        self.word_dimension = 1280
        self.num_emotion = 7
        
        self.gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = '!'
        self.tokenizer.bos_token = '#'
        self.embedding_layer = self.gpt_model.get_input_embeddings()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.word_dimension, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.projection_layer = nn.Linear(self.audio_feature_dimension*2, self.word_dimension, bias=False)
        self.projection_layer.weight = torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        
        self.MMfusion = nn.Linear((25+1)*self.word_dimension, self.word_dimension, bias=True)
        self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
        
        self.emotion_analysis = nn.Linear(self.word_dimension, self.num_emotion, bias=True)
        self.emotion_analysis.weight = torch.nn.init.xavier_uniform_(self.emotion_analysis.weight)
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
                for t in T[mini_batch]:
                    if t == 0:
                        break  # padding appear
                    for i, (audio_start, audio_end) in enumerate(zip(start[mini_batch], end[mini_batch])):
                        if i > len(tokens['attention_mask'][mini_batch]):
                            continue  # ignore when longger than padding_size
                        if (audio_start == 0) and (audio_end == 0):
                            break  # padding appear
                        if audio_start < (t / self.fps) < audio_end:
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
    
    
    def multimodal_fusion(self, inputs_embeds, audio_feature, mask):
        '''
        Input:  text embedding / audio feature
        Output: multimodal fused embedding
        '''
        inputs_embeds = torch.unsqueeze(inputs_embeds, dim=2) # [batch, max_length, 1, 1280]
        x = torch.cat((inputs_embeds, audio_feature), dim=2) # [batch, max_length, 26, 1280] 
        x = x.view(x.shape[0], x.shape[1], -1)  # [batch, max_length, 33280]
        multimodal_embedding = self.act(self.MMfusion(x))  # [batch, max_length, word_dimension]
        
        if self.trans_encoder:
            bos = self.embedding_layer.weight.data[self.tokenizer.bos_token_id]  # bos_token
            bos_multimodal_embedding = torch.cat([bos.repeat(multimodal_embedding.shape[0],1,1), multimodal_embedding], dim=1)  # [batch, max_length + 1, word_dimension]
            
            bos_mask = torch.ones(multimodal_embedding.shape[0], 1).to(self.device)
            bos_concat_mask = torch.cat([bos_mask, mask], dim=1)  # [batch, max_length + 1]
            
            feature = self.transformer_encoder(bos_multimodal_embedding, src_key_padding_mask=bos_concat_mask)
            final_embeds = feature[:,1:,:]
            emotions = feature[:,:1,:]
        else:
            final_embeds = multimodal_embedding
            emotions = None
            
        return final_embeds, emotions

    def forward(self, inputs, labels):
        '''
        inputs: start time, end time, T, tokens= single sentence, waveform= audio feature
        labels: responsive sentence
        '''
        start = inputs[0]   # [batch, max_length]
        end = inputs[1]     # [batch, max_length]
        T = inputs[2]       # [batch, T_pad_len]
        tokens = inputs[3]
        audio_feature = inputs[4]   # [batch, max_length, audio_pad_len, feature_dim]
        
        tokens_labels = labels[0]
        emotion_label = labels[1]
        
        # preprocess
        audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1], audio_feature.shape[2]//2, -1)  # [batch, max_length, audio_pad_len/2, feature_dim*2]
        tokens['input_ids'] = torch.squeeze(tokens['input_ids'], dim=1)  # [batch_size, max_length]
        tokens['attention_mask'] = torch.squeeze(tokens['attention_mask'], dim=1)  # [batch_size, max_length]
        tokens_labels['input_ids'] = torch.squeeze(tokens_labels['input_ids'], dim=1)
        tokens_labels['attention_mask'] = torch.squeeze(tokens_labels['attention_mask'], dim=1)
        
        # ==== step 1. Give weight to word ====
        new_tokens = self.weighted_word(T, start, end, tokens)
        
        # ==== step 2. Extract audio feature ====
        audio_feature = self.projection_layer(audio_feature)  # [batch, max_length, audio_pad_len/2, word_dimension]
        # ==== step 3. Generate next sentence ====  
        inputs_embeds = self.embedding_layer.weight.data[new_tokens['input_ids']]  # get the embedding layer weights
        labels_embeds = self.embedding_layer.weight.data[tokens_labels['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            inputs_embeds, emotions = self.multimodal_fusion(inputs_embeds, audio_feature, new_tokens['attention_mask'])
            if self.multi_task:
                emotion_logits = self.emotion_analysis(emotions)
                emotion_analysis_loss = self.loss_function(emotion_logits.contiguous().view(-1,self.num_emotion), emotion_label.contiguous().view(-1))
            else:
                emotion_analysis_loss = 0.
        
        concat_inputs = torch.cat([inputs_embeds, labels_embeds], dim=1)  # [batch, max_length*2, word_dimension]
        concat_mask = torch.cat([new_tokens['attention_mask'], tokens_labels['attention_mask']], dim=1)  # [batch, max_length*2]
        
        outputs = self.gpt_model(inputs_embeds=concat_inputs,
                                 attention_mask=concat_mask,
                                 )
        sft_idx = tokens['input_ids'].shape[-1]

        p_loss = self.loss_function(outputs.logits[:,sft_idx-1:-1].contiguous().view(-1,50257), tokens_labels['input_ids'][:, :].contiguous().view(-1))
        
        return 0.8*p_loss + 0.2*emotion_analysis_loss
        

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
        
        # step1 = time.time()
        # _ , T = self.FER(frames)
        # T = torch.unsqueeze(torch.tensor(T), dim=0)  # [batch_size=1, T_len]
        # print("==== Step 1. [Facial Expression Recog]\t spent time: {:.4f} ====".format(time.time()-step1))
        
        step2 = time.time()
        word_timestamp = self.forced_alignment(audio_path, transcript)
        start = pad(word_timestamp[1], self.max_length)
        start = torch.unsqueeze(torch.tensor(start), dim=0)  # [batch_size=1, start_len]
        
        end = pad(word_timestamp[2], self.max_length)
        end = torch.unsqueeze(torch.tensor(end), dim=0)  # [batch_size=1, end_len]
        # new_tokens = self.weighted_word(T, start, end, tokens)
        print("==== Step 1. [Word forced alignment]\t spent time: {:.4f} ====".format(time.time()-step2))
        
        step3 = time.time()
        # waveform = self.get_audio_feature(audio_path)
        waveform = torch.squeeze(waveform)
        
        duration = get_wav_duration(audio_path)
        
        a = (waveform.shape[0] / duration)
        waveform_start = torch.tensor(start) * a
        waveform_end = torch.tensor(end) * a
        FA_waveform = []
        audio_padding = 50
        for i, (s, e) in enumerate(zip(waveform_start[0], waveform_end[0])):
            if (i != 0) and (s == 0.) and (e == 0.):  # padding appear
                word_waveform = torch.zeros(audio_padding, waveform.shape[-1])
            else:
                word_waveform = waveform[int(s):int(e), :]  # split waveform along to word duration
                word_waveform = audio_pad(word_waveform, audio_padding)
            FA_waveform.append(word_waveform)
        FA_waveform = torch.stack(FA_waveform, dim=0)  # list to torch.tensor
        print("==== Step 2. [Audio forced alignment]\t spent time: {:.4f} ====".format(time.time()-step3))
        
        step4 = time.time()
        FA_waveform = torch.unsqueeze(FA_waveform, dim=0)
        FA_waveform = FA_waveform.contiguous().view(FA_waveform.shape[0], FA_waveform.shape[1], FA_waveform.shape[2]//2, -1)  # [batch, max_length, audio_pad_len/2, feature_dim*2]
        audio_feature = self.projection_layer(FA_waveform)

        # inputs_embeds = self.gpt_model.transformer.wte(new_tokens['input_ids'])
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            inputs_embeds, emotions = self.multimodal_fusion(inputs_embeds, audio_feature, tokens['attention_mask'])
            if self.multi_task:
                emotion_dic = {'neutral':0,
                                'surprise':1,
                                'fear':2,
                                'sadness':3,
                                'joy':4,
                                'disgust':5,
                                'anger':6}
                reverse_emotion_dic = {v:k for k,v in emotion_dic.items()}
                emotion_logits = self.emotion_analysis(emotions)
                print(reverse_emotion_dic.get(int(torch.argmax(emotion_logits))))
        print("==== Step 3. [Audio feature Fusion]\t spent time: {:.4f} ====".format(time.time()-step4))
        
        step5 = time.time()

        output = self.gpt_model.generate(max_length=100,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         inputs_embeds=inputs_embeds,
                                         attention_mask=tokens['attention_mask'],
                                        #  do_sample=True,
                                        #  top_k=50,
                                        #  top_p=0.90,
                                         )
        
        print("==== Step 4. [Generate next sentence]\t spent time: {:.4f} ====".format(time.time()-step5))
        
        return output
   
def pad(inputs, max_length):
    tmp = [0 for i in range(max_length)]
    if len(inputs) > max_length:
        tmp[:len(inputs)] = inputs[:max_length]  # truncation
    else:
        tmp[:len(inputs)] = inputs  # padding
    return tmp
 
def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the number of frames in the WAV file
        num_frames = wav_file.getnframes()

        # Get the frame rate (number of frames per second)
        frame_rate = wav_file.getframerate()

        # Calculate the duration in seconds
        duration = num_frames / frame_rate

        return duration
        
def audio_pad(inputs, padding_size):
    tmp = torch.zeros(padding_size, inputs.shape[-1])
    if inputs.shape[0] > padding_size:
        tmp[:inputs.shape[0], :] = inputs[:padding_size]  # truncation
    else:
        tmp[:inputs.shape[0], :] = inputs  # padding
    return tmp

