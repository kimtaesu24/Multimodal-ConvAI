#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from . import modules

# Yongsik Part 
from nltk.translate.bleu_score import sentence_bleu
from eval_metric.coco_eval import calculate_eval_matric

class MyArch(torch.nn.Module):
    def __init__(
            self,
            param,
            hyper_param,
    ):
        super(MyArch, self).__init__()
        self.give_weight = param['give_weight']
        self.modal_fusion = param['modal_fusion']
        self.multi_task = param['multi_task']
        self.trans_encoder = param['trans_encoder']
        self.forced_align = param['forced_align']
        self.landmark_append = param['landmark_append']
        
        self.fps = param['fps']
        self.device = param['device']

        self.batch_size = hyper_param['batch_size']
        self.max_length = hyper_param['max_length']
        self.alpha = hyper_param['alpha']
        self.dropout = hyper_param['dropout']
        self.audio_padding = hyper_param['audio_pad_size']
        if hyper_param['act'] =='relu':
            self.act = nn.ReLU()
        
        self.gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = '!'
        self.tokenizer.bos_token = '#'
        self.embedding_layer = self.gpt_model.get_input_embeddings()

        self.emotion_dic = {'neutral':0,
                'surprise':1,
                'fear':2,
                'sadness':3,
                'joy':4,
                'disgust':5,
                'anger':6}
        self.reverse_emotion_dic = {v:k for k,v in self.emotion_dic.items()}
        self.audio_feature_dimension = 768
        self.landmark_dimension = 196
        self.word_dimension = self.gpt_model.config.hidden_size  # 1280
        self.num_emotion = len(self.reverse_emotion_dic)  # 7
        
        if self.trans_encoder:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.word_dimension, nhead=8, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        if self.forced_align:
            self.projection_layer = nn.Linear(self.audio_feature_dimension * 2, self.word_dimension, bias=False)
            if self.landmark_append:
                self.MMfusion = nn.Linear((self.audio_padding//2 + 1) * self.word_dimension + self.landmark_dimension, self.word_dimension, bias=True)
            else:
                self.MMfusion = nn.Linear((self.audio_padding//2 + 1) * self.word_dimension, self.word_dimension, bias=True)
        else:
            self.projection_layer = nn.Linear(self.audio_feature_dimension, self.audio_feature_dimension, bias=False)
            if self.landmark_append:
                self.MMfusion = nn.Linear(self.word_dimension + self.audio_feature_dimension + self.landmark_dimension, self.word_dimension, bias=True)
            else:
                self.MMfusion = nn.Linear(self.word_dimension + self.audio_feature_dimension, self.word_dimension, bias=True)
        
        self.projection_layer.weight = torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
        
        if self.multi_task:
            self.emotion_analysis = nn.Linear(self.word_dimension, self.num_emotion, bias=True)
            self.emotion_analysis.weight = torch.nn.init.xavier_uniform_(self.emotion_analysis.weight)
        # self.feat_drop = nn.Dropout(self.dropout) if self.dropout > 0 else None
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)


    def forward(self, inputs, labels):
        start = inputs[0]   # [batch, max_length]
        end = inputs[1]     # [batch, max_length]
        T = inputs[2]       # [batch, T_len]
        tokens = inputs[3]
        audio_feature = inputs[4]   # [batch, 1, feature_dim]
        landmarks = inputs[5]  # [batch, max_length, landmark_dim]
        
        tokens_labels = labels[0]
        emotion_label = labels[1]
        
        
        # ==== step 0. preprocess ====
        tokens['input_ids'] = torch.squeeze(tokens['input_ids'], dim=1)  # [batch_size, padding_size]
        tokens['attention_mask'] = torch.squeeze(tokens['attention_mask'], dim=1)
        tokens_labels['input_ids'] = torch.squeeze(tokens_labels['input_ids'], dim=1)
        tokens_labels['attention_mask'] = torch.squeeze(tokens_labels['attention_mask'], dim=1)
        
        
        # ==== step 1. Give weight to word ====
        if self.give_weight:
            tokens = modules.weighted_word(T, start, end, tokens, self.fps, self.alpha)
        
        
        # ==== step 2. Extract audio feature ====
        if self.forced_align:
            audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1], audio_feature.shape[2]//2, -1)  # [batch, max_length, audio_pad_len/2, feature_dim*2]
        else:
            audio_feature = torch.squeeze(audio_feature, dim=1)
        audio_feature = torch.squeeze(audio_feature, dim=1)
        audio_feature = self.projection_layer(audio_feature.to(self.device))
        
        
        # ==== step 3. multimodal encoding ====
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids']]  # get the embedding layer weights
        labels_embeds = self.embedding_layer.weight.data[tokens_labels['input_ids']]  # get the embedding layer weights
        
        emotion_analysis_loss = 0.
        if self.modal_fusion:
            if self.forced_align:
                x = modules.forced_alignment_multimodal_concat(inputs_embeds, audio_feature)  # [batch, max_length, -1]
                if self.landmark_append:
                    x = torch.cat((x, landmarks), dim=2)
            else:
                x = modules.multimodal_concat(inputs_embeds, audio_feature)  # [batch, max_length, audio_feature_dim + word_dimension]
                if self.landmark_append:
                    x = torch.cat((x, landmarks), dim=2)
            inputs_embeds = self.act(self.MMfusion(x))
            
            if self.trans_encoder:
                bos = self.embedding_layer.weight.data[self.tokenizer.bos_token_id]  # bos_token
                bos_multimodal_embedding = torch.cat([bos.repeat(inputs_embeds.shape[0],1,1), inputs_embeds], dim=1)  # [batch, max_length + 1, word_dimension]
                bos_mask = torch.ones(inputs_embeds.shape[0], 1).to(self.device)
                bos_concat_mask = torch.cat([bos_mask, tokens['attention_mask']], dim=1)  # [batch, max_length + 1]
                
                feature = self.transformer_encoder(bos_multimodal_embedding, src_key_padding_mask=bos_concat_mask)
                inputs_embeds = feature[:,1:,:]
                emotions = feature[:,:1,:]
                if self.multi_task:
                    emotion_logits = self.emotion_analysis(emotions)
                    emotion_analysis_loss = self.loss_function(emotion_logits.contiguous().view(-1,self.num_emotion), emotion_label.contiguous().view(-1))
                    print(self.reverse_emotion_dic.get(int(torch.argmax(emotion_logits))))
        
        
        # ==== step 4. Generate next sentence ====
        concat_inputs = torch.cat([inputs_embeds, labels_embeds], dim=1)  # [batch, sentence_length*2, word_dimension]
        concat_mask = torch.cat([tokens['attention_mask'], tokens_labels['attention_mask']], dim=1)  # [batch, sentence_length*2]
        
        
        outputs = self.gpt_model(inputs_embeds=concat_inputs,
                                attention_mask=concat_mask,
                                )
        sft_idx = tokens['input_ids'].shape[-1]

        p_loss = self.loss_function(outputs.logits[:,sft_idx-1:-1].contiguous().view(-1,50257), tokens_labels['input_ids'][:, :].contiguous().view(-1))
        
        output = self.gpt_model.generate(max_length=100,
                                pad_token_id=self.tokenizer.pad_token_id,
                                inputs_embeds=inputs_embeds,
                                attention_mask=tokens['attention_mask'],
                                do_sample=True,
                                top_k=50,
                                top_p=0.90,
                                )
        
        bleu = self.get_bleu_score(output, labels[0])
        
        return p_loss + emotion_analysis_loss, bleu
        

    def inference(self, inputs, greedy=True):
        frames = inputs[0]
        audio_path = inputs[1]
        tokens = inputs[2].to(self.device)
        transcript= inputs[3]
        waveform = inputs[4].to(self.device)
        # landmarks = inputs[5]
        landmarks=0
        
        # ==== step 0. preprocess ====
        step0 = time.time()
        if self.give_weight:
            _ , T = modules.FER(frames)
            T = torch.unsqueeze(torch.tensor(T), dim=0)  # [batch_size=1, T_len]
        else:
            T = None
        word_timestamp = modules.forced_alignment(audio_path, transcript)
        start = modules.pad(word_timestamp[1], self.max_length)
        start = torch.unsqueeze(torch.tensor(start), dim=0)  # [batch_size=1, start_len]
        end = modules.pad(word_timestamp[2], self.max_length)
        end = torch.unsqueeze(torch.tensor(end), dim=0)  # [batch_size=1, end_len]
        print("==== Step 0. [Preprocess]\t spent time: {:.4f} ====".format(time.time()-step0))
        
        
        # ==== step 1. Give weight to word ====
        step1 = time.time()
        if self.give_weight:
            tokens = modules.weighted_word(T, start, end, tokens, self.fps, self.alpha)
        print("==== Step 1. [Facial Expression Recog]\t spent time: {:.4f} ====".format(time.time()-step1))            
                
                
        # ==== step 2. Extract audio feature ====
        step2 = time.time()
        if self.forced_align:
            audio_feature = modules.audio_word_align(waveform, audio_path, torch.squeeze(start), torch.squeeze(end), self.audio_padding)
            audio_feature = torch.unsqueeze(audio_feature, dim=0)
            audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1], audio_feature.shape[2]//2, -1)  # [batch, max_length, audio_pad_len/2, feature_dim*2]
        else:
            audio_feature = torch.mean(waveform, dim=1)
        audio_feature = self.projection_layer(audio_feature.contiguous().to(self.device))
        print("==== Step 2. [Extract audio feature]\t spent time: {:.4f} ====".format(time.time()-step2))


        # ==== step 3. multimodal encoding ====
        step3 = time.time()
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            if self.forced_align:
                x = modules.forced_alignment_multimodal_concat(inputs_embeds, audio_feature)
                if self.landmark_append:
                    x = torch.cat((x, landmarks), dim=2)
            else:
                x = modules.multimodal_concat(inputs_embeds, audio_feature)
                if self.landmark_append:
                    x = torch.cat((x, landmarks), dim=2)
            inputs_embeds = self.act(self.MMfusion(x))
            if self.trans_encoder:
                bos = self.embedding_layer.weight.data[self.tokenizer.bos_token_id]  # bos_token
                bos_multimodal_embedding = torch.cat([bos.repeat(inputs_embeds.shape[0],1,1), inputs_embeds], dim=1)  # [batch, max_length + 1, word_dimension]
                bos_mask = torch.ones(inputs_embeds.shape[0], 1).to(self.device)
                bos_concat_mask = torch.cat([bos_mask, tokens['attention_mask']], dim=1)  # [batch, max_length + 1]
                
                feature = self.transformer_encoder(bos_multimodal_embedding, src_key_padding_mask=bos_concat_mask)
                inputs_embeds = feature[:,1:,:]
                emotions = feature[:,:1,:]
                if self.multi_task:
                    emotion_logits = self.emotion_analysis(emotions)
                    print(self.reverse_emotion_dic.get(int(torch.argmax(emotion_logits))))
        print("==== Step 3. [multimodal encoding]\t spent time: {:.4f} ====".format(time.time()-step3))
        
        
        # ==== step 4. Generate next sentence ====
        step4 = time.time()
        if greedy:
            output = self.gpt_model.generate(max_length=100,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=inputs_embeds,
                                            attention_mask=tokens['attention_mask'],
                                            #  do_sample=True,
                                            #  top_k=50,
                                            #  top_p=0.90,
                                            )
        else:
            output = self.gpt_model.generate(max_length=100,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=inputs_embeds,
                                            attention_mask=tokens['attention_mask'],
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.90,
                                            )
        print("==== Step 4. [Generate next sentence]\t spent time: {:.4f} ====".format(time.time()-step4))
        
        return output
    

    def get_bleu_score(self, output, ref):
        outputs_sentence = self.tokenizer.decode(output.tolist()[0], skip_special_tokens=True)
        ref_sentence = self.tokenizer.decode(ref['input_ids'].tolist()[0], skip_special_tokens=True)
        
        outputs_values = outputs_sentence['a'].replace('!','').replace('.','').split()
        ref_values = [ref_sentence['ref'].replace('!','').replace('.','').split()]
        
        bleu_1 = format(sentence_bleu(ref_values, outputs_values, weights=(1, 0, 0, 0)), '.8f')
        bleu_2 = format(sentence_bleu(ref_values, outputs_values, weights=(0.5, 0.5, 0, 0)), '.8f')
        bleu_3 = format(sentence_bleu(ref_values, outputs_values, weights=(1/3, 1/3, 1/3, 0)), '.8f')
        bleu_4 = format(sentence_bleu(ref_values, outputs_values, weights=(0.25, 0.25, 0.25, 0.25)), '.8f')

        return bleu_1, bleu_2, bleu_3, bleu_4
    
    def get_eval_matric(self, output, ref):
        outputs_sentence = self.tokenizer.decode(output.tolist()[0], skip_special_tokens=True)
        ref_sentence = self.tokenizer.decode(ref['input_ids'].tolist()[0], skip_special_tokens=True)
        
        eval_result = calculate_eval_matric(outputs_sentence, ref_sentence)
        
        ## For Taesoo 
        # bleu_1 = eval_result['Bleu_1']
        # bleu_2 = eval_result['Bleu_2']
        # bleu_3 = eval_result['Bleu_3']
        # bleu_4 = eval_result['Bleu_4']
        
        # meteor = eval_result['METEOR']
        # rouge = eval_result['ROUGE_L']
        # cider = eval_result['CIDEr']
        # spice = eval_result['SPICE']
        return eval_result 
 