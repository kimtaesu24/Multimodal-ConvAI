#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
# from nltk.translate.bleu_score import sentence_bleu
# from statistics import mean 
from . import modules

# Yongsik Part 
# from nltk.translate.bleu_score import sentence_bleu
from eval_metric.coco_eval import calculate_eval_matric

torch.set_printoptions(profile="full")

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
        self.history_length = hyper_param['history_length']
        self.alpha = hyper_param['alpha']
        self.beta = hyper_param['beta']
        self.dropout = hyper_param['dropout']
        self.audio_padding = hyper_param['audio_pad_size']
        if hyper_param['act'] =='relu':
            self.act = nn.ReLU()
        
        self.gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')
        # self.tokenizer.pad_token = '!'
        # self.tokenizer.bos_token = '#'
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
            self.audio_projection_layer = nn.Linear(self.audio_feature_dimension * 2, self.word_dimension, bias=False)
            self.audio_projection_layer.weight = torch.nn.init.xavier_uniform_(self.audio_projection_layer.weight)
            if self.landmark_append:
                self.visual_projection_layer = nn.Linear(self.landmark_dimension, self.landmark_dimension, bias=False)
                self.visual_projection_layer.weight = torch.nn.init.xavier_uniform_(self.visual_projection_layer.weight)
                self.MMfusion = nn.Linear((self.audio_padding//2 + 1) * self.word_dimension + self.landmark_dimension, self.word_dimension, bias=True)
                self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
            else:
                self.MMfusion = nn.Linear((self.audio_padding//2 + 1) * self.word_dimension, self.word_dimension, bias=True)
                self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
        else:
            self.audio_projection_layer = nn.Linear(self.audio_feature_dimension, self.audio_feature_dimension, bias=False)
            self.audio_projection_layer.weight = torch.nn.init.xavier_uniform_(self.audio_projection_layer.weight)
            if self.landmark_append:
                self.MMfusion = nn.Linear(self.word_dimension + self.audio_feature_dimension + self.landmark_dimension, self.word_dimension, bias=True)
                self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
                self.visual_projection_layer = nn.Linear(self.landmark_dimension, self.landmark_dimension, bias=False)
                self.visual_projection_layer.weight = torch.nn.init.xavier_uniform_(self.visual_projection_layer.weight)
            else:
                self.MMfusion = nn.Linear(self.word_dimension + self.audio_feature_dimension, self.word_dimension, bias=True)
                self.MMfusion.weight = torch.nn.init.xavier_uniform_(self.MMfusion.weight)
        
        if self.multi_task:
            self.emotion_analysis = nn.Linear(self.word_dimension, self.num_emotion, bias=True)
            self.emotion_analysis.weight = torch.nn.init.xavier_uniform_(self.emotion_analysis.weight)
        # self.feat_drop = nn.Dropout(self.dropout) if self.dropout > 0 else None
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.emo_loss = nn.CrossEntropyLoss()


    def forward(self, inputs, labels, validation=False):
        start = inputs[0]   # [batch, max_length]
        end = inputs[1]     # [batch, max_length]
        T = inputs[2]       # [batch, T_len]
        tokens = inputs[3]
        audio_feature = inputs[4]   # [batch, 1, feature_dim]
        landmarks = inputs[5]  # [batch, max_length, landmark_dim]
        history_tokens = inputs[6]
        
        tokens_labels = labels[0]
        emotion_label = labels[1]
        
        
        # ==== step 0. preprocess ====
        tokens['input_ids'] = torch.squeeze(tokens['input_ids'], dim=1)  # [batch_size, padding_size]
        tokens['attention_mask'] = torch.squeeze(tokens['attention_mask'], dim=1)
        history_tokens['input_ids'] = torch.squeeze(history_tokens['input_ids'], dim=1)  # [batch_size, padding_size]
        history_tokens['attention_mask'] = torch.squeeze(history_tokens['attention_mask'], dim=1)
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
        audio_feature = self.audio_projection_layer(audio_feature.to(self.device))
        
        
        # ==== step 3. multimodal encoding ====
        history_embeds = self.embedding_layer.weight.data[history_tokens['input_ids']]  # get the embedding layer weights
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids']]  # get the embedding layer weights
        labels_embeds = self.embedding_layer.weight.data[tokens_labels['input_ids']]  # get the embedding layer weights
        
        emotion_analysis_loss = 0.
        if self.modal_fusion:
            if self.forced_align:
                x = modules.forced_alignment_multimodal_concat(inputs_embeds, audio_feature)  # [batch, max_length, -1]
                if self.landmark_append:
                    landmarks = self.visual_projection_layer(landmarks)
                    x = torch.cat((x, landmarks), dim=2)
            else:
                x = modules.multimodal_concat(inputs_embeds, audio_feature)  # [batch, max_length, audio_feature_dim + word_dimension]
                if self.landmark_append:
                    landmarks = self.visual_projection_layer(landmarks)
                    x = torch.cat((x, landmarks), dim=2)
            inputs_embeds = self.act(self.MMfusion(x))
            
            if self.trans_encoder:
                bos = self.embedding_layer.weight.data[self.tokenizer.bos_token_id]  # bos_token
                bos_multimodal_embedding = torch.cat([bos.repeat(inputs_embeds.shape[0],1,1), inputs_embeds], dim=1)  # [batch, max_length + 1, word_dimension]
                bos_mask = torch.ones(inputs_embeds.shape[0], 1).to(self.device)
                bos_concat_mask = torch.cat([bos_mask, tokens['attention_mask']], dim=1)  # [batch, max_length + 1]
                
                feature = self.transformer_encoder(bos_multimodal_embedding, src_key_padding_mask=bos_concat_mask)
                inputs_embeds = feature[:,1:,:]
                emotions = feature[:,:1,:]  # [batch, 1, word_dimension]
                if self.multi_task:
                    emotion_logits = self.emotion_analysis(torch.squeeze(emotions, dim=1))  # [batch, num_emotion]
                    # emotion_analysis_loss = self.emo_loss(emotion_logits.contiguous().view(-1,self.num_emotion), emotion_label.contiguous().view(-1))
                    emotion_analysis_loss = self.emo_loss(emotion_logits, emotion_label)                    
                    # print(self.reverse_emotion_dic.get(int(torch.argmax(emotion_logits))))
        
        
        # ==== step 4. Generate next sentence ====
        concat_inputs = torch.cat([history_embeds, inputs_embeds, labels_embeds], dim=1)
        concat_mask = torch.cat([history_tokens['attention_mask'], tokens['attention_mask'], tokens_labels['attention_mask']], dim=1)  
        # concat_inputs = torch.cat([inputs_embeds, labels_embeds], dim=1)
        # concat_mask = torch.cat([tokens['attention_mask'], tokens_labels['attention_mask']], dim=1)  
        
        
        outputs = self.gpt_model(inputs_embeds=concat_inputs,
                                attention_mask=concat_mask,
                                )
        
        sft_idx = tokens['input_ids'].shape[-1] + history_tokens['input_ids'].shape[-1]
        # sft_idx = tokens['input_ids'].shape[-1]
        p_loss = self.loss_function(outputs.logits[:,sft_idx-1:-1].contiguous().view(-1,50257), tokens_labels['input_ids'][:, :].contiguous().view(-1))
        
        total_loss = p_loss + self.beta * emotion_analysis_loss
        
        
        if validation:
            output = self.gpt_model.generate(max_length=self.max_length+self.history_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=torch.cat([history_embeds, inputs_embeds], dim=1),
                                            attention_mask=torch.cat([history_tokens['attention_mask'], tokens['attention_mask']], dim=1),
                                            num_beams=5,
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.90,
                                            )
        
            eval_result = self.get_eval_matric(output, tokens_labels['input_ids'])
        else:
            eval_result = None
        
        # if torch.isnan(torch.tensor(emotion_analysis_loss)):
        #     print("emotion_analysis_loss")
        #     exit()
        return total_loss, eval_result
        

    def inference(self, inputs, greedy=False):
        '''
        frames = inputs[0]
        audio_path = inputs[1]
        tokens = inputs[2].to(self.device)
        transcript= inputs[3]
        waveform = inputs[4].to(self.device)
        landmarks = inputs[5]
        
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
        audio_feature = self.audio_projection_layer(audio_feature.contiguous().to(self.device))
        print("==== Step 2. [Extract audio feature]\t spent time: {:.4f} ====".format(time.time()-step2))


        # ==== step 3. multimodal encoding ====
        step3 = time.time()
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            if self.forced_align:
                x = modules.forced_alignment_multimodal_concat(inputs_embeds, audio_feature)
                if self.landmark_append:
                    landmarks = self.visual_projection_layer(landmarks)
                    x = torch.cat((x, landmarks), dim=2)
            else:
                x = modules.multimodal_concat(inputs_embeds, audio_feature)
                if self.landmark_append:
                    landmarks = self.visual_projection_layer(landmarks)
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
                                            num_beams=5,
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.90,
                                            )
        print("==== Step 4. [Generate next sentence]\t spent time: {:.4f} ====".format(time.time()-step4))
        
        return output
        '''
        start = inputs[0]   # [max_length]
        end = inputs[1]     # [max_length]
        T = inputs[2]       # [T_len]
        tokens = inputs[3]
        audio_feature = inputs[4]   # [1, feature_dim]
        landmarks = inputs[5]  # [max_length, landmark_dim]
        history_tokens = inputs[6]
        
        # ==== step 0. preprocess ====
        start = torch.unsqueeze(start, dim=0)
        end = torch.unsqueeze(end, dim=0)
        T = torch.unsqueeze(T, dim=0)
        tokens['input_ids'] = torch.unsqueeze(tokens['input_ids'], dim=0)
        tokens['attention_mask'] = torch.unsqueeze(tokens['attention_mask'], dim=0)
        audio_feature = torch.unsqueeze(audio_feature, dim=0)
        landmarks = torch.unsqueeze(landmarks, dim=0)
        history_tokens['input_ids'] = torch.unsqueeze(history_tokens['input_ids'], dim=0)
        history_tokens['attention_mask'] = torch.unsqueeze(history_tokens['attention_mask'], dim=0)
        
        tokens['input_ids'] = torch.squeeze(tokens['input_ids'], dim=1)  # [batch_size, padding_size]
        tokens['attention_mask'] = torch.squeeze(tokens['attention_mask'], dim=1)
        history_tokens['input_ids'] = torch.squeeze(history_tokens['input_ids'], dim=1)  # [batch_size, padding_size]
        history_tokens['attention_mask'] = torch.squeeze(history_tokens['attention_mask'], dim=1)
        
        # ==== step 1. Give weight to word ====
        if self.give_weight:
            tokens = modules.weighted_word(T, start, end, tokens, self.fps, self.alpha)

        # ==== step 2. Extract audio feature ====
        if self.forced_align:
            audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1], audio_feature.shape[2]//2, -1)  # [batch, max_length, audio_pad_len/2, feature_dim*2]
        else:
            audio_feature = torch.squeeze(audio_feature, dim=1)
        audio_feature = torch.squeeze(audio_feature, dim=1)
        audio_feature = self.audio_projection_layer(audio_feature.to(self.device))
        
        # ==== step 3. multimodal encoding ====
        history_embeds = self.embedding_layer.weight.data[history_tokens['input_ids']]  # get the embedding layer weights
        inputs_embeds = self.embedding_layer.weight.data[tokens['input_ids']]  # get the embedding layer weights
        
        if self.modal_fusion:
            if self.forced_align:
                x = modules.forced_alignment_multimodal_concat(inputs_embeds, audio_feature)  # [batch, max_length, -1]
                if self.landmark_append:
                    landmarks = self.visual_projection_layer(landmarks)
                    x = torch.cat((x, landmarks), dim=2)
            else:
                x = modules.multimodal_concat(inputs_embeds, audio_feature)  # [batch, max_length, audio_feature_dim + word_dimension]
                if self.landmark_append:
                    landmarks = self.visual_projection_layer(landmarks)
                    x = torch.cat((x, landmarks), dim=2)
            inputs_embeds = self.act(self.MMfusion(x))
            
            if self.trans_encoder:
                bos = self.embedding_layer.weight.data[self.tokenizer.bos_token_id]  # bos_token
                bos_multimodal_embedding = torch.cat([bos.repeat(inputs_embeds.shape[0],1,1), inputs_embeds], dim=1)  # [batch, max_length + 1, word_dimension]
                bos_mask = torch.ones(inputs_embeds.shape[0], 1).to(self.device)
                bos_concat_mask = torch.cat([bos_mask, tokens['attention_mask']], dim=1)  # [batch, max_length + 1]
                
                feature = self.transformer_encoder(bos_multimodal_embedding, src_key_padding_mask=bos_concat_mask)
                inputs_embeds = feature[:,1:,:]
                emotions = feature[:,:1,:]  # [batch, 1, word_dimension]
                if self.multi_task:
                    emotion_logits = self.emotion_analysis(torch.squeeze(emotions, dim=1))  # [batch, num_emotion]
                    print(self.reverse_emotion_dic.get(int(torch.argmax(emotion_logits))))
        
        # ==== step 4. Generate next sentence ====
        if greedy:
            output = self.gpt_model.generate(max_length=self.max_length+self.history_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=torch.cat([history_embeds, inputs_embeds], dim=1),
                                            attention_mask=torch.cat([history_tokens['attention_mask'], tokens['attention_mask']], dim=1),
                                            )
        else:
            output = self.gpt_model.generate(max_length=self.max_length+self.history_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=torch.cat([history_embeds, inputs_embeds], dim=1),
                                            attention_mask=torch.cat([history_tokens['attention_mask'], tokens['attention_mask']], dim=1),
                                            num_beams=5,
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.90,
                                            )
        return output
        
    def get_eval_matric(self, output, ref):
        '''
        output: metric dictionary
            {'Bleu_1': 0.1353352829659427, 
            'Bleu_2': 0.00013533528303361024, 
            'Bleu_3': 1.3533528305616618e-05, 
            'Bleu_4': 4.2796774227674215e-06, 
            'METEOR': 0.14814814814814814, 
            'ROUGE_L': 0.45864661654135336, 
            'CIDEr': 0.0, 
            'SPICE': 0.0}
        '''
        outputs_sentence = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        ref_sentence = self.tokenizer.batch_decode(ref, skip_special_tokens=True)
        
        eval_result = calculate_eval_matric(outputs_sentence, ref_sentence)
        
        return eval_result 