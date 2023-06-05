import torch
import pickle
import datetime

from .arch_data import MyArch_Dataset
from torch.utils.data import DataLoader
from .architecture import MyArch
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb

class MyTrainer:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer.pad_token = '!'
        self.tokenizer.bos_token = '#'

    def train_with_hyper_param(self, param, hyper_param):
        model_name = param['model']
        save_at_every = param['save_at_every']
        debug_mode = param['debug']
        fps = param['fps']
        
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']
        decay_rate = hyper_param['decay_rate']
        batch_size = hyper_param['batch_size']
        max_length = hyper_param['max_length']
        audio_pad_size = hyper_param['audio_pad_size']
        
        if param['give_weight'] == True:
            give_weight = 'give_weight_T'
        else:
            give_weight = 'give_weight_F'
        if param['modal_fusion'] == True:
            modal_fusion = 'modal_fusion_T'
        else:
            modal_fusion = 'modal_fusion_F'
        if param['forced_align'] == True:
            forced_align = 'forced_align_T'
        else:
            forced_align = 'forced_align_F'
        if param['trans_encoder'] == True:
            trans_encoder = 'trans_encoder_T'
        else:
            trans_encoder = 'trans_encoder_F'
        if param['multi_task'] == True:
            multi_task = 'multi_task_T'
        else:
            multi_task = 'multi_task_F'
        if param['landmark_append'] == True:
            landmark_append = 'landmark_append_T'
        else:
            landmark_append = 'landmark_append_F'
        
        model = MyArch(param, hyper_param).to(self.device)
        
        train_dataset = MyArch_Dataset(self.data_path, mode='train', max_length=max_length, FA=param['forced_align'], LM=param['landmark_append'], audio_padding=audio_pad_size, fps=fps, device=self.device)
        valid_dataset = MyArch_Dataset(self.data_path, mode='valid', max_length=max_length, FA=param['forced_align'], LM=param['landmark_append'], audio_padding=audio_pad_size, fps=fps, device=self.device)
        
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
            
        train_batch_len = len(train_dataloader)
        valid_batch_len = len(valid_dataloader)
        '''
        with open("/home2/s20235100/Conversational-AI/MyModel/src/model/inference_file.pickle", "rb") as file:
            inference_file = pickle.load(file)  # inference data for every end of epochs
            outputs = model.inference(inference_file, greedy=True)
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Response: {}".format(sentence))
        '''
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
        
        if not debug_mode:  # code for debugging
            wandb.init(project=f"dialog_Gen")
            d = datetime.datetime.today()
            wandb.run.name = model_name + d.strftime('%c')

        pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')
        for epoch in pbar:
            total_train_loss=0
            total_train_bleu_1=0
            total_train_bleu_2=0
            total_train_bleu_3=0
            total_train_bleu_4=0
            
            total_valid_loss=0
            total_valid_bleu_1=0
            total_valid_bleu_2=0
            total_valid_bleu_3=0
            total_valid_bleu_4=0
            
            # training
            model.train()
            prog_bar = tqdm(train_dataloader, position=1, leave=False, desc='batch')
            for i, (inputs, labels) in enumerate(prog_bar):
                optimizer.zero_grad()
                
                loss, metrics = model(inputs, labels)
                prog_bar.set_postfix({'loss': loss.item()})

                loss.backward()
                optimizer.step()

                # log
                total_train_loss += loss.item()
                total_train_bleu_1 += metrics[0]
                total_train_bleu_2 += metrics[1]
                total_train_bleu_3 += metrics[2]
                total_train_bleu_4 += metrics[3]

                
                if not debug_mode:  # code for debugging
                    if i % (100//batch_size) == 0:
                        wandb.log({'train_loss':loss.item()})
                        wandb.log({'train_bleu-1':metrics[0]})
                        wandb.log({'train_bleu-2':metrics[1]})
                        wandb.log({'train_bleu-3':metrics[2]})
                        wandb.log({'train_bleu-4':metrics[3]})
                    
            # sample check
            '''
            outputs = model.inference(inference_file, greedy=True)
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Response: {}".format(sentence))
            '''
            
            # validation
            with torch.no_grad():
                model.eval()
                for inputs, labels in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    loss, metrics = model(inputs, labels)

                    total_valid_loss += loss.item()
                    
                    total_valid_bleu_1 += metrics[0]
                    total_valid_bleu_2 += metrics[1]
                    total_valid_bleu_3 += metrics[2]
                    total_valid_bleu_4 += metrics[3]
                
            if not debug_mode:  # code for debugging
                wandb.log({'train_loss_epoch': total_train_loss/train_batch_len})
                wandb.log({'train_bleu-1_epoch': total_train_bleu_1/valid_batch_len})
                wandb.log({'train_bleu-2_epoch': total_train_bleu_2/valid_batch_len})
                wandb.log({'train_bleu-3_epoch': total_train_bleu_3/valid_batch_len})
                wandb.log({'train_bleu-4_epoch': total_train_bleu_4/valid_batch_len})
                
                wandb.log({'valid_loss_epoch': total_valid_loss/valid_batch_len})
                wandb.log({'valid_bleu-1_epoch': total_valid_bleu_1/valid_batch_len})
                wandb.log({'valid_bleu-2_epoch': total_valid_bleu_2/valid_batch_len})
                wandb.log({'valid_bleu-3_epoch': total_valid_bleu_3/valid_batch_len})
                wandb.log({'valid_bleu-4_epoch': total_valid_bleu_4/valid_batch_len})

            # save checkpoint
            if (epoch + 1) % save_at_every == 0:
                torch.save(model.state_dict(), f"/home2/s20235100/Conversational-AI/MyModel/pretrained_model/{landmark_append}/{give_weight}/{modal_fusion}/{forced_align}/{trans_encoder}/{multi_task}/{str(epoch+1)}_epochs{str(hyper_param)}.pt")
                pbar.write('Pretrained model has saved at Epoch: {:02} '.format(epoch+1))

            scheduler.step()
            pbar.update()
        pbar.close()

        return model