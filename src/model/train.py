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
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')
        # self.tokenizer.pad_token = '!'
        # self.tokenizer.bos_token = '#'

    def train_with_hyper_param(self, param, hyper_param):
        model_name = param['model']
        save_at_every = param['save_at_every']
        debug_mode = param['debug']
        fps = param['fps']
        metric_at_every = param['metric_at_every']
                
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']
        decay_rate = hyper_param['decay_rate']
        batch_size = hyper_param['batch_size']
        
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
        
        train_dataset = MyArch_Dataset(self.data_path, mode='train', device=self.device, hyper_param=hyper_param, param=param)
        valid_dataset = MyArch_Dataset(self.data_path, mode='valid', device=self.device, hyper_param=hyper_param, param=param)
        
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
            # total_train_bleu_1=0
            # total_train_bleu_2=0
            # total_train_bleu_3=0
            # total_train_bleu_4=0
            # total_train_meteor=0
            # total_train_rouge=0
            # total_train_cider=0
            # total_train_spice=0
            
            total_valid_loss=0
            total_valid_bleu_1=0
            total_valid_bleu_2=0
            total_valid_bleu_3=0
            total_valid_bleu_4=0
            total_valid_meteor=0
            total_valid_rouge=0
            total_valid_cider=0
            total_valid_spice=0
            
            # training
            model.train()
            prog_bar = tqdm(train_dataloader, position=1, leave=False, desc='batch')
            for i, (inputs, labels) in enumerate(prog_bar):
                optimizer.zero_grad()
                
                loss, eval_result = model(inputs, labels)
                prog_bar.set_postfix({'loss': loss.item()})

                loss.backward()
                optimizer.step()

                # log
                total_train_loss += loss.item()
                # total_train_bleu_1 += eval_result['Bleu_1']
                # total_train_bleu_2 += eval_result['Bleu_2']
                # total_train_bleu_3 += eval_result['Bleu_3']
                # total_train_bleu_4 += eval_result['Bleu_4']
                
                # total_train_meteor += eval_result['METEOR']
                # total_train_rouge += eval_result['ROUGE_L']
                # total_train_cider += eval_result['CIDEr']
                # total_train_spice += eval_result['SPICE']
                
                if not debug_mode:  # code for debugging
                    if i % (100//batch_size) == 0:
                        wandb.log({'train_loss':loss.item()})
                        # wandb.log({'train_Bleu_1':eval_result['Bleu_1']})
                        # wandb.log({'train_Bleu_2':eval_result['Bleu_2']})
                        # wandb.log({'train_Bleu_3':eval_result['Bleu_3']})
                        # wandb.log({'train_Bleu_4':eval_result['Bleu_4']})
                        
                        # wandb.log({'train_METEOR':eval_result['METEOR']})
                        # wandb.log({'train_ROUGE_L':eval_result['ROUGE_L']})
                        # wandb.log({'train_CIDEr':eval_result['CIDEr']})
                        # wandb.log({'train_SPICE':eval_result['SPICE']})
                    
            # sample check
            '''
            outputs = model.inference(inference_file, greedy=True)
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Response: {}".format(sentence))
            '''
            
            # validation
            with torch.no_grad():
                model.eval()
                metric_log = (epoch+1) % metric_at_every == 0
                for inputs, labels in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    loss, eval_result = model(inputs, labels, validation=metric_log)

                    total_valid_loss += loss.item()
                    
                    if metric_log:
                        total_valid_bleu_1 += eval_result['Bleu_1']
                        total_valid_bleu_2 += eval_result['Bleu_2']
                        total_valid_bleu_3 += eval_result['Bleu_3']
                        total_valid_bleu_4 += eval_result['Bleu_4']
                        
                        total_valid_meteor += eval_result['METEOR']
                        total_valid_rouge += eval_result['ROUGE_L']
                        total_valid_cider += eval_result['CIDEr']
                        total_valid_spice += eval_result['SPICE']
                
            if not debug_mode:  # code for debugging
                wandb.log({'train_loss (epoch)': total_train_loss/train_batch_len})
                wandb.log({'valid_loss (epoch)': total_valid_loss/valid_batch_len})

                if metric_log:
                    # wandb.log({'train_Bleu-1 (epoch)': total_train_bleu_1/train_batch_len})
                    # wandb.log({'train_Bleu-2 (epoch)': total_train_bleu_2/train_batch_len})
                    # wandb.log({'train_Bleu-3 (epoch)': total_train_bleu_3/train_batch_len})
                    # wandb.log({'train_Bleu-4 (epoch)': total_train_bleu_4/train_batch_len})
                    # wandb.log({'train_METEOR (epoch)': total_train_meteor/train_batch_len})
                    # wandb.log({'train_ROUGE_L (epoch)': total_train_rouge/train_batch_len})
                    # wandb.log({'train_CIDEr (epoch)': total_train_cider/train_batch_len})
                    # wandb.log({'train_SPICE (epoch)': total_train_spice/train_batch_len})
                    wandb.log({'valid_Bleu-1 (epoch)': total_valid_bleu_1/valid_batch_len})
                    wandb.log({'valid_Bleu-2 (epoch)': total_valid_bleu_2/valid_batch_len})
                    wandb.log({'valid_Bleu-3 (epoch)': total_valid_bleu_3/valid_batch_len})
                    wandb.log({'valid_bleu-4 (epoch)': total_valid_bleu_4/valid_batch_len})
                    wandb.log({'valid_METEOR (epoch)': total_valid_meteor/valid_batch_len})
                    wandb.log({'valid_ROUGE_L (epoch)': total_valid_rouge/valid_batch_len})
                    wandb.log({'valid_CIDEr (epoch)': total_valid_cider/valid_batch_len})
                    wandb.log({'valid_SPICE (epoch)': total_valid_spice/valid_batch_len})

            # save checkpoint
            if (epoch+1) % save_at_every == 0:
                torch.save(model.state_dict(), f"/home2/s20235100/Conversational-AI/MyModel/pretrained_model/{landmark_append}/{give_weight}/{modal_fusion}/{forced_align}/{trans_encoder}/{multi_task}/{str(epoch+1)}_epochs{str(hyper_param)}.pt")
                pbar.write('Pretrained model has saved at Epoch: {:02} '.format(epoch+1))

            scheduler.step()
            pbar.update()
        pbar.close()

        return model
    
    