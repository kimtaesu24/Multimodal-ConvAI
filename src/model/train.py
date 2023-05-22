import torch
import pandas as pd
import matplotlib.pyplot as plt

from .arch1_data import ARCH1_Dataset
from torch.utils.data import DataLoader
from .architecture1 import MyArch1
from .architecture2 import MyArch2
from tqdm import tqdm

import wandb

class MyTrainer:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path

    def train_with_hyper_param(self, param, hyper_param):
        model = param['model']
        save_at_every = param['save_at_every']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']
        decay_rate = hyper_param['decay_rate']
        batch_size = hyper_param['batch_size']
        max_length = hyper_param['max_length']
        
        if model == 'arch1':
            model = MyArch1(param, hyper_param).to(self.device)
            
            train_dataset = ARCH1_Dataset(self.data_path, mode='train', max_length=max_length, device=self.device)
            valid_dataset = ARCH1_Dataset(self.data_path, mode='valid', max_length=max_length, device=self.device)
            
            train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        #   num_workers=4,
                                        )
            valid_dataloader = DataLoader(dataset=valid_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        #   num_workers=4,
                                        )
        '''
        elif model == 'arch2':
            model = MyArch2(param, hyper_param).to(self.device)
            
            train_dataset = ARCH1_Dataset(self.data_path, mode='train', max_length=max_length, device=self.device)
            valid_dataset = ARCH1_Dataset(self.data_path, mode='valid', max_length=max_length, device=self.device)
            
            train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        #   num_workers=4,
                                        )
            valid_dataloader = DataLoader(dataset=valid_dataset,
                                        batch_size = 1,
                                        shuffle = False,
                                        #   num_workers=4,
                                        )
        '''    
        train_batch_len = len(train_dataloader)
        valid_batch_len = len(valid_dataloader)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
        
        wandb.init(project=f"dialog_Gen_{param['model']}")

        pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')
        for epoch in pbar:
            total_train_loss=0
            total_valid_loss=0
            
            # training
            model.train()
            prog_bar = tqdm(train_dataloader, position=1, leave=False, desc='batch')
            for i, (inputs, labels) in enumerate(prog_bar):
                optimizer.zero_grad()
                
                loss = model(inputs, labels)
                prog_bar.set_postfix({'loss': loss.item()})

                loss.backward()
                optimizer.step()

                # log
                total_train_loss += loss.item()
                
                if i % (100//batch_size) == 0:
                    wandb.log({'train_loss':loss.item()})

            # validation
            with torch.no_grad():
                model.eval()
                for inputs, labels in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    loss = model(inputs, labels)

                    total_valid_loss += loss.item()
                    
            wandb.log({'train_loss_epoch': total_train_loss/train_batch_len})
            wandb.log({'valid_loss_epoch': total_valid_loss/valid_batch_len})

            if (epoch + 1) % save_at_every == 0:
                if param['give_weight'] == True:
                    give_weight = 'T'
                else:
                    give_weight = 'F'
                    
                if param['modal_fusion'] == True:
                    modal_fusion = 'T'
                else:
                    modal_fusion = 'F'

                torch.save(model.state_dict(), '/home2/s20235100/Conversational-AI/MyModel/pretrained_model/arch1/give_weight_'+give_weight+'/modal_fusion_'+modal_fusion+'/'+str(epoch+1)+'_epochs'+str(hyper_param)+'.pt')
                pbar.write('Pretrained model has saved at Epoch: {:02} '.format(epoch+1))

            # scheduler.step()
            pbar.update()
        pbar.close()

        return model