import torch
import pandas as pd
import matplotlib.pyplot as plt

from .data import MyDataset
from torch.utils.data import DataLoader
from .model1 import MyModel1
from tqdm import tqdm
from loguru import logger

class MyTrainer:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path
        self.train_losses = []
        # self.train_recall = []
        # self.train_ndcg = []
        self.valid_losses = []
        # self.valid_recall = []
        # self.valid_ndcg = []

    def train_with_hyper_param(self, param, hyper_param):
        save_at_every = param['save_at_every']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']
        decay_rate = hyper_param['decay_rate']
        batch_size = hyper_param['batch_size']
        max_length = hyper_param['max_length']
        
        train_dataset = MyDataset(self.data_path, mode='train', max_length=max_length, device=self.device)
        valid_dataset = MyDataset(self.data_path, mode='valid', max_length=max_length, device=self.device)
        
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size = batch_size,
                                      shuffle = False,
                                    #   num_workers=4,
                                      )
        
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size = 1,
                                      shuffle = False,
                                    #   num_workers=4,
                                      )
        
        # train_batch_len = len(train_dataloader)
        # valid_batch_len = len(valid_dataloader)
        
        model = MyModel1(param, hyper_param).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batch_size, gamma=decay_rate)
        
        # patience = 20
        # inc = 0
        # early_stopping = False
        # highest_val_ndcg = 0

        pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')
        for epoch in pbar:
            total_train_loss = 0
            total_valid_loss = 0
            #total_recall = 0
            #total_ndcg = 0
            train_batch_len = 0
            valid_batch_len = 0
            
            # training
            model.train()
            for inputs, labels in tqdm(train_dataloader, position=1, leave=False, desc='batch'):
                optimizer.zero_grad()
                #loss, recall_k, ndcg = model(inputs, labels)
                loss = model(inputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # log
                total_train_loss += loss.item()
                #total_recall += recall_k
                #total_ndcg += ndcg
                
                #self.train_recall.append(recall_k)
                #self.train_ndcg.append(ndcg)
                train_batch_len += 1

            # validation
            with torch.no_grad():
                model.eval()
                for inputs, labels in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    loss = model(inputs, labels)

                    total_valid_loss += loss.item()
                    # print(loss.item())
                    valid_batch_len += 1
                    '''
                    if val_ndcg >= highest_val_ndcg:
                        highest_val_ndcg = val_ndcg
                        inc = 0
                    else:
                        inc += 1

                
                # early stopping
                if inc >= patience:
                    early_stopping = True
                    break
                

            if early_stopping:
                pbar.write('Early stop at epoch: {}, batch steps: {}'.format(epoch+1, batch))
                pbar.update(pbar.total)
                break
            '''
            self.train_losses.append(total_train_loss/train_batch_len)
            self.valid_losses.append(total_valid_loss/valid_batch_len)
            
            if (epoch + 1) % save_at_every == 0:
                if param['give_weight'] == True:
                    give_weight = 'T'
                else:
                    give_weight = 'F'
                    
                if param['modal_fusion'] == True:
                    modal_fusion = 'T'
                else:
                    modal_fusion = 'F'

                torch.save(model.state_dict(), '/home2/s20235100/Conversational-AI/MyModel/pretrained_model/arch1/give_weight_'+give_weight+'/modal_fusion_'+modal_fusion+'/'+str(epoch + 1)+'epochs.pt')
                pbar.write('Pretrained model has saved at Epoch: {:02} '.format(epoch+1))

            '''
            pbar.write(
                'Epoch {:02}: train loss: {:.4}\t  train recall@20: {:.4}\t  train NDCG: {:.4}'
                .format(epoch+1, total_loss/batch_len, total_recall/batch_len, total_ndcg/batch_len))
            pbar.write(
                'Epoch {:02}: valid loss: {:.4}\t  valid recall@20: {:.4}\t  valid NDCG: {:.4}\n'
                .format(epoch+1, val_loss, val_recall_k, val_ndcg))
            '''
                
            pbar.write('Epoch {:02}: train loss: {:.4}'.format(epoch+1, total_train_loss/train_batch_len))
            pbar.write('Epoch {:02}: val loss: {:.4}'.format(epoch+1, total_valid_loss/valid_batch_len))
            pbar.update()

        pbar.close()

        # plot training loss graph
        plt.figure(figsize=(10, 5))
        plt.title(" Training Loss")
        plt.plot(self.train_losses, label="train")
        plt.plot(self.valid_losses, label="val")
        plt.xlabel("epochs")
        plt.ylabel("cross entropy loss")
        plt.legend()
        plt.savefig('training loss.png')
        plt.clf()
        
        # plot training metric graph
        '''plt.figure(2, figsize=(10, 5))
        plt.title(" Training metric")
        plt.plot(self.train_recall, label="recall")
        plt.plot(self.train_ndcg, label="ndcg")
        plt.xlabel("time step (=iterations)")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(' training metric --data_name ' + str(data_name) + ' --seed ' + str(seed) + ' --emb ' + str(embedding_size) + '.png')
        plt.clf()

        # plot validation metric graph
        plt.figure(3, figsize=(10, 5))
        plt.title(" Validation metric")
        plt.plot(self.val_recall, label="recall")ß
        plt.plot(self.val_ndcg, label="ndcg")
        plt.xlabel("time step (=batch_size/10)")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(' validation metric --data_name ' + str(data_name) + ' --seed ' + str(seed) + ' --emb ' + str(embedding_size) + '.png')
        plt.clf()'''

        return model

