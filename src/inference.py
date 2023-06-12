import torch
import sys
import pandas as pd
import ast
import json

from transformers import AutoTokenizer
from model.architecture import MyArch
from utils import log_param
from model import modules

# sys.path.insert(0, '/home2/s20235100/Conversational-AI/MyModel/src/model/')

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'

def load_data(data_path, hyper_param, param, dia, utt):
    audio_feature_path = data_path + 'audio_feature/test'
    single_file_path = data_path + 'test'
    max_length = hyper_param["max_length"]
    history_length = hyper_param["history_length"]
    audio_padding = hyper_param["audio_pad_size"]
    forced_align = param["forced_align"]
    landmark_append = param["landmark_append"]
    
    FA = pd.read_csv(data_path + 'new_test_FA_matched.csv')
    fer = pd.read_csv(data_path + 'new_test_FER_matched.csv')
    # emotion = pd.read_csv(data_path + 'new_test_emotion_matched.csv')
    landmark = pd.read_csv(data_path + 'new_test_LM_matched.csv')
    history_path = data_path+ 'test'
    T_padding = max(len(i) for i in fer['T_list'].apply(eval))
    
    idx = FA[(FA['Dialogue_ID'] == dia) & (FA['Utterance_ID'] == utt)].index[0]
    
    context = ' '.join(ast.literal_eval(FA['word'][idx])).lower() + '.'
    
    start = ast.literal_eval(FA['start'][idx])
    start = modules.pad(start, max_length)
    end = ast.literal_eval(FA['end'][idx])
    end = modules.pad(end, max_length)
    
    T = ast.literal_eval(fer['T_list'][idx])
    if len(T) < T_padding:  # padding
        T = modules.pad(T, T_padding)
    
    tokens = tokenizer(context + tokenizer.eos_token,
                        padding='max_length',
                        max_length=max_length,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                        )

    waveform = torch.load(audio_feature_path+f'/dia{dia}_utt{utt}_16000.pt')
    if forced_align:
        audio_path = single_file_path+f'/dia{dia}/utt{utt}/dia{dia}_utt{utt}_16000.wav'
        audio_feature, waveform_start = modules.audio_word_align(waveform, audio_path, start, end, audio_padding)
    else:
        audio_feature = torch.mean(waveform, dim=1)
        waveform_start = None
    
    if landmark_append:
        landmark_set = torch.tensor(ast.literal_eval(landmark['landmark_list'][idx]))
        landmarks = modules.get_aligned_landmark(landmark_set, waveform_start)
    else:
        landmarks = torch.tensor([])
        
    with open(f"{history_path}/dia{dia}/utt{utt}/dia{dia}_utt{utt}_history.json", "r") as json_file:
        historys = json.load(json_file)
        
    input_historys = ""
    for utt_hist in historys:
        input_historys += utt_hist+tokenizer.eos_token
        
    input_historys_tokens = tokenizer(input_historys,
                                        padding='max_length',
                                        max_length=history_length,
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_tensors='pt'
                                        )
                            
    inputs = [torch.tensor(start).to(param['device']),
            torch.tensor(end).to(param['device']), 
            torch.tensor(T).to(param['device']), 
            tokens.to(param['device']),
            audio_feature.to(param['device']),
            landmarks.to(param['device']),
            input_historys_tokens.to(param['device']),
            ]

    historys.append(context)
    
    return inputs, historys
   
   
def main(param, hyper_param, dia, utt, checkpoint, batch_size):
    data_path = '/home2/dataset/MELD/'
        
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

    weight_path = "../pretrained_model/" + f"{landmark_append}/{give_weight}/{modal_fusion}/{forced_align}/{trans_encoder}/{multi_task}/" + \
                    str(checkpoint) + "_epochs{'epochs': "+str(checkpoint)+", 'act': 'relu', 'batch_size': "+str(batch_size)+", 'learning_rate': 5e-05, 'max_length': 60, 'history_length': 256, 'audio_pad_size': 50, 'alpha': 2, 'dropout': 0.2, 'decay_rate': 0.98}.pt"
    model = MyArch(param=param, hyper_param=hyper_param)
    
    model.load_state_dict(torch.load(weight_path))
    model.to(device=param['device'])
    model.eval()
    print('pretrained model has loaded')
    
    inputs, historys = load_data(data_path, hyper_param, param, dia, utt)
    # for i, k in enumerate(historys):
    #     print(f'{i}-th sentence: {k}')
    
    outputs = model.inference(inputs, greedy=False)
    # print(outputs)
    sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Response: {}".format(sentence))


if __name__ == '__main__':    
    param = dict()
    param['model'] = sys.argv[1]
    param['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    param['data_name'] = 'MELD'
    param['fps'] = 24
    
    if param['model'] == 'Arch0':
        param['give_weight'] = False
        param['modal_fusion'] = False
        param['forced_align'] = False
        param['trans_encoder'] = False
        param['multi_task'] = False
        param['landmark_append'] = False
        checkpoint=200
        batch_size=20
    elif param['model'] == 'Arch1':
        param['give_weight'] = True
        param['modal_fusion'] = True
        param['forced_align'] = False
        param['trans_encoder'] = True
        param['multi_task'] = False
        param['landmark_append'] = False
        checkpoint=200
        batch_size=20
    elif param['model'] == 'Arch2':
        param['give_weight'] = False
        param['modal_fusion'] = True
        param['forced_align'] = True
        param['trans_encoder'] = True
        param['multi_task'] = True
        param['landmark_append'] = False
        checkpoint=100
        batch_size=16
    elif param['model'] == 'Arch3':
        param['give_weight'] = False
        param['modal_fusion'] = True
        param['forced_align'] = True
        param['trans_encoder'] = True
        param['multi_task'] = True
        param['landmark_append'] = True
        checkpoint=100
        batch_size=16
    else:
        print('model name error')
        exit()
    log_param(param)

    hyper_param = dict()
    hyper_param['act'] = 'relu'
    hyper_param['batch_size'] = 1
    hyper_param['max_length'] = 60
    hyper_param['history_length'] = 256
    hyper_param['audio_pad_size'] = 50
    hyper_param['alpha'] = 2
    hyper_param['beta'] = 1
    hyper_param['dropout'] = 0.2
    log_param(hyper_param)
    
    print('checkpoint at:', checkpoint)
    print('batch_size with:', batch_size)
    
    dia_utt=[[13,6], 
             [16, 8], 
             [37, 6], 
             [65, 9],
             [80, 8], 
             [99, 7], 
             [114, 4], 
             [121, 4], 
             [127, 4], 
             [131, 7], 
             [183, 8],
             ]
    for d_u in dia_utt:
        dia = d_u[0]
        utt = d_u[1]
        print(f'Test Data: dia{dia}_utt{utt}.mp4')
        
        main(param, hyper_param, dia, utt, checkpoint=checkpoint, batch_size=batch_size)
        