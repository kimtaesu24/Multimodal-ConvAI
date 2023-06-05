import torch
import wave
import cv2
import dlib
import numpy as np
# from spiga.inference.config import ModelConfig
# from spiga.inference.framework import SPIGAFramework
# detector = dlib.get_frontal_face_detector()
# dataset = 'wflw'
# processor = SPIGAFramework(ModelConfig(dataset))

def weighted_word(T, start, end, tokens, fps=24, alpha=2):
    '''
    Input: timestamp of expression transition / raw voice data of single speaker
    Output: 
    
    Goal: give weight to specific word's attention mask when transition occur
    Used: model.forward(), model.inference()
    '''
    for mini_batch in range(start.shape[0]):
        non_zero = torch.count_nonzero(tokens['attention_mask'][mini_batch])
        zeros = tokens['attention_mask'][mini_batch].shape[-1] - non_zero
        pre_t = 0
        for t in T[mini_batch]:
            if t == 0:  # padding appear
                break
            if (t - pre_t) >= fps:  # at least 1 second
                for i, (audio_start, audio_end) in enumerate(zip(start[mini_batch], end[mini_batch])):
                    if i > len(tokens['attention_mask'][mini_batch]):  # ignore when longger than padding_size
                        continue
                    if (audio_start == 0) and (audio_end == 0):  # padding appear
                        break
                    if audio_start < (t / fps) < audio_end:
                        if tokens['attention_mask'][mini_batch][i+zeros] < alpha:  # duplication block
                            tokens['attention_mask'][mini_batch][i+zeros] *= alpha 
            pre_t = t
    return tokens
   
def multimodal_concat(inputs_embeds, audio_feature):
    '''
    Input:  text embedding / audio feature
    Output: multimodal fused embedding
    Used: model.forward(), model.inference()
    '''
    audio_feature = torch.unsqueeze(audio_feature, dim=1)
    audio_feature = audio_feature.repeat(1, len(inputs_embeds[0]),1)  # [batch, audio_feature_dim] -> [batch, max_length, audio_feature_dim]
    x = torch.cat((inputs_embeds, audio_feature), dim=2)  # [batch, max_length, audio_feature_dim + word_dimension]
    return x

def forced_alignment_multimodal_concat(inputs_embeds, audio_feature):
    '''
    Input:  text embedding / audio feature
    Output: multimodal fused embedding
    Used: model.forward(), model.inference()
    '''
    inputs_embeds = torch.unsqueeze(inputs_embeds, dim=2) # [batch, max_length, 1, word_dimension]
    x = torch.cat((inputs_embeds, audio_feature), dim=2) # [batch, max_length, 26, word_dimension] 
    x = x.view(x.shape[0], x.shape[1], -1)  # [batch, max_length, -1]
    return x

def FER(frames):
    '''
    Input: frames of single speaker utterance video
    Output: speaker's facial expression list / timestamp of expression transition
    
    Module: DAN
    Used: model.inference()
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

def forced_alignment(audio_path, transcript):
    '''
    Input: raw voice data of single speaker
    Output: 2d array = [['word1', word2', ...], [start_t1, start_t2,...], [end_t1, end_t2, ...]]
    
    Module: WAV2VEC 
    Used: model.inference()
    '''
    from .Forced_Alignment.FA import get_dic as FA
    word_timestamp = FA(audio_path, transcript)
    
    return word_timestamp

def pad(inputs, max_length):
    '''
    Used: dataset, model.inference()
    '''
    tmp = [0 for i in range(max_length)]
    if len(inputs) > max_length:
        tmp[:len(inputs)] = inputs[:max_length]  # truncation
    else:
        tmp[:len(inputs)] = inputs  # padding
    return tmp
 
def get_wav_duration(file_path):
    '''
    Used: dataset, model.inference()
    '''
    with wave.open(file_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()  # Get the number of frames in the WAV file
        frame_rate = wav_file.getframerate()  # Get the frame rate (number of frames per second)
        duration = num_frames / frame_rate # Calculate the duration in seconds

        return duration
        
def audio_pad(inputs, padding_size):
    '''
    Used: dataset, model.inference()
    '''
    tmp = torch.zeros(padding_size, inputs.shape[-1])
    if inputs.shape[0] > padding_size:
        tmp[:inputs.shape[0], :] = inputs[:padding_size]  # truncation
    else:
        tmp[:inputs.shape[0], :] = inputs  # padding
    return tmp

def audio_word_align(waveform, audio_path, start, end, audio_padding=50):
    '''
    Used: dataset, model.inference()
    '''
    waveform = torch.squeeze(waveform)
        
    duration = get_wav_duration(audio_path)
    
    a = (waveform.shape[0] / duration)
    waveform_start = torch.tensor(start).clone() * a
    waveform_start = [ int(x)+1 for x in waveform_start ]
    waveform_end = torch.tensor(end).clone() * a
    waveform_end = [ int(x)+1 for x in waveform_end ]
    
    audio_feature = []
    for i, (s, e) in enumerate(zip(waveform_start, waveform_end)):
        if (i != 0) and (s == 1) and (e == 1):  # padding appear
            word_waveform = torch.zeros(audio_padding, waveform.shape[-1])
        else:
            word_waveform = waveform[s:e, :]  # split waveform along to word duration
            word_waveform = audio_pad(word_waveform, audio_padding)
        audio_feature.append(word_waveform)
    torch_audio_feature = torch.stack(audio_feature, dim=0)  # list to torch.tensor
    return torch_audio_feature, waveform_start

def img_to_landmark(img_path, detector, processor):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        rects = detector(gray,1)
        top_left = rects[0].tl_corner()
        x0 = top_left.x
        y0 = top_left.y
        w = rects[0].width()
        h = rects[0].height()
        bbox = [x0*1.0, y0*1.0, w, h]
        features = processor.inference(image, [bbox])
        landmarks = np.array(features['landmarks'][0])
        # headpose = np.array(features['headpose'][0])

        diag = (w**2 + h**2)**0.5
        landmarks[:,0:1] = (landmarks[:,0:1] - x0) / diag 
        landmarks[:,1:2] = (landmarks[:,1:2] - y0) / diag
        
        return landmarks.flatten()
    except:
        return [0 for i in range(196)]

def get_landmark(dir_path, start, fps):
    # landmark_list = []
    # for frame in start:
    #     if frame == 0:
    #         landmark_list.append([0 for i in range(196)])
    #         continue
    #     target = int(frame*fps+1)
    #     target = '{0:06d}'.format(target)
    #     landmark_list.append(img_to_landmark(dir_path+target+'.jpg', detector, processor))
    # # print(landmark_list)
    # return torch.tensor(np.array(landmark_list), dtype=torch.float32)
    return torch.tensor([])

def get_aligned_landmark(landmark_set, waveform_start):
    output_landmark = []
    for s in waveform_start:
        if s >= landmark_set.shape[0]:  # padding appear
            output_landmark.append(torch.zeros(landmark_set.shape[-1]))
        else:
            output_landmark.append(landmark_set[s])
            
    return torch.stack(output_landmark, dim=0)  # list to torch.tensor
        
    