import torch
import sys
import os
import cv2
import torchaudio
import pickle
import librosa
import soundfile as sf
from PIL import Image
import moviepy.editor as mp
from transformers import AutoTokenizer, Wav2Vec2Processor, Wav2Vec2Model
from model.architecture import MyArch
from utils import log_param

# sys.path.insert(0, '/home2/s20235100/Conversational-AI/MyModel/src/model/')

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
tokenizer.pad_token = '!'

wave_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wave_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def video2frames(input_video):
    image_list = []
    vidcap = cv2.VideoCapture(input_video)
    success,image = vidcap.read()
    while success:
        image_list.append(Image.fromarray(image).convert('RGB'))
        success,image = vidcap.read()
    return image_list

def video2audio(input_video):
    my_clip = mp.VideoFileClip(input_video)
    my_clip.audio.write_audiofile(filename="sample.wav")
    audio_path = '/home2/s20235100/Conversational-AI/MyModel/src/sample.wav'
    return audio_path

def audio2text(audio_path):
    bundle= torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()

    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        
    with torch.inference_mode():
        emission, _ = model(waveform)
        
    class GreedyCTCDecoder(torch.nn.Module):
        def __init__(self, labels, blank=0):
            super().__init__()
            self.labels = labels
            self.blank = blank

        def forward(self, emission: torch.Tensor) -> str:
            """Given a sequence emission over labels, get the best path string
            Args:
            emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

            Returns:
            str: The resulting transcript
            """
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            return "".join([self.labels[i] for i in indices])

    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])
    print(transcript)
    context = transcript.replace("|", " ").lower()
    context = context[:-1] + "."
    print(context, type(context))
    
    return transcript, context

def audio2feature(SPEECH_FILE, desired_sampling_rate=16000):
    audio, sr = librosa.load(SPEECH_FILE, sr=None)  # Load the input audio
    print("1st sr: ",sr)
    resampled_audio = librosa.resample(audio, sr, desired_sampling_rate)  # Resample the audio to the desired sampling rate
    sf.write(SPEECH_FILE, resampled_audio, desired_sampling_rate)  # Save the resampled audio to a new file
    
    # Preprocess the audio input
    audio, sr = librosa.load(SPEECH_FILE, sr=16_000)  # Load the input audio
    print("2nd sr: ",sr)
    inputs = wave_processor(audio, sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        outputs = wave_model(**inputs)

    features = outputs.last_hidden_state  # Extract the audio features from the output
    print(features.shape)
    
    return features

def preprocess_video(input_video, max_length):
    image_list = video2frames(input_video=input_video)
    audio_path = video2audio(input_video=input_video)
    transcript, context = audio2text(audio_path)
    
    # text -> token
    tokens = tokenizer(context + tokenizer.eos_token,
                        padding='max_length',
                        max_length=max_length,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                        )
    
    waveform = audio2feature(audio_path, desired_sampling_rate=16000)
    
    inputs = [image_list, audio_path, tokens, transcript, waveform]
    '''
    file_path = "/home2/s20235100/Conversational-AI/MyModel/src/model/inference_file.pickle"
    with open(file_path, "wb") as file:
        pickle.dump(inputs, file)
    file.close()
    '''
    
    return inputs
    
def main(param, hyper_param, input_video, checkpoint):
    print(checkpoint)
    path = "/home2/s20235100/Conversational-AI/MyModel/pretrained_model/"
    weigth = checkpoint+"_epochs{'epochs': 200, 'act': 'relu', 'batch_size': 32, 'learning_rate': 5e-05, 'max_length': 60, 'alpha': 2, 'dropout': 0.2, 'decay_rate': 0.98}.pt"
    
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
                
    weight_path = path + f"{landmark_append}/{give_weight}/{modal_fusion}/{forced_align}/{trans_encoder}/{multi_task}/" + weigth
    model = MyArch(param=param, hyper_param=hyper_param)
    
    model.load_state_dict(torch.load(weight_path, map_location=param['device']))
    model.eval()
    
    print("==== preprocessing ====")
    inputs = preprocess_video(input_video, hyper_param['max_length'])

    outputs = model.inference(inputs, greedy=True)
    print(outputs)
    sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Response: {}".format(sentence))

if __name__ == '__main__':
    input_video = 'dia0_utt3.mp4'
    # input_video = 'dia855_utt6.mp4'
    # input_video = 'dia231_utt10.mp4'
    # input_video = 'dia12_utt9.mp4'
    
    param = dict()
    param['model'] = 'MyArch'
    param['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    param['fps'] = 24
    param['give_weight'] = True
    param['modal_fusion'] = True
    param['forced_align'] = False
    param['trans_encoder'] = False
    param['multi_task'] = False
    param['landmark_append'] = False
    log_param(param)

    hyper_param = dict()
    hyper_param['act'] = 'relu'
    hyper_param['batch_size'] = 1
    hyper_param['max_length'] = 60
    hyper_param['audio_pad_size'] = 50
    hyper_param['alpha'] = 2
    hyper_param['dropout'] = 0.2
    log_param(hyper_param)
    
    main(param, hyper_param, input_video, sys.argv[1])