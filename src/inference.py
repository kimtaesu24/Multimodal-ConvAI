import torch
import sys
import os
import cv2
import torchaudio
from PIL import Image
import moviepy.editor as mp
from transformers import AutoTokenizer
from model.model1 import MyModel1

# sys.path.insert(0, '/home2/s20235100/Conversational-AI/MyModel/src/model/')
   
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token

def video2frames(input_video):
    image_list = []
    vidcap = cv2.VideoCapture(input_video)
    success,image = vidcap.read()
    count = 0
    while success:
        #cv2.imwrite(input_video+"/%06d.jpg" % count, image)     # save frame as JPEG file
        image_list.append(Image.fromarray(image).convert('RGB'))
        success,image = vidcap.read()
        count += 1
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
    word = transcript.replace("|", " ").lower()
    
    return transcript, word

def preprocess_video(input_video):
    image_list = video2frames(input_video=input_video)
    audio_path = video2audio(input_video=input_video)
    transcript, word = audio2text(audio_path)
    
    # text -> token
    tokens = tokenizer(word + tokenizer.eos_token,
                       return_tensors='pt',
                       )
    
    name_without_extension = os.path.splitext(input_video)[0]
    waveform = torch.load('/home2/dataset/MELD/audio_feature/train/'+name_without_extension+'.pt')
    
    return [image_list, audio_path, tokens, transcript, waveform]
    
def main(param, hyper_param, input_video):
    weight_path = '/home2/s20235100/Conversational-AI/MyModel/pretrained_model/architecture1_10epochs.pt'
    device = param['device']
    
    model = MyModel1(param=param, hyper_param=hyper_param)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    print("==== preprocessing ====")
    inputs = preprocess_video(input_video)
    history = inputs[2].input_ids.shape[-1]
    
    print("==== model forward start ====")
    outputs = model.inference(inputs, tokenizer.eos_token_id)
    print(outputs)
    print(outputs[0])
    sentence = tokenizer.decode(outputs[:, history:][0], skip_special_tokens=True)
    print("Response: {}".format(sentence))

if __name__ == '__main__':
    input_video = 'dia0_utt3.mp4'
    
    param = dict()
    param['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    param['fps'] = 24
    param['give_weight'] = True
    param['modal_fusion'] = True

    hyper_param = dict()
    hyper_param['act'] = 'relu'
    hyper_param['batch_size'] = 1
    hyper_param['max_length'] = 60
    hyper_param['alpha'] = 2
    
    main(param, hyper_param, input_video)