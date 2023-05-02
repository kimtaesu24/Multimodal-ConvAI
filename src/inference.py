import torch
import sys
import cv2
import torchaudio
from PIL import Image
import moviepy.editor as mp
from transformers import AutoTokenizer

sys.path.insert(0, '/home2/s20235100/Conversational-AI/MyModel/src/model/')
from model.model1 import MyModel1

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_video(input_video, max_length):
    # video -> video frames 
    image_list = []
    vidcap = cv2.VideoCapture(input_video)
    success,image = vidcap.read()
    count = 0
    while success:
        #cv2.imwrite(input_video+"/%06d.jpg" % count, image)     # save frame as JPEG file
        image_list.append(Image.fromarray(image).convert('RGB'))
        success,image = vidcap.read()
        count += 1
    print("finish! convert video to frame")
    
    # video -> audio
    my_clip = mp.VideoFileClip(input_video)
    my_clip.audio.write_audiofile(filename="sample.wav")
    audio_path = '/home2/s20235100/Conversational-AI/MyModel/src/sample.wav'
    print("finish! convert video to audio")
    
    # audio -> text
    SPEECH_FILE = audio_path

    bundle= torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()

    waveform, sample_rate = torchaudio.load(SPEECH_FILE)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        
    # with torch.inference_mode():
    #     features, _ = model.extract_features(waveform)
        
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

    print("finish! convert audio to text")
    
    tokens = tokenizer(word,
                        padding='max_length',
                        max_length=max_length,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                        )
    
    waveform = torch.load('/home2/dataset/MELD/audio_feature/train/dia0_utt3.pt')
    
    return [image_list, audio_path, tokens, transcript, waveform]
    
def main(param,
         hyper_param,
         input_video
         ):
    device = param['device']
    model = MyModel1(param=param, hyper_param=hyper_param)
    model.to(device)
    
    weight_path = '/home2/s20235100/Conversational-AI/MyModel/pretrained_model/modal_fusion_F/architecture1_100epochs.pt'
    model = torch.load(weight_path, map_location=device)
    model.eval()
    
    print(model)

    inputs = preprocess_video(input_video, hyper_param['max_length'])
    output = model.inference(inputs)
    sentence = tokenizer.decode(output[0], skip_special_tokens=True)
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