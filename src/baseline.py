import torch
import sys
import torchaudio
import moviepy.editor as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

# sys.path.insert(0, '/home2/s20235100/Conversational-AI/MyModel/src/model/')

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_video(input_video, max_length):
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
    input_ids = tokenizer.encode(word + tokenizer.eos_token,
                            # padding='max_length',
                            # max_length=max_length,
                            # truncation=True,
                            # return_attention_mask=True,
                            return_tensors='pt'
                            )
    return input_ids
    
def main(max_length, input_video):
    input_ids = preprocess_video(input_video, max_length)
    # print(input_ids)
    
    inputs_embeds = model.transformer.wte(input_ids)
    # print(inputs_embeds)
    # print(model.generation_config)
    
    outputs = model.generate(max_length=1000, pad_token_id=tokenizer.eos_token_id, inputs_embeds=inputs_embeds)
    # print(outputs)
    print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    max_length = 60
    input_video = 'dia0_utt3.mp4'
    main(max_length, input_video)