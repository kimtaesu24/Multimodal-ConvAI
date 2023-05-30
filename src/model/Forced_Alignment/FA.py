import torch
import torchaudio
from dataclasses import dataclass

import re
#import IPython
# import matplotlib
# import matplotlib.pyplot as plt


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        print("ValueError('Failed to align')")
        return
    return path[::-1]

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

    
def get_dic(SPEECH_FILE, transcript):
    return main(SPEECH_FILE, transcript)


def main(SPEECH_FILE, transcript):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

    torch.random.manual_seed(0)

    #SPEECH_FILE = audio_path

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    with torch.inference_mode():
        waveform, sample_rate = torchaudio.load(SPEECH_FILE)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    #transcript = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
    #transcript = "SO|LETS|TALK|A|LITTLE|BIT|ABOUT|YOUR|DUTIES"
    #transcript = "HEYHEYHEY|YOU|WANNA|HEAR|SOMETHING|THAT|SUCKS"

    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]
    #print(list(zip(transcript, tokens)))

    trellis = get_trellis(emission, tokens)
    
    path = backtrack(trellis, emission, tokens)
    if path == None:
        return
    '''for p in path:
        print(p)'''
    
    segments = merge_repeats(path, transcript)
    # for seg in segments:
        # print(seg)
    
    word_segments = merge_words(segments)
    # for word in word_segments:
    #     print(word)
        
    word_list = []
    start_list = []
    end_list = []
    
    for i in range(len(word_segments)):
        ratio = waveform.size(1) / (trellis.size(0) - 1)
        single_word = word_segments[i]
        x0 = int(ratio * single_word.start)
        x1 = int(ratio * single_word.end)
        # seg_dic[single_word.label] = [x0 / bundle.sample_rate, x1 / bundle.sample_rate]  # cannot handle 중복
        word_list.append(single_word.label)
        start_list.append(x0 / bundle.sample_rate)
        end_list.append(x1 / bundle.sample_rate)
    
    return [word_list, start_list, end_list]
    
if __name__=="__main__":
    SPEECH_FILE = 'train/dia0/utt0/dia0_utt0.wav'
    
    transcript = "also I was the point person on my company's transition from the KL-5 to GR-6 system."
    pattern_punctuation = re.compile(r'[^\w\s\']')
    output_string = pattern_punctuation.sub('', transcript)
    output_string = re.sub(r"[0-9]", "", output_string)
    print(output_string)
    transcript = output_string.upper().replace(" ","|")
    print(transcript)
    
    seg_dic = main(SPEECH_FILE, transcript)
    print(seg_dic)