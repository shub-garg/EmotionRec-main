from transformers import AutoTokenizer, AutoModel
import av
import torch
import numpy as np

from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

np.random.seed(0)


def read_video_pyav(container):

    frames = []
    container.seek(0)
    # start_index = indices[0]
    # end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        # if i > end_index:
        #     break
        # if i >= start_index and i in indices:
        frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):

    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path =r"D:\Research\DL Depression\EmoRec\backend\video.webm"
container = av.open(file_path)
print(container)
print(container.size)

# sample 8 frames
# indices = sample_frame_indices(clip_len=1, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container)

tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

inputs = processor(videos=list(video), return_tensors="pt")

video_features = model.get_video_features(**inputs)

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)


print(text_features.shape)
print(video_features.shape)