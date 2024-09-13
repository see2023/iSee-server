import numpy as np
import torch

# from memoryview to tensor
# buf: wave bytes
# is_2d: if True, return 2D tensor, otherwise return 1D tensor
def memoryview_to_tensor(buf,  is_2d=False):
    audio_data = np.frombuffer(buf, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    if is_2d:
        audio_tensor = torch.from_numpy(audio_data.reshape(1, -1))
    else:
        audio_tensor = torch.from_numpy(audio_data)
    return audio_tensor

#from memoryview to np.ndarray
def memoryview_to_ndarray(buf, is_2d=False):
    audio_data = np.frombuffer(buf, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    if is_2d:
        audio_data = audio_data.reshape(1, -1)
    return audio_data