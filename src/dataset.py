from utils.dataset_processing import *
from utils.audio_plotting import *
from utils.decorators import chronometer

import torchaudio
from torch.utils.data import Dataset

class NSynth(Dataset):

    def __init__(self, partition, transform=None):
        # Partition is a string: 'training', 'validation', or 'testing'

        self._partition = partition
        self._transform = transform

        self._metadata = load_json(partition)
        self._metadata = process_metadata(self._metadata)
        
        # Keys are used by the getitem function
        self._keys = list(self._metadata.keys())

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        # Get the key corresponding to this index
        key = self._keys[index]
        metadata = self._metadata[key]

        # Get the audio file ready for torchaudio
        wav_file = get_audio_file(f"{key}.wav", self._partition)
    
        # Wrap the bytes in a BytesIO object so torchaudio can read it
        waveform, sample_rate = torchaudio.load(wav_file, format="wav")

        # waveform.shape = [num_channels, time] = [1, num_samples = 4 * 16000 = 64000]

        if self._transform:
            waveform = self._transform(waveform)
        
        return metadata, waveform, sample_rate
