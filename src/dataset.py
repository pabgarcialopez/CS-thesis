from src.utils.dataset import *
from src.utils.audio_plotting import *
from src.utils.decorators import chronometer

import torchaudio
from torch.utils.data import Dataset

class NSynth(Dataset):

    def __init__(self, partition, transform=None):
        # Partition is a string: 'training', 'validation', or 'testing'

        self._partition = partition
        self._transform = transform

        json_data = load_json(partition)
        self._metadata = process_metadata(json_data)
        
        # Keys are used by the getitem function
        self._keys = list(self._metadata.keys())

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        # Get the key corresponding to this index
        key = self._keys[index]
        metadata = self._metadata[key]

        # Get the audio file ready for torchaudio
        wav_path = DATA_PATH / self._partition / "audio" / f"{key}.wav"
        waveform, sample_rate = torchaudio.load(str(wav_path))
        # waveform.shape = [num_channels, time] = [1, num_samples = 4 * 16000 = 64000]

        if self._transform:
            waveform = self._transform(waveform)
        
        return metadata, waveform, sample_rate
