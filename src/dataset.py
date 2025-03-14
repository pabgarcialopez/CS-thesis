from src.utils.dataset import *
from src.utils.audio_plotting import *
from src.utils.decorators import chronometer

import torchaudio
from torch.utils.data import Dataset

class NSynth(Dataset):
    def __init__(self, partition, transform=None):
        self._partition = partition
        self._transform = transform

        json_data = load_json(partition)
        self._metadata = process_metadata(json_data)
        self._keys = list(self._metadata.keys())

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        key = self._keys[index]
        metadata = self._metadata[key]

        # load the raw .wav from local folder
        wav_path = DATA_PATH / self._partition / "audio" / f"{key}.wav"
        waveform, sample_rate = torchaudio.load(str(wav_path))

        if self._transform:
            waveform = self._transform(waveform)

        # Return (waveform, sample_rate, key, metadata)
        return waveform, sample_rate, key, metadata
