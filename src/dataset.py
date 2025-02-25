import json
from config import BASE_PATH
import torchaudio
from torch.utils.data import Dataset

class NSynth(Dataset):
    def __init__(self, partition, transform=None):

        # Partition is a string: 'training', 'validation', or 'test'
        self._audio_dir = BASE_PATH / 'data' / partition / 'audio'
        audio_info_path = BASE_PATH / 'data' / partition / 'example.json'
        with open(audio_info_path, 'r') as file:
            self._metadata = json.load(file)
        self._keys = list(self.metadata.keys())
    
        self._transform = transform

    def __len__(self):
        # Return the total number of samples.
        return len(self._metadata)

    def __getitem__(self, index):
        # Get the key corresponding to this index
        key = self._keys[index]

        # Retrieve metadata for this key
        metadata = self._metadata[key]
        
        # Construct the audio filename (assume .wav extension)
        audio_path = self._audio_dir / (key + '.wav')
        
        # Load the audio (e.g., using torchaudio)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Optionally apply a transformation
        if self._transform:
            waveform = self._transform(waveform)
        
        # Return waveform along with any metadata you need
        return waveform, sample_rate, metadata

