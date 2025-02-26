from config import BASE_PATH
from utils.dataset_processing import *
from utils.audio_plotting import *

import io, json, zipfile, torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram

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
        # Return the dataset's length
        return len(self._metadata)

    def __getitem__(self, index):
        
        # Get the key corresponding to this index
        key = self._keys[index]
        metadata = self._metadata[key]

        # Get the audio file ready for torchaudio
        wav_file = get_audio_file(f"{key}.wav", self._partition)
    
        # Wrap the bytes in a BytesIO object so torchaudio can read it
        waveform, sample_rate = torchaudio.load(wav_file, format="wav")

        if self._transform:
            waveform = self._transform(waveform)
        
        return metadata, waveform, sample_rate
    

# Notes:

# 1) The waveform returned by __getitem__ is a tensor of shape [num_channels, time]. In our case,
#    we have mono audios, so num_channels = 1, and time "= num_samples" = 64000 (4 seconds * 16000 samples/second)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Visualización de una onda de audio sin transformación, y su Mel-espectograma
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1) Instancia del dataset crudo
raw_dataset = NSynth('testing', transform=None)

# 2) Instancia de la transformación MelSpectrogram
mel_transform = MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

# 3) Seleccionamos un índice para ver un ejemplo
metadata, waveform, sample_rate = raw_dataset[0]  # ejemplo de índice 0
# waveform.shape -> [1, num_samples], por ser mono
print(waveform.shape)

# 4) Plot waveform
plot_waveform(waveform)

# 5) Aplicar la transformación para obtener el espectrograma
mel_spec = mel_transform(waveform)  # Returns [1, n_mels, time_frames]

# A veces se aplica una conversión a dB para que se vea mejor:
db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
mel_spec_db = db_transform(mel_spec)
plot_spectogram(mel_spec_db)