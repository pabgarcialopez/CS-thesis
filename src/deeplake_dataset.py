import torch
import numpy as np
from torch.utils.data import Dataset
import deeplake # type: ignore

# Reuse your old mappings if you want string fields:
INSTRUMENT_SOURCE_ID2STR = {
    0: "acoustic",
    1: "electronic",
    2: "synthetic"
}
INSTRUMENT_FAMILY_ID2STR = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string",
    9: "synth_lead",
    10: "vocal"
}
QUALITIES_ID2STR = {
    0: "bright",
    1: "dark",
    2: "distortion",
    3: "fast_decay",
    4: "long_release",
    5: "multiphonic",
    6: "nonlinear_env",
    7: "percussive",
    8: "reverb",
    9: "tempo-synced"
}

class DeepLakeNSynth(Dataset):
    def __init__(self, partition, transform=None):
        if partition not in ['train', 'val', 'test']:
            raise AttributeError("Argument partition must be one of: 'train', 'val', 'test'")
        self._partition = partition
        self._transform = transform

        # 1. Load the dataset (read-only in 3.x)
        self.ds = deeplake.load(f"hub://activeloop/{partition}", read_only=True)
        self.length = len(self.ds)

        # 2. Build an in-memory metadata dict keyed by note string + a list of keys
        self._metadata = {}
        self._keys = []  # each entry is a note string
        for i in range(self.length):
            sample = self.ds[i]

            # 'note' is shape (1,) => index [0]
            note_arr = sample['note'].numpy()
            note_str = note_arr[0]
            if isinstance(note_str, bytes):
                note_str = note_str.decode('utf-8', errors='replace')

            pitch = int(sample['pitch'].numpy()[0])
            instrument = int(sample['instrument'].numpy()[0])
            instrument_family = int(sample['instrument_family'].numpy()[0])
            instrument_source = int(sample['instrument_source'].numpy()[0])
            velocity = int(sample['velocity'].numpy()[0])

            # 'qualities' is an array of indices => build a 10-element binary vector
            qualities_arr = sample['qualities'].numpy()  # e.g. [0,3,7] or []
            qualities_binary = np.zeros(10, dtype=int)
            for q_idx in qualities_arr:
                qualities_binary[int(q_idx)] = 1

            # Optionally map instrument_source to a string, etc.
            instrument_source_str = INSTRUMENT_SOURCE_ID2STR.get(instrument_source, "unknown")
            instrument_family_str = INSTRUMENT_FAMILY_ID2STR.get(instrument_family, "unknown")

            # Build a metadata dict
            metadata = {
                "pitch": pitch,
                "instrument": instrument,
                "instrument_family": instrument_family,
                "instrument_source": instrument_source,
                "velocity": velocity,
                "qualities": qualities_binary.tolist(),  # 10-element vector
                "note": note_str,
                "instrument_source_str": instrument_source_str,
                "instrument_family_str": instrument_family_str,
            }

            self._metadata[note_str] = metadata
            self._keys.append(note_str)

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, index):
        # 1. Retrieve note string
        key = self._keys[index]
        metadata = self._metadata[key]

        # 3. Retrieve the waveform from 'audios' tensor
        audio_np = self.ds['audios'][index].numpy()
        # shape might be (64000, 1), we want (1, 64000)
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        waveform = torch.from_numpy(audio_np).float()

        # 4. Retrieve sample_rate
        sr_np = self.ds['sample_rate'][index].numpy()
        sample_rate = int(sr_np[0])  # shape (1,)

        # 5. Apply transform if any
        if self._transform:
            waveform = self._transform(waveform)

        return metadata, waveform, sample_rate
    

def get_metadata(sample):
    """
    Given a Deep Lake NSynth sample (e.g. ds[idx]),
    returns a dictionary with numeric/text fields, including
    a 10-element binary vector for qualities.
    """
    # Each numeric field is shape (1,), so index [0] to get the scalar.
    pitch_val = int(sample['pitch'].numpy()[0])
    instrument_val = int(sample['instrument'].numpy()[0])
    instrument_family_val = int(sample['instrument_family'].numpy()[0])
    instrument_source_val = int(sample['instrument_source'].numpy()[0])
    velocity_val = int(sample['velocity'].numpy()[0])

    # 'qualities' is an array of indices. e.g. [0, 3, 7] or [].
    qualities_arr = sample['qualities'].numpy()  # shape (N,) where N can vary
    # Create a 10-element binary vector initialized to zeros.
    qualities_binary = np.zeros(10, dtype=int)
    for q_idx in qualities_arr:
        qualities_binary[int(q_idx)] = 1  # Mark index as 1 if present

    # 'note' is shape (1,) containing a string
    note_arr = sample['note'].numpy()  # shape (1,)
    note_val = note_arr[0]
    if isinstance(note_val, bytes):
        note_val = note_val.decode('utf-8', errors='replace')

    metadata = {
        "pitch": pitch_val,
        "instrument": instrument_val,
        "instrument_family": instrument_family_val,
        "instrument_source": instrument_source_val,
        "velocity": velocity_val,
        # "qualities_indices": qualities_arr.tolist(),    # e.g. [0,3,7]
        "qualities": qualities_binary.tolist(),  # e.g. [1,0,0,1,0,0,0,1,0,0]
        "note": note_val,
    }

    return metadata


# Example usage:
# ds = deeplake.load("hub://activeloop/nsynth-train", read_only=True)
# metadata0 = get_metadata(ds[127])
# print(metadata0)
