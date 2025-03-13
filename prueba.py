import numpy as np
import deeplake

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
        "qualities_indices": qualities_arr.tolist(),    # e.g. [0,3,7]
        "qualities_binary": qualities_binary.tolist(),  # e.g. [1,0,0,1,0,0,0,1,0,0]
        "note": note_val,
    }

    return metadata


# Example usage:
ds = deeplake.load("hub://activeloop/nsynth-train", read_only=True)
metadata0 = get_metadata(ds[127])
print(metadata0)
