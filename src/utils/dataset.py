import json
from pathlib import Path
from config import DATA_PATH
import torchaudio

# ------------------------------------------------------------------------------
# Hard-coded mappings for NSynth, based on the official specification.
# ------------------------------------------------------------------------------

# instrument_source -> instrument_source_str (0=acoustic, 1=electronic, 2=synthetic)
INSTRUMENT_SOURCE_ID2STR = {
    0: "acoustic",
    1: "electronic",
    2: "synthetic"
}
INSTRUMENT_SOURCE_STR2ID = {v: k for k, v in INSTRUMENT_SOURCE_ID2STR.items()}

# instrument_family -> instrument_family_str (0=bass, 1=brass, 2=flute, 3=guitar, ...)
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
INSTRUMENT_FAMILY_STR2ID = {v: k for k, v in INSTRUMENT_FAMILY_ID2STR.items()}

# numeric indices -> qualities_str (0=bright, 1=dark, etc.)
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
QUALITIES_STR2ID = {v: k for k, v in QUALITIES_ID2STR.items()}

INSTRUMENTS_ID2STR = {}

# ------------------------------------------------------------------------------
# JSON Loading and Processing
# ------------------------------------------------------------------------------

def load_json(partition):
    """
    Reads `examples.json` from the unzipped folder:
    e.g. data/training/examples.json
    """
    json_path = DATA_PATH / partition / "examples.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_metadata(json_data):
    """
    Convert certain string fields to their corresponding integer IDs
    for each entry in the entire JSON data. Also build the instrument -> instrument_str map.
    """
    for _, metadata in json_data.items():
        if "instrument_source_str" in metadata:
            s = metadata["instrument_source_str"]
            if s in INSTRUMENT_SOURCE_STR2ID:
                metadata["instrument_source"] = INSTRUMENT_SOURCE_STR2ID[s]
            del metadata["instrument_source_str"]

        if "instrument_family_str" in metadata:
            f = metadata["instrument_family_str"]
            if f in INSTRUMENT_FAMILY_STR2ID:
                metadata["instrument_family"] = INSTRUMENT_FAMILY_STR2ID[f]
            del metadata["instrument_family_str"]

        if "qualities_str" in metadata:
            str_qualities = metadata["qualities_str"]
            numeric_qualities = []
            for q_str in str_qualities:
                if q_str in QUALITIES_STR2ID:
                    numeric_qualities.append(QUALITIES_STR2ID[q_str])
            del metadata["qualities_str"]

        if "instrument" in metadata and "instrument_str" in metadata:
            instrument_id = metadata["instrument"]
            INSTRUMENTS_ID2STR[instrument_id] = metadata["instrument_str"]

        if "note_str" in metadata:
            del metadata["note_str"]
        if "instrument_str" in metadata:
            del metadata["instrument_str"]

    return json_data

# ------------------------------------------------------------------------------
# Load raw waveform (no transform applied)
# ------------------------------------------------------------------------------

def load_raw_waveform(partition, key):
    """
    Loads the raw .wav file for a given key from e.g. data/training/audio/<key>.wav.
    Returns (waveform, sample_rate).
    """
    wav_path = DATA_PATH / partition / "audio" / f"{key}.wav"
    waveform, sr = torchaudio.load(str(wav_path))
    return waveform, sr
