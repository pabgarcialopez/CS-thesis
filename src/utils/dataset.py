import json
from pathlib import Path
from config import DATA_PATH
import torchaudio
import torch

NUM_INSTRUMENTS = 11

# List of instrument names
INSTRUMENT_ID_2_STR = [
    "bass",         # 0
    "brass",        # 1
    "flute",        # 2
    "guitar",       # 3
    "keyboard",     # 4
    "mallet",       # 5
    "organ",        # 6
    "reed",         # 7
    "string",       # 8
    "synth_lead",   # 9
    "vocal"         # 10
]

# ------------------------------------------------------------------------------
# JSON Loading and Processing
# ------------------------------------------------------------------------------

def load_json(partition):
    """
    Reads `examples.json` from folder:
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

    model_metadata = {}
    for key, metadata in json_data.items():
        instrument_family = metadata["instrument_family"]
        one_hot = [int(instrument_family == i) for i in range(NUM_INSTRUMENTS)]
        model_metadata[key] = {"one_hot_instrument": torch.tensor(one_hot, dtype=torch.float)}

    
    return model_metadata


# ------------------------------------------------------------------------------
# Load raw waveform (no transform applied)
# ------------------------------------------------------------------------------

def load_raw_waveform(partition, key):
    """
    Loads the raw .wav file for a given key from e.g. data/training/audio/<key>.wav.
    Returns (waveform, sample_rate).
    """
    wav_path = DATA_PATH / partition / "audio" / f"{key}.wav"
    waveform, sr = torchaudio.load(wav_path)
    return waveform, sr

# json_data = {
#     "bass_synthetic_068-049-025": {
#         "qualities": [
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 49, 
#         "note": 217499, 
#         "instrument_source_str": "synthetic", 
#         "velocity": 25, 
#         "instrument_str": "bass_synthetic_068", 
#         "instrument": 656, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "dark"
#         ], 
#         "instrument_source": 2, 
#         "note_str": "bass_synthetic_068-049-025", 
#         "instrument_family": 0, 
#         "instrument_family_str": "bass"
#     }, 
#     "keyboard_electronic_001-021-127": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 21, 
#         "note": 299359, 
#         "instrument_source_str": "electronic", 
#         "velocity": 127, 
#         "instrument_str": "keyboard_electronic_001", 
#         "instrument": 40, 
#         "sample_rate": 16000, 
#         "qualities_str": [], 
#         "instrument_source": 1, 
#         "note_str": "keyboard_electronic_001-021-127", 
#         "instrument_family": 4, 
#         "instrument_family_str": "keyboard"
#     }, 
#     "guitar_acoustic_010-066-100": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 66, 
#         "note": 72288, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 100, 
#         "instrument_str": "guitar_acoustic_010", 
#         "instrument": 219, 
#         "sample_rate": 16000, 
#         "qualities_str": [], 
#         "instrument_source": 0, 
#         "note_str": "guitar_acoustic_010-066-100", 
#         "instrument_family": 3, 
#         "instrument_family_str": "guitar"
#     }, 
#     "reed_acoustic_037-068-127": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             1, 
#             0
#         ], 
#         "pitch": 68, 
#         "note": 22259, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 127, 
#         "instrument_str": "reed_acoustic_037", 
#         "instrument": 387, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "reverb"
#         ], 
#         "instrument_source": 0, 
#         "note_str": "reed_acoustic_037-068-127", 
#         "instrument_family": 7, 
#         "instrument_family_str": "reed"
#     }, 
#     "flute_acoustic_002-077-100": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             1, 
#             0
#         ], 
#         "pitch": 77, 
#         "note": 72001, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 100, 
#         "instrument_str": "flute_acoustic_002", 
#         "instrument": 86, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "reverb"
#         ], 
#         "instrument_source": 0, 
#         "note_str": "flute_acoustic_002-077-100", 
#         "instrument_family": 2, 
#         "instrument_family_str": "flute"
#     }, 
#     "string_acoustic_056-047-075": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             1, 
#             0
#         ], 
#         "pitch": 47, 
#         "note": 45336, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 75, 
#         "instrument_str": "string_acoustic_056", 
#         "instrument": 436, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "fast_decay", 
#             "reverb"
#         ], 
#         "instrument_source": 0, 
#         "note_str": "string_acoustic_056-047-075", 
#         "instrument_family": 8, 
#         "instrument_family_str": "string"
#     }, 
#     "vocal_synthetic_003-088-025": {
#         "qualities": [
#             1, 
#             0, 
#             1, 
#             0, 
#             1, 
#             0, 
#             1, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 88, 
#         "note": 22833, 
#         "instrument_source_str": "synthetic", 
#         "velocity": 25, 
#         "instrument_str": "vocal_synthetic_003", 
#         "instrument": 37, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "bright", 
#             "distortion", 
#             "long_release", 
#             "nonlinear_env"
#         ], 
#         "instrument_source": 2, 
#         "note_str": "vocal_synthetic_003-088-025", 
#         "instrument_family": 10, 
#         "instrument_family_str": "vocal"
#     }, 
#     "brass_acoustic_046-101-050": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 101, 
#         "note": 80852, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 50, 
#         "instrument_str": "brass_acoustic_046", 
#         "instrument": 414, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "long_release"
#         ], 
#         "instrument_source": 0, 
#         "note_str": "brass_acoustic_046-101-050", 
#         "instrument_family": 1, 
#         "instrument_family_str": "brass"
#     }, 
#     "guitar_acoustic_014-070-050": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 70, 
#         "note": 19884, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 50, 
#         "instrument_str": "guitar_acoustic_014", 
#         "instrument": 263, 
#         "sample_rate": 16000, 
#         "qualities_str": [], 
#         "instrument_source": 0, 
#         "note_str": "guitar_acoustic_014-070-050", 
#         "instrument_family": 3, 
#         "instrument_family_str": "guitar"
#     }, 
#     "string_acoustic_014-063-050": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             1, 
#             1, 
#             0
#         ], 
#         "pitch": 63, 
#         "note": 83687, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 50, 
#         "instrument_str": "string_acoustic_014", 
#         "instrument": 100, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "fast_decay", 
#             "percussive", 
#             "reverb"
#         ], 
#         "instrument_source": 0, 
#         "note_str": "string_acoustic_014-063-050", 
#         "instrument_family": 8, 
#         "instrument_family_str": "string"
#     }, 
#     "bass_synthetic_033-041-075": {
#         "qualities": [
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 41, 
#         "note": 78888, 
#         "instrument_source_str": "synthetic", 
#         "velocity": 75, 
#         "instrument_str": "bass_synthetic_033", 
#         "instrument": 417, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "dark"
#         ], 
#         "instrument_source": 2, 
#         "note_str": "bass_synthetic_033-041-075", 
#         "instrument_family": 0, 
#         "instrument_family_str": "bass"
#     }, 
#     "keyboard_electronic_001-063-075": {
#         "qualities": [
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 63, 
#         "note": 51390, 
#         "instrument_source_str": "electronic", 
#         "velocity": 75, 
#         "instrument_str": "keyboard_electronic_001", 
#         "instrument": 40, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "dark"
#         ], 
#         "instrument_source": 1, 
#         "note_str": "keyboard_electronic_001-063-075", 
#         "instrument_family": 4, 
#         "instrument_family_str": "keyboard"
#     }, 
#     "keyboard_electronic_098-023-050": {
#         "qualities": [
#             0, 
#             1, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 23, 
#         "note": 234194, 
#         "instrument_source_str": "electronic", 
#         "velocity": 50, 
#         "instrument_str": "keyboard_electronic_098", 
#         "instrument": 905, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "dark", 
#             "distortion"
#         ], 
#         "instrument_source": 1, 
#         "note_str": "keyboard_electronic_098-023-050", 
#         "instrument_family": 4, 
#         "instrument_family_str": "keyboard"
#     }, 
#     "bass_synthetic_068-084-050": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 84, 
#         "note": 244950, 
#         "instrument_source_str": "synthetic", 
#         "velocity": 50, 
#         "instrument_str": "bass_synthetic_068", 
#         "instrument": 656, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "fast_decay"
#         ], 
#         "instrument_source": 2, 
#         "note_str": "bass_synthetic_068-084-050", 
#         "instrument_family": 0, 
#         "instrument_family_str": "bass"
#     }, 
#     "mallet_acoustic_062-066-050": {
#         "qualities": [
#             0, 
#             1, 
#             0, 
#             1, 
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 66, 
#         "note": 125499, 
#         "instrument_source_str": "acoustic", 
#         "velocity": 50, 
#         "instrument_str": "mallet_acoustic_062", 
#         "instrument": 644, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "dark", 
#             "fast_decay", 
#             "multiphonic"
#         ], 
#         "instrument_source": 0, 
#         "note_str": "mallet_acoustic_062-066-050", 
#         "instrument_family": 5, 
#         "instrument_family_str": "mallet"
#     }, 
#     "guitar_electronic_028-086-100": {
#         "qualities": [
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 86, 
#         "note": 46930, 
#         "instrument_source_str": "electronic", 
#         "velocity": 100, 
#         "instrument_str": "guitar_electronic_028", 
#         "instrument": 510, 
#         "sample_rate": 16000, 
#         "qualities_str": [], 
#         "instrument_source": 1, 
#         "note_str": "guitar_electronic_028-086-100", 
#         "instrument_family": 3, 
#         "instrument_family_str": "guitar"
#     }, 
#     "flute_synthetic_000-053-025": {
#         "qualities": [
#             0, 
#             0, 
#             1, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0, 
#             0
#         ], 
#         "pitch": 53, 
#         "note": 37336, 
#         "instrument_source_str": "synthetic", 
#         "velocity": 25, 
#         "instrument_str": "flute_synthetic_000", 
#         "instrument": 82, 
#         "sample_rate": 16000, 
#         "qualities_str": [
#             "distortion"
#         ], 
#         "instrument_source": 2, 
#         "note_str": "flute_synthetic_000-053-025", 
#         "instrument_family": 2, 
#         "instrument_family_str": "flute"
#     },
# }

# print(process_metadata(json_data))

# Result

