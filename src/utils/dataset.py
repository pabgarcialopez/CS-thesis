import io, json, zipfile
from config import DATA_PATH

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


#  numeric indices -> qualities_str (0=bright, 1=dark, 2=distortion, etc.)
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

# This map is filled up dinamically by convert_strings_to_ids
INSTRUMENTS_ID2STR = {}


# json_data has this structure:
# {
#   "bass_synthetic_068-049-025": { ...metadata... },
#   "keyboard_electronic_001-021-127": { ...metadata... },
#   ...
# }

# ------------------------------------------------------------------------------
# Functions for dataset processing
# ------------------------------------------------------------------------------

def load_json(partition):
    """
    Returns the loaded json file examples.com from a specific dataset partition
    """
    zip_path = DATA_PATH / (partition + '.zip')
    json_file_name = f'{partition}/examples.json'
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_file_name, 'r') as json_file:
            with io.TextIOWrapper(json_file, encoding='utf-8') as text_file:
                return json.load(text_file)
            
def get_audio_file(audio_file_name, partition):
    """
    Returns the ready-to-use file to be used by `torchaudio.load(...)`
    """
    zip_path = DATA_PATH / (partition + '.zip')
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(f"{partition}/audio/{audio_file_name}") as audio_file:
            return io.BytesIO(audio_file.read())

def process_metadata(json_data):
    """
    Convert certain string fields to their corresponding integer IDs
    for each entry in the entire JSON data. Also build the instrument -> instrument_str map.

    json_data: the entire dictionary loaded from examples.json
    """
    for _, metadata in json_data.items():
        # If present, convert instrument_source_str -> instrument_source
        if "instrument_source_str" in metadata:
            s = metadata["instrument_source_str"]
            if s in INSTRUMENT_SOURCE_STR2ID:
                metadata["instrument_source"] = INSTRUMENT_SOURCE_STR2ID[s]
            del metadata["instrument_source_str"]

        # If present, convert instrument_family_str -> instrument_family
        if "instrument_family_str" in metadata:
            f = metadata["instrument_family_str"]
            if f in INSTRUMENT_FAMILY_STR2ID:
                metadata["instrument_family"] = INSTRUMENT_FAMILY_STR2ID[f]
            del metadata["instrument_family_str"]

        # If present, convert qualities_str -> qualities
        if "qualities_str" in metadata:
            str_qualities = metadata["qualities_str"]
            numeric_qualities = []
            for q_str in str_qualities:
                if q_str in QUALITIES_STR2ID:
                    numeric_qualities.append(QUALITIES_STR2ID[q_str])
            del metadata["qualities_str"]

        # Build the instrument -> instrument_str map if needed
        if "instrument" in metadata and "instrument_str" in metadata:
            instrument_id = metadata["instrument"]
            INSTRUMENTS_ID2STR[instrument_id] = metadata["instrument_str"]

        # Remove note_str and instrument_str
        if "note_str" in metadata:
            del metadata["note_str"]
        if "instrument_str" in metadata:
            del metadata["instrument_str"]

    return json_data