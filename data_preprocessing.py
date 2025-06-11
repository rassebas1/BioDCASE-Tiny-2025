from functools import partial
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from tqdm.dask import TqdmCallback
import pyarrow as pa

from config import load_config, Config
from paths import CLIPS_DIR, PREPROC_PRQ_PATH, EVAL_DIR, EVAL_PREPROC_PRQ_PATH

_PREPROC_DASK_BATCH_SIZE = 100

SPLIT_FOLDER_TO_SPLIT = {
    "Training_Set": "train", 
    "Validation_Set": "validation", 
    "Test_Set": "test",
    "Evaluation_Set": "evaluation"  
}
def extract_loudest_slice(audio_array, sample_rate, audio_slice_duration_ms):
    """Find the max of the audio, then return a slice of duration audio_slice_duration_ms centred on the max.
    Also deal with boundary conditions.

    :param audio_slice_duration_ms:
    :return: slice with duration audio_slice_duration_ms
    """
    slice_n_samples = int(audio_slice_duration_ms / 1000 * sample_rate)
    audio_n_samples = audio_array.shape[0]
    left_edge = slice_n_samples // 2
    right_edge = slice_n_samples - left_edge

    max_index = np.argmax(audio_array)
    start_index = max(max_index - left_edge, 0)
    end_index = min(max_index + right_edge, audio_n_samples)

    # Handle edge cases where the slice would go beyond bounds
    if end_index - start_index < slice_n_samples:
        if start_index == 0:  # Cut at left edge
            end_index = slice_n_samples
        else:  # Cut at right edge
            start_index = audio_n_samples - slice_n_samples
            end_index = audio_n_samples
    return audio_array[start_index:end_index]


def apply_audio_augmentations(audio_array, sample_rate, config):
    """Apply random audio augmentations"""
    import random
    
    # # Time stretching (speed change without pitch change)
    # if random.random() < 0.3:  # 30% chance
    #     stretch_factor = random.uniform(0.8, 1.2)
    #     audio_array = librosa.effects.time_stretch(audio_array.astype(float), rate=stretch_factor)
    
    # Pitch shifting
    if random.random() < 0.3:
        n_steps = random.uniform(-2, 2)  # +/- 2 semitones
        audio_array = librosa.effects.pitch_shift(audio_array.astype(float), sr=sample_rate, n_steps=n_steps)
    
    # Add noise
    if random.random() < 0.4:
        noise_factor = random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_factor, audio_array.shape)
        audio_array = audio_array + noise
    
    # Volume scaling
    # if random.random() < 0.5:
    #     volume_factor = random.uniform(0.7, 1.3)
    #     audio_array = audio_array * volume_factor
    
    # Clip to prevent overflow
    #audio_array = np.clip(audio_array, -1.0, 1.0)
    return audio_array

def run_preprocessing(config: Config, evaluation_mode=False, augment_training=False):
    def do_batch(batch: Iterable[Path]):
        slice_data = []
        for path in batch:
            audio_array, sample_rate = librosa.load(path, sr=config.data_preprocessing.sample_rate, mono=True)
            
            if evaluation_mode:
                # Evaluation data: no augmentation, no labels
                audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
                slice_orig = extract_loudest_slice(audio_array_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
                slice_data.append({
                    "data": slice_orig,
                    "path": str(path),
                    "label": None,
                    "split": "evaluation",
                    "sample_rate": sample_rate
                })
            else:
                # Development data
                is_training = path.parents[1].name == "Training_Set"
                label = path.parent.name == "Yellowhammer"
                split = SPLIT_FOLDER_TO_SPLIT[path.parents[1].name]
                
                # Always add original version
                audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
                slice_orig = extract_loudest_slice(audio_array_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
                slice_data.append({
                    "data": slice_orig,
                    "path": str(path),
                    "label": label,
                    "split": split,
                    "sample_rate": sample_rate
                })
                
                # Add augmented version only for training data
                if augment_training and is_training:
                    aug_audio = apply_audio_augmentations(audio_array, sample_rate, config)
                    aug_audio_int16 = (aug_audio * np.iinfo(np.int16).max).astype(np.int16)
                    slice_aug = extract_loudest_slice(aug_audio_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
                    slice_data.append({
                        "data": slice_aug,
                        "path": str(path),  # Same path as original
                        "label": label,     # Same label as original
                        "split": split,     # Same split as original
                        "sample_rate": sample_rate
                    })
        
        return pd.DataFrame(slice_data)

    # Choose source directory and output path based on mode
    source_dir = EVAL_DIR if evaluation_mode else CLIPS_DIR
    output_path = EVAL_PREPROC_PRQ_PATH if evaluation_mode else PREPROC_PRQ_PATH
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    clips = list(source_dir.rglob("*.wav"))

    batches = []
    total_batches = (len(clips) + _PREPROC_DASK_BATCH_SIZE - 1) // _PREPROC_DASK_BATCH_SIZE
    for batch_idx in range(total_batches):
        start_idx = batch_idx * _PREPROC_DASK_BATCH_SIZE
        end_idx = min(start_idx + _PREPROC_DASK_BATCH_SIZE, len(clips))
        batches.append(clips[start_idx:end_idx])

    desc = f"Preprocessing {'evaluation' if evaluation_mode else 'development'} clips"
    if augment_training and not evaluation_mode:
        desc += " (with augmentation)"
    
    with TqdmCallback(desc=desc):
        dask.config.set({"dataframe.convert-string": False})
        all_data: dd.DataFrame = dd.from_map(
            do_batch, batches, meta=pd.DataFrame({
                "data": pd.Series([], dtype=object),
                "path": pd.Series(dtype="string"),
                "split": pd.Series(dtype="string"),
                "label": pd.Series(dtype="float64"),
                "sample_rate": pd.Series(dtype="int32")
            })
        )
        all_data.to_parquet(
            output_path,
            write_index=False,
            schema={'data': pa.list_(pa.int16())}
        )

def run_preprocessing(config: Config, evaluation=False, eval_path=None, augment_training=False):
    def do_batch(batch: Iterable[Path]):
        slice_data = []
        for path in batch:
            audio_array, sample_rate = librosa.load(path, sr=config.data_preprocessing.sample_rate, mono=True)
            audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
            slice_orig = extract_loudest_slice(audio_array_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
            if evaluation:
                # Evaluation data: no augmentation, no labels
                audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
                slice_orig = extract_loudest_slice(audio_array_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
                slice_data.append({
                    "data": slice_orig,
                    "path": str(path),
                    "label": None,
                    "split": "evaluation",
                    "sample_rate": sample_rate
                })
            else:
                # Development data
                is_training = path.parents[1].name == "Training_Set"
                label = path.parent.name == "Yellowhammer"
                split = SPLIT_FOLDER_TO_SPLIT[path.parents[1].name]
                
                # Always add original version
                audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
                slice_orig = extract_loudest_slice(audio_array_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
                slice_data.append({
                    "data": slice_orig,
                    "path": str(path),
                    "label": label,
                    "split": split,
                    "sample_rate": sample_rate
                })
                
                # Add augmented version only for training data
                if augment_training and is_training:
                    aug_audio = apply_audio_augmentations(audio_array, sample_rate, config)
                    aug_audio_int16 = (aug_audio * np.iinfo(np.int16).max).astype(np.int16)
                    slice_aug = extract_loudest_slice(aug_audio_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
                    slice_data.append({
                        "data": slice_aug,
                        "path": str(path),  # Same path as original
                        "label": label,     # Same label as original
                        "split": split,     # Same split as original
                        "sample_rate": sample_rate
                    })
            
            # slice_data.extend([{
            #     "data": slice,
            #     "path": str(path),
            #     "label": label,
            #     "split": split,
            #     "sample_rate": sample_rate
            # }])
        return pd.DataFrame(slice_data)

    # Choose source directory and output path based on mode
    source_dir = EVAL_DIR if evaluation else CLIPS_DIR
    output_path = EVAL_PREPROC_PRQ_PATH if evaluation else PREPROC_PRQ_PATH

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    clips = list(source_dir.rglob("*.wav"))
    
    batches = []
    total_batches = (len(clips) + _PREPROC_DASK_BATCH_SIZE - 1) // _PREPROC_DASK_BATCH_SIZE
    for batch_idx in range(total_batches):
        start_idx = batch_idx * _PREPROC_DASK_BATCH_SIZE
        end_idx = min(start_idx + _PREPROC_DASK_BATCH_SIZE, len(clips))
        batches.append(clips[start_idx:end_idx])
    
    with TqdmCallback(desc="Preprocessing clips in batches"):
        dask.config.set({"dataframe.convert-string": False})
        all_data: dd.DataFrame = dd.from_map(
            do_batch, batches, meta=pd.DataFrame({
                "data": pd.Series([], dtype=object),
                "path": pd.Series(dtype="string"),
                "split": pd.Series(dtype="string"),
                "label": pd.Series(dtype="int32"),
                "sample_rate": pd.Series(dtype="int32")
            })
        )
        all_data.to_parquet(
            output_path,
            write_index=False,
            schema={'data': pa.list_(pa.int16())}  # make sure array is serialized correctly
        )


if __name__ == "__main__":
    config = load_config()
    run_preprocessing(config)
