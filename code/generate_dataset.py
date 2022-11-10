import numpy as np
import random
import librosa
import musdb
import stdct
import os

from settings import *

def createDataset(SUBSETS, SPLIT, DURATION, DIRECTORY_PATH, MUSDB18_PATH, SHUFFLE=True):
    """
    Creates dataset indices before training
    """
    # Import MUSDB18 dataset
    mus = musdb.DB(root=os.path.join(
        DIRECTORY_PATH, MUSDB18_PATH), subsets=SUBSETS, split=SPLIT)

    start = 0
    track_indices = np.arange(len(mus.tracks))
    len_tracks = len(track_indices)
    parts = []
    new_track_indices = []

    # Divide tracks into segments of certain duration
    for i in range(len(track_indices)):
        parts.append(np.arange(int(mus.tracks[i].duration // DURATION)))
        new_track_indices.append(i * np.ones(len(np.arange(int(mus.tracks[i].duration // DURATION)))))
    parts = np.hstack(parts)
    track_indices = np.hstack(new_track_indices)
    track_indices = track_indices.astype('int32')
    parts = DURATION * parts

    # Shuffle dataset
    random.seed(13)
    seed = random.randint(0,100)
    if SHUFFLE == True:
        np.random.seed(seed)
        np.random.shuffle(track_indices)
        np.random.seed(seed)
        np.random.shuffle(parts)
    return track_indices, parts, mus

def shuffleData(track_indices, parts):
    """
    Shuffles dataset after every epoch
    """
    seed = random.randint(0,100)
    np.random.seed(seed)
    np.random.shuffle(track_indices)
    np.random.seed(seed)
    np.random.shuffle(parts)
    return track_indices, parts

def generateNextBatch(index, track_indices, parts, batch_size):
    """
    Generates next batch
    """
    batch_indices = track_indices[index * batch_size:(index + 1) * batch_size]
    batch_parts = parts[index * batch_size:(index + 1) * batch_size]
    return batch_indices, batch_parts

def getSTDCTSpec(audio, orig_sr, target_sr, win_len, hop_size):
    """
    Generates STDCT spectrogram
    """
    if orig_sr!= target_sr:
        audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)
        stdct_spec = stdct.frameWind(audio, frame=win_len, hop_size=hop_size)
        return stdct_spec

def dataGeneration(batch_indices, batch_parts, mus, component):
    """
    Generate STDCT spectrograms for each batch
    """
    mixtures = []
    source = []

    # Assign track parts from batch indices
    # Input is stereo, output is mono
    for i in range(len(batch_indices)):
        track = mus.tracks[batch_indices[i]]
        track.chunk_start = batch_parts[i]
        track.chunk_duration = DURATION
        mixtures.append(track.audio.T)
        source.append(librosa.to_mono(track.targets[component].audio.T))
    mixtures = np.array(mixtures)
    source = np.array(source)
    mix_power_specs = []
    source_power_specs = []

    # Transforms track parts to STDCT spectrograms
    for i in range(mixtures.shape[0]):
        temp1 = np.expand_dims(getSTDCTSpec(mixtures[i,0], ORIGINAL_SAMPLING_RATE, TARGET_SAMPLING_RATE, WINDOW_LENGTH, HOP_LENGTH), axis=-1)
        temp2 = np.expand_dims(getSTDCTSpec(mixtures[i,1], ORIGINAL_SAMPLING_RATE, TARGET_SAMPLING_RATE, WINDOW_LENGTH, HOP_LENGTH), axis=-1)
        mix_power_spec = np.concatenate((temp1, temp2), axis=-1)
        mix_power_spec = mix_power_spec[:MAX_BINS,:TIME_BINS,:]
        mix_power_specs.append(mix_power_spec)
        source_power_spec = getSTDCTSpec(source[i], ORIGINAL_SAMPLING_RATE, TARGET_SAMPLING_RATE, WINDOW_LENGTH, HOP_LENGTH)
        source_power_spec = source_power_spec[:MAX_BINS,:TIME_BINS]
        source_power_specs.append(source_power_spec)
    mix_power_specs = np.array(mix_power_specs)
    source_power_specs = np.array(source_power_specs)
    input = mix_power_specs
    target = source_power_specs
    return input, target, source
