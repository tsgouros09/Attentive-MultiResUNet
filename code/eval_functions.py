import numpy as np
import librosa
import musdb
import stdct
import mir_eval

from scipy.signal import butter, lfilter
from settings import *

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Butterworth Filter for output
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def energy_thresholding(signal, energy_threshold):
    """
    Post processing energy thresholding
    """
    for frame in range(0,signal.shape[1]):
        if (np.sum(np.square(np.abs(signal[:,frame]))) < energy_threshold):
          signal[:,frame] = np.zeros((1,signal.shape[0]))
    return signal

def estimate_and_evaluate(trackCut, model, component, energy_threshold):
  """
  Separation scheme with trained network
  """
  mix_channel = trackCut.audio.T
  temp1 = np.expand_dims(stdct.frameWind_eval(librosa.resample(y=mix_channel[0], orig_sr=ORIGINAL_SAMPLING_RATE, target_sr=TARGET_SAMPLING_RATE), frame=WINDOW_LENGTH, hop_size=HOP_LENGTH), axis=-1)
  temp2 = np.expand_dims(stdct.frameWind_eval(librosa.resample(y=mix_channel[1], orig_sr=ORIGINAL_SAMPLING_RATE, target_sr=TARGET_SAMPLING_RATE), frame=WINDOW_LENGTH, hop_size=HOP_LENGTH), axis=-1)
  dct_mix_spec = np.concatenate((temp1,temp2), axis=-1)
  source = np.zeros_like(dct_mix_spec[:,:,0])

  #Training
  for i in range((dct_mix_spec.shape[1]//TIME_BINS)):
    start=i*TIME_BINS
    end = start + TIME_BINS
    input = np.expand_dims(dct_mix_spec[:MAX_BINS,start:end], axis=0)
    four_stems = model(input, training=False)
    source[:MAX_BINS,start:end] = four_stems.numpy().reshape((MAX_BINS,TIME_BINS))
    if EN_THRES == True:
        source = energy_thresholding(source, energy_threshold)

  remainder = dct_mix_spec.shape[1]-(dct_mix_spec.shape[1]//TIME_BINS)*TIME_BINS
  gt_source = librosa.resample(librosa.to_mono(trackCut.targets[component].audio.T), orig_sr=ORIGINAL_SAMPLING_RATE, target_sr=TARGET_SAMPLING_RATE)
  reconst_source = stdct.inverseWind_eval(source[:,:-remainder], hop_size=HOP_LENGTH)
  length = reconst_source.shape[0]
  gt_source = gt_source[0:length]
  if component == 'vocals':
      cutoff = VOCALS_FILTER_CUTOFF
  elif component == 'bass':
      cutoff = BASS_FILTER_CUTOFF
  else:
      cutoff = OTHER_FILTER_CUTOFF
  if component != 'drums':
      reconst_source = butter_lowpass_filter(reconst_source, cutoff, TARGET_SAMPLING_RATE, 5)
  estimates = {
          'time_bins': reconst_source.shape[0],
          'gt_source': gt_source,
          'source': reconst_source,
          }
  return estimates

def are_sources_silent(sources):
    """
    Checks for source silent parts in songs
    """
    signal = stdct.frameWind_eval(np.squeeze(sources), frame=WINDOW_LENGTH, hop_size=HOP_LENGTH)
    if(np.sum(np.square(signal))<1e-6):
      return True
    else:
      return False
