import numpy as np
import mir_eval
import musdb
from model import Attentive_MultiResUNet
from eval_functions import estimate_and_evaluate, are_sources_silent
from settings import *

mus_test = musdb.DB(root=MUSDB18_PATH, subsets='test', download=False)
model = Attentive_MultiResUNet()
component = COMPONENT
model.load_weights(TRAINED_MODEL)
if component == 'vocals':
    energy_threshold = VOCALS_ENERGY_THRESHOLD
elif component == 'bass':
    energy_threshold = BASS_ENERGY_THRESHOLD
elif component == 'drums':
    energy_threshold = DRUMS_ENERGY_THRESHOLD
else:
    energy_threshold = OTHER_ENERGY_THRESHOLD

mean_sdr = []
median_sdr = []

for track in mus_test:
  sdr_list = []
  print('Processing... ',track.name)
  print('\n')

  # Evaluate track part with mir_eval after separation
  estimates = estimate_and_evaluate(track, model, component, energy_threshold)
  length = estimates['time_bins']
  sec_length = 16000*EVAL_SEC
  for i in range((length//sec_length)):
    start=i*sec_length
    end = start + sec_length
    ground_truth = np.array([]).reshape((-1,sec_length))
    ground_truth = np.append(ground_truth,np.expand_dims(estimates['gt_source'][start:end], axis=0), axis=0)
    if(are_sources_silent(ground_truth)):
      print('At least one ground truth source is silent. Skipping current segment.')
      continue

    est_sources = np.array([]).reshape((-1,sec_length))
    est_sources = np.append(est_sources,np.expand_dims(estimates['source'][start:end], axis=0), axis=0)
    if(are_sources_silent(est_sources)):
      print('At least one estimated source is silent. Skipping current segment.')
      continue

    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(ground_truth, est_sources, compute_permutation=False)
    print('segment', i)
    print('\n')
    print(f'metrics for {component}')
    print('sdr',sdr)
    print('\n')
    sdr_list.append(sdr)

  sdr_array = np.array(sdr_list)
  sdr_array[sdr_array == np.inf] = np.nan
  sdr_array[sdr_array == -np.inf] = np.nan
  median_sdr.append(np.nanmedian(sdr_array, axis=0))
  mean_sdr.append(np.nanmean(sdr_array, axis=0))

# Print evaluation results
mean_sdr = np.array(mean_sdr)
median_sdr = np.array(median_sdr)
print(f'mean metrics for {component}')
print('mean sdr',np.nanmean(mean_sdr, axis=0))
print('\n')

print(f'median metrics for {component}')
print('median sdr',np.nanmean(median_sdr, axis=0))
print('\n')

print(f'std metrics for {component}')
print('std sdr',np.nanstd(mean_sdr, axis=0))
