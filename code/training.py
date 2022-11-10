import numpy as np
import tensorflow as tf
import logging
import os
from tqdm import tqdm
from generate_dataset import *
from model import Attentive_MultiResUNet
from train_functions import *
from settings import *

train_track_indices, train_parts, mus_train = createDataset('train', 'train', DURATION, DIRECTORY_PATH, MUSDB18_PATH, SHUFFLE=True)
valid_track_indices, valid_parts, mus_valid = createDataset('train', 'valid', DURATION, DIRECTORY_PATH, MUSDB18_PATH, SHUFFLE=True)
model = Attentive_MultiResUNet()
optimiser = OPTIMIZER
component = COMPONENT
batch_size = BATCH_SIZE
epochs = EPOCHS
steps_per_train_epoch = STEPS_PER_EPOCH
steps_per_valid_epoch = VALIDATION_STEPS
best_valid_loss = 1
exit_verbose = 5
verbose=0

# Create Checkpoints
if not os.path.exists('ckpts'):
   os.makedirs('ckpts')
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimiser=optimiser, model=model)
latest_checkpoint = tf.train.latest_checkpoint('./ckpts')
if latest_checkpoint:
  ckpt.restore(latest_checkpoint)
manager = tf.train.CheckpointManager(ckpt, './ckpts', max_to_keep=1)

# Save loss in log file
if not os.path.exists('Losses'):
   os.makedirs('Losses')
logging.basicConfig(filename="Losses/losses.txt", level=logging.INFO)


if not os.path.exists('Models'):
   os.makedirs('Models')

# Start training
for i in range(epochs):
    print(f"Epoch {i + 1}/{epochs}")

    # Train steps
    train_batch_loss = np.zeros(steps_per_train_epoch)
    for j in tqdm(range(steps_per_train_epoch)):
        batch_indices, batch_parts = generateNextBatch(j, train_track_indices, train_parts, batch_size)
        batch_input, batch_target, batch_target_time = dataGeneration(batch_indices, batch_parts, mus_train, component)
        train_batch_loss[j] = train_step(batch_input, batch_target, batch_target_time, model, optimiser)
        if j % 5 ==0 :
            print(f"Epoch {i + 1}/{epochs}, Train Loss for batch {j}: {train_batch_loss[j]}")
            save_output(batch_input[0], batch_target[0], model)
        # Checkpoint save
        if j % 50 == 0:
            manager.save()

    # Shuffle data for next epoch
    train_track_indices, train_parts = shuffleData(train_track_indices, train_parts)
    train_loss = np.mean(train_batch_loss)

    # Validation steps
    valid_batch_loss = np.zeros(steps_per_valid_epoch)
    for j in tqdm(range(steps_per_valid_epoch)):
        batch_indices, batch_parts = generateNextBatch(j, valid_track_indices, valid_parts, batch_size)
        batch_input, batch_target, batch_target_time = dataGeneration(batch_indices, batch_parts, mus_valid, component)
        valid_batch_loss[j] = valid_step(batch_input, batch_target, batch_target_time, model)
        if j % 5 ==0 :
            print(f"Epoch {i + 1}/{epochs}, Valid Loss for batch {j}: {valid_batch_loss[j]}")
    valid_track_indices, valid_parts = shuffleData(valid_track_indices, valid_parts)
    valid_loss = np.mean(valid_batch_loss)
    print(f"Mean train loss: {train_loss}")
    print(f"Mean validation loss: {valid_loss}")

    # Save model after first epoch
    if i == 0 and best_valid_loss == 1:
        best_valid_loss = valid_loss
        model.save_weights(f'Models/model_{component}_weights')
    else:
        # Save model after valid loss decrease
        if best_valid_loss > valid_loss:
            print(f"Previous loss was {best_valid_loss} and now we have {valid_loss}. Saving model...")
            best_valid_loss = valid_loss
            model.save_weights(f'Models/model_{component}_weights')
            verbose = 0
        else:
            verbose += 1
            print(f"Verbose: {verbose}")
    logging.info(f"Valid Loss {best_valid_loss}")

    # Exit if no valid loss decreasing after some epochs
    if verbose == exit_verbose:
        break
