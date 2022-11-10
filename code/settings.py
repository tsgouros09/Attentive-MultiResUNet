from tensorflow.keras.optimizers import Adam
import os

# DATASET SETTINGS
ORIGINAL_SAMPLING_RATE = 44100                              # Setting the original rate(in Hz)
TARGET_SAMPLING_RATE = 16000                                # Setting the downsampling rate(in Hz)
DURATION = 4.0                                              # Setting the duration of sample excerpt (in seconds)
WINDOW_LENGTH = 2048                                        # Setting the window size for STDCT
HOP_LENGTH = 0.125                                          # Setting the hop length percentage parameter for STDCT
FREQUENCY_BINS = WINDOW_LENGTH                              # Setting the frequency bins deriving from STDCT
MAX_BINS = 2048
TIME_BINS = 128                                             # Setting the time bins deriving from MDCT and the current settings
EVAL_SEC = 1                                                # Seconds for the evaluation
COMPONENT = 'vocals'                                        # Setting for the target stem (vocals, bass, drums, other)

# NETWORK SETTINGS
BATCH_SIZE = 5                                              # Setting the batch size
LEARNING_RATE = 5e-4                                        # Setting the learning rate
OPTIMIZER = Adam(LEARNING_RATE)                             # Setting the optimiser
ALPHA = 1.72                                                # Setting the Attentive MultiResUNet's scaler coefficient
EPOCHS = 40                                                 # Setting the maximum number of epochs
STEPS_PER_EPOCH =4750 // BATCH_SIZE                         # Setting the steps per epoch (This is for 4s duration)
VALIDATION_STEPS = 917 // BATCH_SIZE                        # Setting the validation steps
SHUFFLE = True

# DIRECTORIES
DIRECTORY_PATH = os.path.dirname(os.path.realpath(__file__))
MUSDB18_PATH = '.../musdb18/'                               # Set MUSDB18 folder path here
TRAINED_MODEL = 'Models/model_name'                         # Set path for trained model

# EVALUATION SETTINGS
VOCALS_ENERGY_THRESHOLD = 1e-3
BASS_ENERGY_THRESHOLD = 1e-3
DRUMS_ENERGY_THRESHOLD = 1e-3
OTHER_ENERGY_THRESHOLD = 1e-4

VOCALS_FILTER_CUTOFF = 5000
BASS_FILTER_CUTOFF = 1000
OTHER_FILTER_CUTOFF = 5000
