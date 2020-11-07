import os
import sys
from collections import defaultdict

# Model
DEFAULT_CODE_SIZE = 512
N_CRITIC = 1
INIT_SIZE = 8
NFD_CHANNELS = [16, 32, 64, 128, 256, 512, 512, 512, 512, 512]
MIXING_NUM = 2
MAX_LAYERS_NUM = 9  # 4 - 1024

# Learning
BATCH_SIZE = defaultdict(lambda: 32, {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32})
LR = defaultdict(lambda: 0.001, {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003})
MAX_ITERATIONS = 3_000_000
NUM_WORKERS = 0

# Dataset
NUM_WORKERS = 0
IMAGE_EXTENSIONS_LOWERCASE = {'jpg', 'png', 'jpeg'}
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS_LOWERCASE.union({f.upper() for f in IMAGE_EXTENSIONS_LOWERCASE})
VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})

# Visualization
SAMPLE_SIZE = (1080, 1920)
IMAGE_N_FRAMES = 1
VIDEO_N_FRAMES = 100
FPS = 30
TRUNCATION_PSI = 0.7
STYLE_CHANGE_COEF = 1.
VELOCITY = 1.
HORIZON_LINE = 0.5

# Logging
SAVE_FREQUENCY = 5000
SAMPLE_FREQUENCY = 5000
LOG_LOSS_FREQUENCY = 100

# Paths
RESULT_DIR = os.environ.get('DEEP_LANDSCAPE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.realpath(__file__)),'results'))
CHECKPOINT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
SAMPLE_DIR = os.path.join(RESULT_DIR, 'samples')
LOG_DIR = os.path.join(RESULT_DIR, 'logs')
TB_DIR = os.path.join(RESULT_DIR, 'tb_logs')
GEN_DIR = os.path.join(RESULT_DIR, 'generated')
# paths for encoding and reenactment
ENCODER_TRAIN_DIR = os.path.join(RESULT_DIR, 'encoders')
