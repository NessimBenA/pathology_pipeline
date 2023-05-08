import os

# Environment
WORK_SPACE = os.path.abspath(__file__)
DATA_SPACE = WORK_SPACE
PROJECT_SPACE = WORK_SPACE

# Slides
SLIDES_PATH = os.path.join(DATA_SPACE, 'slides')
SLIDE_EXT = "tif"
PREVIEWS_PATH = os.path.join(DATA_SPACE, 'previews')
PREVIEW_EXT = "png"

# Slices
SLICES_PATH = os.path.join(DATA_SPACE, 'slices')
SLICE_EXT = "png"
SLICE_RESOLUTION = 10 # histopathology magnification

# Segmentation (x1 resolution)
FP_VALUE = 16 # footprint size for morphology
MINIMUM_AREA = 10000 # keep each bigger segmentation
BLUR_SIGMA = 2

# Computation
USED_GPU = 0 # optional
USED_CPU = 2 # need at least 1

# Training
LABELIZATION_PATH = os.path.join(DATA_SPACE, 'train.csv')
TRAIN_PERCENTAGE = 0.7
TRAINING_SET_PATH = os.path.join(DATA_SPACE, 'train')
TESTING_SET_PATH = os.path.join(DATA_SPACE, 'test')
UNIQUE_LABELS = ['CE', 'LAA']

# Harmonization
REF_IMAGE = os.path.join(SLICES_PATH, 'ref_image.png')
HARMONIZED_SLICES_PATH = os.path.join(DATA_SPACE, 'slices_harmonized')