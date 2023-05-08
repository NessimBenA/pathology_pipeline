# Whole Slide Image Segmentation and Classification

This project was made for the competition Mayo Clinic - STRIP AI Image Classification of Stroke Blood Clot Origin in partnership with [@ThomasPDM](https://github.com/ThomasPDM). It aims to segment whole-slide images (WSIs) into smaller image tiles and then train an image classification model using those tiles. It utilizes OpenSlide for reading WSIs, OpenCV for image processing tasks, skimage for image segmentation, and PyTorch Lightning for model training.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Processing and Segmenting Whole-Slide Images](#processing-and-segmenting-whole-slide-images)
  - [Training an Image Classification Model](#training-an-image-classification-model)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.7 or later
- OpenSlide
- OpenCV
- Scikit-image
- NumPy
- Pillow
- Joblib
- Tqdm
- PyTorch Lightning
- Torchvision
- Cupy (optional for GPU computation)
- Cucim (optional for GPU computation)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/whole-slide-image-segmentation-classification.git
cd whole-slide-image-segmentation-classification
```

2. Create and activate a virtual environment (optional, but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Processing and Segmenting Whole-Slide Images

1. Place your whole-slide images in the `slides` directory.
2. Modify the `config.py` file to match your desired settings.
3. Run the `main.py` script to process and segment the WSIs:

```bash
python main.py
```

The script will process each whole-slide image, segment it into smaller image tiles, and save the tiles in the `slices` directory.

### Training an Image Classification Model

1. Organize your dataset into train and test sets, placing them in the `train` and `val` directories, respectively. Each set should have subdirectories for each class, containing the corresponding image tiles.
2. Modify the `config.py` file to match your desired settings.
3. Run the `main.py` script to train the image classification model:

```bash
python main.py
```

The script will train an image classification model using the provided dataset and save the best model checkpoint and final model weights in the `project_space` directory.

## Configuration

The `config.py` file contains various configuration settings for the project. You can modify these settings to customize the behavior of the script.

- `SLIDES_PATH`: The directory containing the whole-slide images.
- `SLICES_PATH`: The directory where the segmented image tiles will be saved.
- `SLIDE_EXT`: The file extension of the whole-slide images.
- `SLICE_EXT`: The file extension for the saved image tiles.
- `SLICE_RESOLUTION`: The resolution for the segmented image tiles.
- `FP_VALUE`: The footprint value used for segmentation.
- `BLUR_SIGMA`: The sigma value for the Gaussian filter applied before segmentation.
- `USED_GPU`: The number of GPUs to use for computation (set to 0 for CPU-only computation).
- `USED_CPU`: The number of CPU cores to use for parallel processing.
- `TRAINING_SET_PATH`: The directory containing the training set.
- `TESTING_SET_PATH`: The directory containing the testing set.
- `PROJECT_SPACE`: The directory where the trained model and checkpoints will be saved.

## Acknowledgements

This project uses the following open-source libraries and frameworks:

OpenSlide: A C library that provides a simple interface for reading whole-slide images.
OpenCV: A powerful open-source computer vision library.
Scikit-image: A collection of algorithms for image processing in Python.
PyTorch Lightning: A lightweight PyTorch wrapper for high-performance AI research.
Special thanks to the developers and maintainers of these libraries and frameworks for their invaluable contributions to the scientific and AI research communities.