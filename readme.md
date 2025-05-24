# DLC-Classifier

This repository contains code for training and evaluating a DeepLabCut-based neural network for mouse behavioral zone detection.

## File Overview

### Notebooks
- **DLC_Compatible_File_Creation.ipynb**  
  Creates DLC-compatible CSV files from preprocessed video data and zone annotations. The commented out steps in this notebook are also where the DLC project is created.

- **DLC_Evaluation.ipynb**  
  Evaluates a trained DLC model on a held-out evaluation set, visualizing predictions and confidence.

- **ZonalClassification.ipynb**  
  Original prototype notebook for exploring zone-wise classification logic. Used during early development.

### Python Scripts
- **eval_set_create.py**  
  Constructs the evaluation dataset by sampling frames and saving metadata needed for DLC evaluation.

- **train_set_create.py**  
  Constructs the training dataset by selecting high-movement or representative frames across zones.

- **train_network.py**  
  Trains the DeepLabCut network using the generated training data.

### Bash Scripts
- **submit_evalset.sh**  
  SLURM submission script for generating evaluation frames in batch on a cluster.

- **submit_trainset.sh**  
  SLURM submission script for creating the training dataset via zone-guided frame sampling.

- **train_network.sh**  
  SLURM submission script for training the DLC network using cluster resources.

- **video_sampling.sh**  
  Shell script that identifies videos with usable zone annotation files and extracts relevant frames.

### Data Files
- **sampled_videos.txt**  
  List of all videos selected for training and evaluation, used to maintain reproducibility and sampling consistency.

### Directories
- **.ipynb_checkpoints/**  
  Automatically generated checkpoint versions of notebooks. These are not used in execution.

---

## Using

1. Run `train_set_create.py` to build a training dataset, assuming that the root folder is a folder that contains data with a .mat file and a .avi video file.
2. Submit the network training job with `submit_trainset.sh`.
3. Train the network using train_network.sh
4. Evaluate results using `eval_set_create.py` followed by `submit_evalset.sh` to collect a set of images to use as evalaution
5. Overlay the neural networks predictions for these evaluation images using `DLC_Evaluation.ipynb`
6. User changes involve choosing thresholds for movement and accuracy, and the directory where images are stored.
