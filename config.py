import os
import torch
# base path of the dataset
# DATASET_PATH = "./tgs-salt-identification-challenge/competition_data/train/"
#DATASET_PATH = "./cell128/train/"
DATASET_PATH = "./data/crack_Rissbilder_for_florian/"
# DATASET_PATH = "./nuclei_images/train"

# DATASET_PATH = "./nuclei512/"

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images_squeeze")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks_squeeze")
NUM_WORKERS =1
# define the test split
TEST_SPLIT = 0.2

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 1

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
TOPO_SIZE = 64 #64
# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 32 #64
# topology parameters
LAMBDA = 0.00001
WITH_TOPO_LOSS = True
EPOCHS_WITHOUT_TOPOLOSS = 10
DIM_BETTI_NUMBER = 0 
ADD_PATCH_EDGE = True
INVERSE_IMAGE = True

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training plot, and testing image paths
#MODEL_PATH = "./result/pretrained_model/unet_128_florian_topo.pth"
# PLOT_PATH = "./result/loss_plot/plot_crack_volker_cell.png"
# TEST_PATHS = "./result/test_path/test_paths_crack_cell.txt"
# LOSS_PATH = "./result/res_loss/loss_crack_cell.npy"