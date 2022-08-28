# import the necessary packages
import os
import time
import cv2
import zipfile
from torch.utils.data import Dataset
from config import *
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms as tfs
from imutils import paths as pas
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.unet_v2 import UNet
from models.fcn import FCN
import numpy as np

from topoloss import getTopoLoss
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]

        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)

        # return a tuple of the image and its mask
        return (image, mask)


def prepare_data():
    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(pas.list_images(IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(pas.list_images(MASK_DATASET_PATH)))

    # partition the data into training and testing splits using 85% of the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths, test_size=TEST_SPLIT, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # write the testing image paths to disk so that we can use then when evaluating/testing our model
    print("[INFO] saving testing image paths...")
    with open(TEST_PATHS, "w") as f:
        f.write("\n".join(testImages))
    # define transformations
    transforms = tfs.Compose(
        [tfs.ToPILImage(), tfs.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)), tfs.ToTensor()])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
                             num_workers=NUM_WORKERS)
    testLoader = DataLoader(testDS, shuffle=False, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
                            num_workers=NUM_WORKERS)

    return trainDS, testDS, trainLoader, testLoader


def train_nn(trainDS, testDS, trainLoader, testLoader, withTopoloss = False):
    # initialize our UNet model
    unet = UNet(n_channels=NUM_CHANNELS,n_classes=NUM_CLASSES).to(DEVICE)
    # unet =FCN(NUM_CLASSES).to(DEVICE)
    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    #This loss combines a `Sigmoid` layer and the `BCELoss` in one single class.
    # This version is more numerically stable

    opt = Adam(unet.parameters(), lr=INIT_LR)

    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // BATCH_SIZE
    testSteps = len(testDS) // BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": [], "topo_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        trainTopoLoss = 0

        # loop over the training set
        for x, y in trainLoader:
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # perform a forward pass and calculate the training loss
            pred = unet(x)
            if not withTopoloss:
                loss = lossFunc(pred, y)
                totalTrainLoss += loss
            else:
                topo_loss = 0
                if (e <= EPOCHS_WITHOUT_TOPOLOSS):
                    loss = lossFunc(pred, y)
                    totalTrainLoss += loss
                else:
                    loss = lossFunc(pred, y)
                    totalTrainLoss += loss
                    for dp in range(pred.shape[0]):
                        topo_loss += getTopoLoss(pred[dp, 0, :, :], y[dp, 0, :, :], topo_size=TOPO_SIZE)
                    loss += LAMBDA * topo_loss
                trainTopoLoss +=topo_loss
            # first, zero out any previously accumulated gradients, then perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
 
            # loop over the validation set
            acc = 0
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))

                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)
                lh = torch.sigmoid(pred)
                acc += (torch.round(lh)==y).sum()
            print("Accuracy:",acc/(len(testLoader)*INPUT_IMAGE_HEIGHT*INPUT_IMAGE_WIDTH))

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        avgTopoLoss = trainTopoLoss / testSteps

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        H["topo_loss"].append(avgTopoLoss)

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print("Train loss: {:.6f}, Test loss: {:.4f}, Topo loss: {:.4f}".format(avgTrainLoss, avgTestLoss,avgTopoLoss))

    # display the total time needed to perform the training
    endTime = time.time()
    np.save(LOSS_PATH, H)
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)

    # serialize the model to disk
    torch.save(unet, MODEL_PATH)


if __name__ == '__main__':
    trainDS, testDS, trainLoader, testLoader = prepare_data()
    train_nn(trainDS, testDS, trainLoader, testLoader,withTopoloss=WITH_TOPO_LOSS)
