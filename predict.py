# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from config import *


def prepare_plot(pathImage, origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)

    # set the titles of the subplots
    ax[0].set_title("Image:" + pathImage.split(os.path.sep)[-1])
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    #figure.show()
    plt.show()
    #save as pdf
    # plt.imsave('origMask.pdf', origMask)
    # plt.imsave('predMask.pdf', predMask)



def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
        orig = image.copy()

        # find the filename and generate the path to ground truth mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(MASK_DATASET_PATH, filename)
        # load the ground-truth segmentation mask in grayscale mode and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_HEIGHT))

        # make the channel axis to be the leading one, add a batch dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

        # filter out the weak predictions and convert them to integers
        predMask = (predMask > THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        # prepare a plot for visualization
        prepare_plot(imagePath,orig, gtMask, predMask)


def main(compare=False):
    # load the image paths in our testing file and randomly select 10 image paths
    print("[INFO] loading up test image paths...")
    # imagePaths = open(TEST_PATHS).read().strip().split("\n")
    # imagePaths = np.random.choice(imagePaths, size=5,replace=False)

    imagePaths = [files for root,dirs,files in os.walk(IMAGE_DATASET_PATH)]
    # imagePaths =[["25_02_01.png"]]
    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    unet = torch.load('./result/pretrained_model/unet_128_florian_raw.pth',map_location=torch.device('cpu')).to(DEVICE)
    # iterate over the randomly selected test image paths
    for path in imagePaths[0][-100:]:
        path = IMAGE_DATASET_PATH+"\\" + path
        print(path)
        # make predictions and visualize the results
        make_predictions(unet, path)

        if compare:
            MODEL_PATH_TOPO = "./result/pretrained_model/unet_128_florian_topo.pth"
            unet_topo = torch.load(MODEL_PATH_TOPO, map_location=torch.device('cpu')).to(DEVICE)
            make_predictions(unet_topo,path)


if __name__ == '__main__':
    main(compare=True)