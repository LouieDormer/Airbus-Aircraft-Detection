import pandas as pd
import numpy as np
import cv2
import os
import re
import ast

from PIL import Image,ImageDraw

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50, ResNet50_Weights

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Function to convert a string into a Python object
def f(x):
    return ast.literal_eval(x.rstrip('\r\n'))

# Function to compute the bounding box of a given geometry
def getBounds(geometry):
    try:
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        ymin = np.min(arr[1])
        xmax = np.max(arr[0])
        ymax = np.max(arr[1])
        return (xmin, ymin, xmax, ymax)
    except:
        return np.nan # Need to return a value in case of error

# Functions to compute the areas of the bounding box
def getWidth(bounds):
    try:
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(xmax - xmin)
    except:
        return np.nan


def getHeight(bounds):
    try:
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(ymax - ymin)
    except:
        return np.nan # Need to return a value in case of error

# Functions to extract the absolute values of the x and y coordinates of the bounding box
def getX(bounds):
    try:
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(xmin)
    except:
        return np.nan

def getY(bounds):
    try:
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(ymin)
    except:
        return np.nan # Need to return a value in case of error


class Averager:
    """
   To be used for tracking loss and accuracy over multiple iterations of the training loop.

    Attributes:
        current_total (float): The cumulative sum of all values added so far.
        iterations (float): The number of values added so far.
    """

    def __init__(self):
        """
        Initializes the Averager with a total of 0 and no iterations.
        """
        self.current_total = 0.0  # Initialize the total sum to 0.
        self.iterations = 0.0    # Initialize the iteration count to 0.

    def send(self, value):
        """
        Adds a new value to the running total and increments the iteration count.

        Args:
            value (float): The new value to include in the average.
        """
        self.current_total += value  # Add the value to the cumulative total.
        self.iterations += 1         # Increment the iteration count.

    @property
    def value(self):
        """
        Calculates and returns the current average.

        Returns:
            float: The current average of all values sent so far.
                   Returns 0 if no values have been added.
        """
        if self.iterations == 0:  # Avoid division by zero.
            return 0
        else:
            return 1.0 * self.current_total / self.iterations  # Calculate the average.

    def reset(self):
        """
        Resets the Averager, clearing the total and iteration count.
        """
        self.current_total = 0.0  # Reset the total sum to 0.
        self.iterations = 0.0    # Reset the iteration count to 0.
