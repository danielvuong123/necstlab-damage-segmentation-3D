import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir


relabeled_dataset_directory = "./processed_data/UoS_SR_C1_Relabelled1_1550N_660_1360_1712/annotations/"
original_dataset_directory = "./processed_data/UoS_SR_C1_1550N_660_1360_1712/annotations/"
print("here")
for image in os.listdir(relabeled_dataset_directory):
    print(image)
