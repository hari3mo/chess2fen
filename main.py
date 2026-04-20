import numpy as np
import cv2
import os

def load_screenshot(path):
    img = cv2.imread(path)
    return img