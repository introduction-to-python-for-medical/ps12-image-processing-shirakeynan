from PIL import Image
import numpy as np
from scipy.ndimage import convolve
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection  
def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array  
def edge_detection(image_array):
    gray_image = np.mean(image_array, axis=2) 
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  
    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
    edgeX = convolve(gray_image, kernelx)
    edgeY = convolve(gray_image, kernelY)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG  
