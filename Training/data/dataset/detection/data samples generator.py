from scipy.misc import imread, imresize
import cv2
import glob
from random import shuffle
import os
import numpy as np

f_files = glob.glob('face/*.jpg')
nf_files = glob.glob('nonface/*.jpg')
faces = np.zeros((len(f_files), 160, 160, 3), dtype=np.uint8)
non_faces = np.zeros((len(nf_files), 160, 160, 3), dtype=np.uint8)

for i, file in enumerate(f_files):
    faces[i] = (imresize(imread(file), (160, 160, 3)))
for i, file in enumerate(nf_files):
    non_faces[i] = (imresize(imread(file), (160, 160, 3)))

def image_stitch(image_array, row=5, col=5):
    if image_array.ndim != 4:
        raise ValueError("Dimension is not 4")
    if image_array.shape[0] < row*col:
        raise ValueError("Not sufficient images to form the grid of {}x{}".format(row, col))

    grid = image_array[0, :, :, :]
    for i in range(1, col):
        grid = np.concatenate((grid, image_array[i]), axis=1)
    
    
    for i in range(col, row * col, col):
        buffer = image_array[i]
        for j in range(1, col):
            buffer = np.concatenate((buffer, image_array[i+j]), axis=1)
        grid = np.concatenate((grid, buffer), axis=0)

    return grid

cv2.imwrite("face.jpg", cv2.cvtColor(image_stitch(faces, row=58, col=103), cv2.COLOR_BGR2RGB))
cv2.imwrite("nonface.jpg", cv2.cvtColor(image_stitch(non_faces, row=58, col=103), cv2.COLOR_BGR2RGB))
