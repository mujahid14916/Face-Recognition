import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

file = h5py.File('dataset.h5', 'r')

print(type(file))

face = np.array(file['X'])
face = np.array(face)
rand = np.random.permutation(23896)
face = face[rand]
print(face.shape)

# print(list(file['list_classes']))


def image_stitch(image_array, row=5, col=5):
    if image_array.ndim != 4:
        raise ValueError("Dimension is not 4")
    if image_array.shape[0] < row*col:
        raise ValueError("Not sufficient images to form the grid of {}x{}".format(row, col))

    grid = image_array[0, :, :, :]
    # print(0, end=' ')
    for i in range(1, col):
        # print(i, end=' ')
        grid = np.concatenate((grid, image_array[i]), axis=1)
    
    # print()
    
    for i in range(col, row * col, col):
        buffer = image_array[i]
        # print(i, end=' ')
        for j in range(1, col):
            # print(i+j, end=' ')
            buffer = np.concatenate((buffer, image_array[i+j]), axis=1)
        # print()
        grid = np.concatenate((grid, buffer), axis=0)

    return grid


cv2.imwrite("Image.jpg", cv2.cvtColor(image_stitch(face, row=116, col=206), cv2.COLOR_BGR2RGB))


