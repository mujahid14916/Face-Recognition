import numpy as np
from scipy.misc import imresize
import glob
import h5py
import matplotlib.pyplot as plt

face = glob.glob('dataset/detection/face/*.jpg')
nonface = glob.glob('dataset/detection/nonface/*.jpg')

size = 2 * len(face) + 2 * len(nonface)
img_size = 160
X = np.zeros((size, img_size, img_size, 3), dtype=np.int16)
Y = np.zeros(size, dtype=np.int8)
index = 0
print("Positive")
for i in face:
    raw = plt.imread(i)
    img = imresize(plt.imread(i), (img_size, img_size))
    X[index] = img
    Y[index] = 1
    index += 1
    X[index] = np.fliplr(img)
    Y[index] = 1
    index += 1

print("Negative")
for i in nonface:
    img = imresize(plt.imread(i), (img_size, img_size))
    X[index] = img
    Y[index] = 0
    index += 1
    X[index] = np.fliplr(img)
    Y[index] = 0
    index += 1

rand = np.random.permutation(X.shape[0])
X = X[rand]
Y = Y[rand]

file = h5py.File('dataset.h5', 'w')
file.create_dataset('X', data=X, dtype=np.uint8)
file.create_dataset('Y', data=Y, dtype=np.uint8)
file.close()

