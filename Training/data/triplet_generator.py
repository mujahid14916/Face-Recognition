import glob
from imageio import imread
from scipy.misc import imresize
import h5py
import numpy as np
import os
import random as rand
import time

root = 'dataset/similarity'
sub_dir = os.listdir(root)

length = []
final_list = []

start = time.time()
for person_dir in sub_dir:
    files = glob.glob(root + '/' + person_dir + '/*.jpg')
    person_list = []
    for f in files:
        person_list.append(imresize(imread(f), (160, 160)))
    length.append(len(person_list))
    final_list.append(person_list)
print("Time take to load images: {:3f} sec".format(time.time() - start))

start = time.time()
print(len(final_list))
img_mat = np.array(final_list[0])
for i in range(1, len(final_list)):
    img_mat = np.concatenate((img_mat, final_list[i]), axis=0)

length = np.array(length)
cum_length = np.pad(np.cumsum(length), (1, 0), 'constant', constant_values=0)

anchor = []
positive = []
negative = []

for i in range(len(length)):
    anc = img_mat[cum_length[i]:cum_length[i+1]]
    r = np.random.permutation(length[i])
    while np.any(np.equal(np.arange(length[i]), r)):
        r = np.random.permutation(length[i])
    pos = anc[r]
    neg_list = np.concatenate((img_mat[:cum_length[i]], img_mat[cum_length[i+1]:]), axis=0)
    neg = []
    for j in range(length[i]):
        neg.append(rand.choice(neg_list))
    neg = np.array(neg)
    if i == 0:
        anchor = anc
        positive = pos
        negative = neg
    else:
        anchor = np.concatenate((anchor, anc), axis=0)
        positive = np.concatenate((positive, pos), axis=0)
        negative = np.concatenate((negative, neg), axis=0)

    anchor = np.concatenate((anchor, np.flip(anc, axis=2)), axis=0)
    positive = np.concatenate((positive, np.flip(pos, axis=2)), axis=0)
    negative = np.concatenate((negative, np.flip(neg, axis=2)), axis=0)

print("Time take to process images: {:3f} sec".format(time.time() - start))
file = h5py.File('triplets.h5', 'w')
file.create_dataset('anchor', dtype=np.uint8, data=anchor)
file.create_dataset('positive', dtype=np.uint8, data=positive)
file.create_dataset('negative', dtype=np.uint8, data=negative)
file.close()
print(anchor.shape, positive.shape, negative.shape)
