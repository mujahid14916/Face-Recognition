import numpy as np
from time import time
import cv2
from keras.models import load_model
from utils import Utils
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Detection:
    model = load_model(Utils.DETECTION_MODEL_PATH)
    graph = tf.get_default_graph()

    def __init__(self):
        self.win_size = Utils.IMG_WIN_SIZE
        self.step = Utils.IMG_STEP_SIZE
        self.save_path = Utils.DETECTION_STORAGE_PATH

    def reduce_windows(self, array):
        new_array = []
        pos = 0
        offset = 2 * self.step
        a = np.copy(array)
        while pos < len(a):
            c = a[pos]
            if np.all(c != -offset):
                indexes = np.all(np.absolute(a - c) <= offset, axis=1)
                new_array.append(np.mean(a[indexes], axis=0))
                a[indexes] = -offset
            pos += 1
        return np.array(new_array, dtype=np.uint16)

    def detect_face(self, img):
        position = []
        img_slices = []

        for i in range(0, img.shape[0], self.step):
            if i+self.win_size <= img.shape[0]:
                for j in range(0, img.shape[1], self.step):
                    if j+self.win_size <= img.shape[1]:
                        if i == 0 and j == 0:
                            img_slices = np.reshape(img[0:self.win_size, 0:self.win_size, :3], (1, 160, 160, 3))
                            position.append([i, j])
                        else:
                            img_slices = np.concatenate((img_slices,
                                                         np.expand_dims(img[i:i+self.win_size, j:j+self.win_size, :3],
                                                                        axis=0)), axis=0)
                            position.append([i, j])

        with Detection.graph.as_default():
            result = Detection.model.predict_classes(img_slices) if len(img_slices) != 0 else []
        position = np.array(position)
        valid_positions = position[np.argwhere(result.flatten())].reshape((-1, 2))
        valid_positions = self.reduce_windows(valid_positions)
    
        face_array = []
        for k, pos in enumerate(valid_positions):
            i, j = np.ravel(pos)
            face = img[i:i+self.win_size, j:j+self.win_size, :3]
            if k == 0:
                face_array = np.reshape(face, (1, 160, 160, 3))
            else:
                face_array = np.concatenate([face_array, np.expand_dims(face, axis=0)], axis=0)
            cv2.imwrite(self.save_path + str(time()) + '.jpg', cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    
        # for k in valid_positions:
        #     i, j = np.ravel(k)
        #     img[i:i+win_size, j:j+2, :] = 0
        #     img[i:i+2, j:j+win_size, :] = 0
        #     img[i:i+win_size+2, j+win_size:j+win_size+2, :] = 0
        #     img[i+win_size:i+win_size+2, j:j+win_size+2, :] = 0
        #
        # cv2.imwrite('1.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # print("{} faces detected".format(len(face_array)))
    
        print(valid_positions)
        return face_array
