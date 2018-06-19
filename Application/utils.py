import numpy as np


class Utils:
    DETECTION_MODEL_PATH = 'models/face_detection_model.h5'
    VERIFICATION_MODEL_PATH = 'models/face_verification_model.h5'
    DATABASE_PATH = 'stored_data/mwr1.h5'

    DETECTION_STORAGE_PATH = 'captured/detection/'
    RECOGNITION_STORAGE_PATH = 'captured/recognition/'
    STORAGE_ITERATION = 10

    IMG_WIDTH = 640
    IMG_HEIGHT = 360
    IMG_WIN_SIZE = 160
    IMG_STEP_SIZE = 50

    NET_IMG_WIDTH = 160
    NET_IMG_HEIGHT = 160
    NET_IMG_CHANNEL = 3

    IMG_WIDTH_RESIZED = 435
    IMG_HEIGHT_RESIZED = 240

    NO_OF_ENCODINGS = 250
    MAX_NAME_LENGTH = 25

    V_THRESHOLD = 0.7

    @staticmethod
    def encode_name(str_name):
        ascii_encoded_name = list(map(ord, str_name))
        return np.pad(ascii_encoded_name, (0, Utils.MAX_NAME_LENGTH - len(ascii_encoded_name)), mode='constant')

    @staticmethod
    def decode_name(coded_name):
        ascii_encoded_list = list(coded_name)
        ascii_encoded_name = ascii_encoded_list[:ascii_encoded_list.index(0)]
        return ''.join(list(map(chr, ascii_encoded_name)))
