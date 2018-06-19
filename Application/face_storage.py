import h5py
import numpy as np
import os
from utils import Utils


class Storage:
    def __init__(self, verification_obj=None):
        self.path = Utils.DATABASE_PATH
        self.v_obj = verification_obj

    def read_from_db(self):
        with h5py.File(self.path, 'r') as database:
            d_encodings = np.array(database['encodings'])
            d_images = np.array(database['images'])
            d_names = list(database['names'])
        return d_encodings, d_images, d_names

    def store_face(self, img, name):
        img = np.expand_dims(img, axis=0)
        encoding = self.v_obj.get_encoding(img)
        name_encoded = [Utils.encode_name(name)]
        if os.path.isfile(path=self.path):
            with h5py.File(self.path, 'a') as file:
                size = 1
                file['images'].resize((file['images'].shape[0] + size), axis=0)
                file['images'][-size:] = img
                file['encodings'].resize((file['encodings'].shape[0] + size), axis=0)
                file['encodings'][-size:] = encoding
                file['names'].resize((file['names'].shape[0] + size), axis=0)
                file['names'][-size:] = name_encoded
        else:
            with h5py.File(self.path, 'w') as file:
                file.create_dataset('images', dtype=np.uint8, data=img,
                                    maxshape=(None, Utils.NET_IMG_HEIGHT, Utils.NET_IMG_WIDTH, Utils.NET_IMG_CHANNEL))
                file.create_dataset('encodings', dtype=np.float32, data=encoding,
                                    maxshape=(None, Utils.NO_OF_ENCODINGS))
                file.create_dataset('names', dtype=np.uint16, data=name_encoded,
                                    maxshape=(None, Utils.MAX_NAME_LENGTH))
        print("Data stored Successfully")
