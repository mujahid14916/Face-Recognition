import numpy as np
import os
from PIL import Image, ImageTk
from utils import Utils
from keras.models import load_model
from face_storage import Storage
import tensorflow as tf
# import h5py


class Verification:
    model = load_model(Utils.VERIFICATION_MODEL_PATH)
    graph = tf.get_default_graph()

    def __init__(self):
        self.encodings = []

    def get_names(self, label, name):
        path = Utils.DATABASE_PATH
        if os.path.isfile(path=path):
            db = Storage()
            d_encodings, d_images, d_names = db.read_from_db()
            # with h5py.File(path, 'r') as database:
            #     d_encodings = np.array(database['encodings'])
            #     d_images = np.array(database['images'])
            #     d_names = list(database['names'])
            decoded_names = list(map(Utils.decode_name, d_names))
            res_index = []
            res_value = []
            for i in self.encodings:
                diff = np.linalg.norm(d_encodings - i, axis=1)
                res_index.append(np.argmin(diff))
                res_value.append(np.min(diff))
            print(res_value)
            for i, j in enumerate(res_index):
                if res_value[i] <= Utils.V_THRESHOLD:
                    img = d_images[int(j)]
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    label.imgtk = imgtk
                    label.config(image=imgtk)
                    name.config(text=decoded_names[int(j)])
                    print(decoded_names[int(j)])
        else:
            print("Database Not Found")

    def get_encoding(self, face):
        with Verification.graph.as_default():
            self.encodings = Verification.model.predict(face)
        return self.encodings

    def recognize_face(self, face, str_label, str_name):
        self.get_encoding(face)
        self.get_names(str_label, str_name)
