from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import math
import h5py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

m = 23000
image_size = 160
epoch_size = 40
cost_display_interval = 1
learning_rate = 1e-3
batch_size = 32
no_of_batches = math.ceil(m / batch_size)

data_sets = h5py.File('data/dataset.h5', 'r')
X_data = np.array(list(data_sets['X']))
Y_data = np.array(list(data_sets['Y']), dtype=np.float64).reshape((-1, 1))
X_train = X_data[0:m]
y_train = Y_data[0:m]
X_valid = X_data[m:]
y_valid = Y_data[m:]


model = Sequential()
model.add(Lambda(lambda x: x/255., input_shape=[160, 160, 3]))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(1024, kernel_regularizer=l2(1e-3), activation='relu'))
model.add(Dense(512, kernel_regularizer=l2(1e-3), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


checkpoint = ModelCheckpoint('face_model.h5',
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

logger = TensorBoard(log_dir='face', histogram_freq=5, write_graph=True)

model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_size, verbose=1, shuffle=True,
          validation_data=[X_valid, y_valid], callbacks=[logger, checkpoint])

pred = model.evaluate(X_valid, y_valid)

print("Loss: {}".format(pred[0]))

print("Accuracy: {}".format(pred[1]))


