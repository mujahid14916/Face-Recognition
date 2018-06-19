from keras import backend as K
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input, Lambda
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.regularizers import l2
from callback import SaveModel
import tensorflow as tf
import keras
import math
import h5py
import numpy as np
from scipy.misc import imread, imresize
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def triplet_loss(_, y_pred):
    a_encoding, p_encoding, n_encoding = y_pred[:, :250], y_pred[:, 250:500], y_pred[:, 500:]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(a_encoding, p_encoding)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(a_encoding, n_encoding)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), 0.2)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


def build_model(input_shape):
    inputs = Input(input_shape)
    x = Lambda(lambda p: p/255., name='input_layer')(inputs)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_1')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_2')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_3')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_4')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1500, activation='relu', name='dense1', kernel_regularizer=l2(1e-3))(x)
    x = Dense(500, activation='relu', name='dense2', kernel_regularizer=l2(1e-3))(x)
    x = Dense(250, name='dense3')(x)
    x = Lambda(lambda p: K.l2_normalize(p, axis=1), name='output_layer')(x)
    network = Model(inputs=inputs, outputs=x)
    return network


data_sets = h5py.File('data/triplets.h5', 'r')
anchor = np.array(list(data_sets['anchor']))
positive = np.array(list(data_sets['positive']))
negative = np.array(list(data_sets['negative']))


m = anchor.shape[0]
image_size = 160
epoch_size = 10
cost_display_interval = 1
learning_rate = 1e-3
batch_size = 32


r = np.random.permutation(anchor.shape[0])
anchor = anchor[r]
positive = positive[r]
negative = negative[r]

X_anchor = anchor[:m]
X_positive = positive[:m]
X_negative = negative[:m]
y = np.zeros(m)

shape = (160, 160, 3)

model = build_model(shape)
model.summary()

anchor_input = Input((160, 160, 3), name='anchor_input')
positive_input = Input((160, 160, 3), name='positive_input')
negative_input = Input((160, 160, 3), name='negative_input')

anchor_encoding = model(anchor_input)
positive_encoding = model(positive_input)
negative_encoding = model(negative_input)

out = keras.layers.concatenate([anchor_encoding, positive_encoding, negative_encoding], axis=1)

a = imresize(imread('test/similarity/2.jpg'), (160, 160))
b = imresize(imread('test/similarity/6.jpg'), (160, 160))
c = imresize(imread('test/similarity/7.jpg'), (160, 160))


logs = TensorBoard(log_dir='logs/similarity', write_graph=True)
save_model = SaveModel()
save_model.set_parameters(model=model, period=15)
save_model.set_test_images(a, b, c)

siamese_model = Model([anchor_input, positive_input, negative_input], out)

siamese_model.compile(optimizer=Adam(), loss=triplet_loss)
siamese_model.fit([X_anchor, X_positive, X_negative], y, batch_size=16, epochs=180, verbose=1, callbacks=[logs, save_model])


model.save('siamese_v2.h5', include_optimizer=False)




