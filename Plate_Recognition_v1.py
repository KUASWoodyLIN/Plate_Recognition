import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
up_path, _ = os.path.split(path)
sys.path.append(up_path)
import numpy as np
from IPython.display import Image

from Plate_Recognition.common.read_dirfile import read_dirfile_name, license_plate

from Plate_Recognition.common.generator import train_generator, valid_generator, characters

from sklearn.cross_validation import train_test_split

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.callbacks import *


DATA_DIR = os.path.join(os.getcwd() + '/data/train/')
ALL_FILES = read_dirfile_name(DATA_DIR)
train_data, valid_data = train_test_split(ALL_FILES, test_size=0.1)

# ALL_LABELS = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'.split()
# id2name = {i: name for i, name in enumerate(ALL_LABELS)}
# name2id = {name: i for i, name in id2name.items()}

# image size
width, height = 320, 240

# Network parameters
conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 32
rnn_size = 512
input_shape = (width, height, 3)



def ctc_lambda_func(args):
  y_pred, labels, input_length, label_length = args
  y_pred = y_pred[:, 2:, :]
  return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



input_data = Input(name='the_input', shape=input_shape, dtype='float32')
inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv1')(input_data)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv2')(inner)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

conv_to_rnn_dims = (width // (pool_size ** 2), (height // (pool_size ** 2)) * conv_filters)
inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

# cuts down input size going into RNN:
inner = Dense(time_dense_size, activation='relu', name='dense1')(inner)

# Two layers of bidirecitonal GRUs
# GRU seems to work as well, if not better than LSTM:
gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

# transforms RNN output to character activations:
inner = Dense(len(characters)+1, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
y_pred = Activation('softmax', name='softmax')(inner)

Model(inputs=input_data, outputs=y_pred).summary()
labels = Input(name='the_labels', shape=[7], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
# if start_epoch > 0:
#     weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
#     model.load_weights(weight_file)

epochs = 50
batch_size = 64
file_name = str(epochs) + '_' + str(batch_size)
callbacks = [
  EarlyStopping(monitor='val_loss',
                patience=5,
                verbose=1,
                min_delta=0.01,
                mode='min'),
  ReduceLROnPlateau(monitor='val_loss',
                    factor=0.1,
                    patience=3,
                    verbose=1,
                    epsilon=0.01,
                    mode='min'),
  ModelCheckpoint(monitor='val_loss',
                  filepath='h5/' + file_name + '.hdf5',
                  save_best_only=True,
                  save_weights_only=True,
                  mode='min'),
  TensorBoard(log_dir='logs/' + file_name)
]

model.fit_generator(generator=train_generator(train_data, batch_size),
                    steps_per_epoch=312,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(valid_data, batch_size),
                    validation_steps=int(np.ceil(len(valid_data)/64))
                    )
