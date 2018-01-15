import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
up_path, _ = os.path.split(path)
sys.path.append(up_path)
import numpy as np

from Plate_Recognition.common.characters import *
from Plate_Recognition.common.read_dirfile import read_dirfile_name, license_plate
from Plate_Recognition.common.generator import train_generator, valid_generator
from sklearn.cross_validation import train_test_split

from keras import backend as K
from keras.layers.merge import add, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda, Dropout, merge
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import *

DATA_DIR = os.path.join(os.getcwd() + '/data/train/')
ALL_FILES = read_dirfile_name(DATA_DIR)
train_data, valid_data = train_test_split(ALL_FILES, test_size=0.1)
OUTPUT_DIR = 'image_ocr'

# image size
width, height = 320, 240

epoch = 200
batch_size = 64

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
x = input_data
for i in range(3):
  x = Conv2D(32, (3, 3), activation='relu')(x)
  x = Conv2D(32, (3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)
gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             kernel_initializer='he_normal', name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])
x = Dropout(0.25)(x)
x = Dense(len(characters)+1, kernel_initializer='he_normal', activation='softmax')(x)
base_model = Model(inputs=input_data, outputs=x)

labels = Input(name='the_labels', shape=[7], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])


sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)  #optimizer='adadelta'


def evaluate(model, batch_num=5):
  batch_acc = 0
  generator = valid_generator(valid_data)
  for i in range(batch_num):
    inputs, output = next(generator)
    y_pred = base_model.predict(inputs['the_input'])
    shape = y_pred[:, 2:, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :4]
    print('output', out)
    if out.shape[1] == 7:
      batch_acc += ((inputs['the_labels'] == out).sum(axis=1) == 7).mean()
  return batch_acc / batch_num


class Evaluate(Callback):
  def __init__(self):
    self.accs = []

  def on_epoch_end(self, epoch, logs=None):
    acc = evaluate(base_model) * 100
    self.accs.append(acc)
    print()
    print('acc: %f%%' % acc)


evaluator = Evaluate()


file_name = str(epoch) + '_' + str(batch_size)
model.fit_generator(generator=train_generator(train_data, batch_size),
                    steps_per_epoch=5,#int(np.ceil(len(train_data)/64)),
                    epochs=epoch,
                    verbose=1,
                    callbacks=[EarlyStopping(patience=10), TensorBoard(log_dir='logs/' + file_name), evaluator],
                    validation_data=valid_generator(valid_data, batch_size),
                    validation_steps=int(np.ceil(len(valid_data)/64)),
                    )
model.save('h5/' + file_name + '.h5')
