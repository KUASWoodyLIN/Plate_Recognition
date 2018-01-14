import string

from Plate_Recognition.common.read_dirfile import read_dirfile_name, license_plate, license_plate_single
from Plate_Recognition.common.read_xml import read_location

import cv2
import numpy as np
from keras.utils.np_utils import to_categorical


ALL_LABELS = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'.split()
characters = string.digits + string.ascii_uppercase
id2name = {i: name for i, name in enumerate(characters)}
name2id = {name: i for i, name in id2name.items()}
width, height, n_class = 320, 240, len(characters)
absolute_max_string_len = 7


def train_generator(train_data, batch_size=64):
  # input(X_data, labels, input_length, label_length)
  # X_data = [batch_size, width, height, grad or RGB]
  # labels = [batch_size, labels_length]
  # input_length = [batch_size, CNN_width_output]
  # label_length = [batch_size, label_length]
  # output(loss_out)
  while True:
    for start in range(0, len(train_data), batch_size):
      end = min(start + batch_size, len(train_data))
      size = batch_size if end < len(train_data) else len(train_data) - start

      X_data = np.ones([size, width, height, 3])
      labels = np.full([size, absolute_max_string_len], -1)
      input_length = np.zeros([size, 1])
      label_length = np.zeros([size, 1])
      source_str = []
      i_train_batch = train_data[start:end]

      for i, file in enumerate(i_train_batch):
        X_data[i] = np.array(cv2.imread(file+'.jpg'), dtype=np.float32).transpose(1, 0, 2) / 255
        input_length[i] = width // 2**3 - 2
        license_plate = license_plate_single(file)
        label_length[i] = len(license_plate)
        source_str.append(license_plate)
        for j, key in enumerate(license_plate):
          labels[i, j] = name2id[key]

      inputs = {'the_input': np.array(X_data),
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                'source_str': source_str  # used for visualization only
                }
      outputs = {'ctc': np.zeros([batch_size])}  # dummy data for dummy loss function
    yield inputs, outputs


def valid_generator(valid_data, batch_size=64):
  # input(X_data, labels, input_length, label_length)
  # X_data = [batch_size, width, height, grad or RGB]
  # labels = [batch_size, labels_length]
  # input_length = [batch_size, CNN_width_output]
  # label_length = [batch_size, label_length]
  # output(loss_out)
  while True:
    for start in range(0, len(valid_data), batch_size):
      end = min(start + batch_size, len(valid_data))
      size = batch_size if end < len(valid_data) else len(valid_data) - start

      X_data = np.ones([size, width, height, 3])
      labels = np.full([size, absolute_max_string_len], -1)
      input_length = np.zeros([size, 1])
      label_length = np.zeros([size, 1])
      source_str = []
      i_valid_batch = valid_data[start:end]

      for i, file in enumerate(i_valid_batch):
        X_data[i] = np.array(cv2.imread(file+'.jpg'), dtype=np.float32).transpose(1, 0, 2) / 255
        input_length[i] = width // 2**2 - 2
        license_plate = license_plate_single(file)
        label_length[i] = len(license_plate)
        source_str.append(license_plate)
        for j, key in enumerate(license_plate):
          labels[i, j] = name2id[key]

      inputs = {'the_input': np.array(X_data),
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                'source_str': source_str  # used for visualization only
                }
      outputs = {'ctc': np.zeros([batch_size])}  # dummy data for dummy loss function
    yield inputs, outputs


def train_generator_old(train_data, batch_size=64):
  while True:
    for start in range(0, len(train_data), batch_size):
      x_batch = []
      y_batch = []
      end = min(start + batch_size, len(train_data))
      i_train_batch = train_data[start:end]

      for image in i_train_batch:
        x_batch.append(np.array(cv2.imread(image+'.jpg'), dtype=np.float32) / 255)

      for label in license_plate(i_train_batch):
        i_y_train = [name2id[key] for key in label]
        i_y_train = to_categorical(i_y_train, num_classes=len(ALL_LABELS))
        y_batch.append(np.concatenate((i_y_train)))

      size_batch = [read_location(file+'.xml') for file in i_train_batch]

      yield np.array(x_batch), y_batch, size_batch


def valid_generator_old(valid_data, batch_size=64):
  while True:
    for start in range(0, len(valid_data), batch_size):
      x_batch = []
      y_batch = []
      end = min(start + batch_size, len(valid_data))
      i_valid_batch = valid_data[start:end]

      for i in i_valid_batch:
        x_batch.append(np.array(cv2.imread(i+'.jpg'), dtype=np.float32) / 255)

      for label in license_plate(i_valid_batch):
        i_y_valid = [name2id[key] for key in label]
        i_y_valid = to_categorical(i_y_valid, num_classes=len(ALL_LABELS))
        y_batch.append(np.concatenate((i_y_valid)))

      size_batch = [read_location(file+'.xml') for file in i_valid_batch]

      yield x_batch, y_batch, size_batch


if __name__ == '__main__':
  DATA_DIR = '/home/woodylin/tensorflow3/Plate_Recognition/data/train/'
  ALL_FILES = read_dirfile_name(DATA_DIR)
  for x_batch, y_batch, size_batch in gen(ALL_FILES):
    print('x_batch', x_batch)
    print('y_batch', y_batch)
    print('size_batch', size_batch)
    print('Exit')
