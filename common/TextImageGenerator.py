import os
import string
import codecs
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import cairocffi as cairo
import keras.callbacks
from keras.preprocessing import image
from keras import backend as K
from keras.utils.data_utils import get_file


# character classes and matching regex filter
# 用于过滤合法字符的正则表达式
regex = r'^[a-z ]+$'
alphabet = u'abcdefghijklmnopqrstuvwxyz '


# Translation of characters to unique integer values
def text_to_labels(text):
  ret = []
  for char in text:
    ret.append(alphabet.find(char))
  return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
  ret = []
  for c in labels:
    if c == len(alphabet):  # CTC Blank
      ret.append("")
    else:
      ret.append(alphabet[c])
  return "".join(ret)


# only a-z and space..probably not to difficult
# to expand to uppercase and symbols
# 检查是否有效的字符串输入

def is_valid_str(in_str):
  search = re.compile(regex, re.UNICODE).search
  return bool(search(in_str))


# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1
# 添加高斯噪音
def speckle(img):
  severity = np.random.uniform(0, 0.6)
  blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
  img_speck = (img + blur)
  img_speck[img_speck > 1] = 1
  img_speck[img_speck <= 0] = 0
  return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise
# 输出一串字符到一个图片里面
def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
  surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
  with cairo.Context(surface) as context:
    context.set_source_rgb(1, 1, 1)  # White
    context.paint()
    # this font list works in CentOS 7
    if multi_fonts:
      fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono']
      context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                               np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
    else:
      context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(25)
    box = context.text_extents(text)
    border_w_h = (4, 4)
    if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
      raise IOError('Could not fit string into image. Max char count is too large for given image width.')

    # teach the RNN translational invariance by
    # fitting text box randomly on canvas, with some room to rotate
    max_shift_x = w - box[2] - border_w_h[0]
    max_shift_y = h - box[3] - border_w_h[1]
    top_left_x = np.random.randint(0, int(max_shift_x))
    if ud:
      top_left_y = np.random.randint(0, int(max_shift_y))
    else:
      top_left_y = h // 2
    context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
    context.set_source_rgb(0, 0, 0)
    context.show_text(text)

  buf = surface.get_data()
  a = np.frombuffer(buf, np.uint8)
  a.shape = (h, w, 4)
  a = a[:, :, 0]  # grab single channel
  a = a.astype(np.float32) / 255
  a = np.expand_dims(a, 0)
  if rotate:
    a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
  a = speckle(a)

  return a


# 打乱list里面的图片顺序
def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# 用于内存中产生不同的训练图片
# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations
class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(alphabet) + 1

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        # monogram file is sorted by frequency in english speech
        with codecs.open(self.monogram_file, mode='r', encoding='utf-8') as f:
            for line in f:
                if len(tmp_string_list) == int(self.num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)

        # bigram file contains common word pairings in english speech
        with codecs.open(self.bigram_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == self.num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        if len(tmp_string_list) != self.num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
        # interlace to mix up the easy and hard words
        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = text_to_labels(word)
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(16000, 4, 1)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, ud=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if 3 <= epoch < 6:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=False)
        elif 6 <= epoch < 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=True, ud=True, multi_fonts=True)
        if epoch >= 21 and self.max_string_len < 12:
            self.build_word_list(32000, 12, 0.5)


if __name__ == '__main__':
  img_w = 128
  img_h = 64
  pool_size = 2
  minibatch_size = 32
  words_per_epoch = 16000
  val_split = 0.2
  val_words = int(words_per_epoch * (val_split))

  fdir = os.path.dirname(get_file('wordlists.tgz', origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))
  img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                               bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                               minibatch_size=minibatch_size,
                               img_w=img_w,
                               img_h=img_h,
                               downsample_factor=(pool_size ** 2),
                               val_split=words_per_epoch - val_words
                               )

  ## 测试一下 TextImageGenerator
  img_gen.on_train_begin()
  ret = img_gen.get_batch(0, 20, train=True)
  X_data = ret[0]['the_input']
  one_image = X_data[0].reshape((128, 64))
  print(one_image.shape)
  plt.imshow(one_image)
  plt.show()
