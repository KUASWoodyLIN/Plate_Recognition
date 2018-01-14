import os
import numpy as np
import xml.etree.ElementTree as ET
from Plate_Recognition.common.read_dirfile import read_dirfile_name


def read_location(file_path):
  tree = ET.ElementTree(file=file_path)
  root = tree.getroot()
  for bndbox in root.iter('bndbox'):
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text
  return xmin, ymin, xmax, ymax


def read_location_max(file_path):
  tree = ET.ElementTree(file=file_path)
  root = tree.getroot()
  for bndbox in root.iter('bndbox'):
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text
    width = int(xmax) - int(xmin)
    height = int(ymax) - int(ymin)
  return width, height


if __name__ == '__main__':
  # path = os.path.join(os.path.dirname(__file__), 'test.xml')
  # print(read_location(path))

  DATA_DIR = '/home/woodylin/tensorflow3/Plate_Recognition/data/train/'
  ALL_FILES = read_dirfile_name(DATA_DIR)

  size = [read_location(file + '.xml') for file in ALL_FILES]
  size_max = [read_location_max(file + '.xml') for file in ALL_FILES]
  size_mean = np.mean(size_max, axis=0)
  print('Exit')