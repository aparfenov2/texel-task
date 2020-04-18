import unittest
import math

import torch
from torch import nn
import numpy as np
import cv2
import time
import test_task

from torch.utils.cpp_extension import load


class UT1(unittest.TestCase):

    def test_orig(self):
      img = cv2.imread(r'img.jpg')
      res = test_task.run(img, 15, 3)
      
      start_time = time.time()
      res = test_task.run(img, 15, 3)
      print("--- test_orig: %s seconds ---" % (time.time() - start_time))

    def test_optimized(self):
      my_impl = load(
          name='my_impl',
          sources=['my_impl.cpp'],
          extra_cflags=['-O2'],
          verbose=True)
      img = cv2.imread(r'img.jpg')
      res = my_impl.run(img, 15, 3)

      start_time = time.time()
      res = my_impl.run(img, 15, 3)
      print("--- test_optimized: %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':

    my_impl = load(
        name='my_impl',
        sources=['my_impl.cpp'],
        extra_cflags=['-O2'],
        verbose=True)

    img = cv2.imread(r'img.jpg')
    res = my_impl.run(img, 15, 3)
    cv2.imshow('img', np.swapaxes(np.squeeze(res.astype(np.uint8)), 0, -1))
    cv2.waitKey()

