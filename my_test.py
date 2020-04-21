import unittest
import math
import numpy as np
import cv2
import time

import test_task
import my_impl

# prepare kernel here to simplify
def my_impl_facade(image: np.ndarray, kernel_size: int, sigma: float):

    kernel_width = kernel_size
    if kernel_width % 2 == 0:
        kernel_width = kernel_width - 1  # make sure kernel width only sth 3,5,7 etc

    # create empty matrix for the gaussian kernel #
    kernel_matrix = np.empty((kernel_width, kernel_width), np.float32)
    kernel_half_width = kernel_width // 2
    for i in range(-kernel_half_width, kernel_half_width + 1):
        for j in range(-kernel_half_width, kernel_half_width + 1):
            kernel_matrix[i + kernel_half_width][j + kernel_half_width] = (
                    np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                    / (2 * np.pi * sigma ** 2)
            )
    gaussian_kernel = kernel_matrix / kernel_matrix.sum()
    start_time = time.time()

    my_impl.run(image, gaussian_kernel)

    end_time = time.time()

    return image, start_time, end_time

class UT1(unittest.TestCase):

    def test_orig(self):
      img = cv2.imread(r'img.jpg')
      res = test_task.run(img, 15, 3)
      
      start_time = time.time()
      res = test_task.run(img, 15, 3)
      print("--- test_orig: %s seconds ---" % (time.time() - start_time))

    def test_optimized(self):
      img = cv2.imread(r'img.jpg')
      # res, start_time, end_time = my_impl_facade(img, 15, 3)
      res, start_time, end_time = my_impl_facade(img, 15, 3)
      res, start_time, end_time = my_impl_facade(img, 15, 3)
      
      print("--- test_optimized: %s seconds ---" % (end_time - start_time))

if __name__ == '__main__':
    img = cv2.imread(r'img.jpg')
    ret, start_time, end_time = my_impl_facade(img, 15, 3)
    cv2.imshow('img', ret)
    cv2.waitKey()
    print("took: %s seconds ---" % (end_time - start_time))

