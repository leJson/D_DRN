# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 23:20:13 2021
@author: li jing song
"""
import shutil
import sys
import os
import cv2
import numpy as np
import json
# import pyrealsense2 as rs
import time
import random
from matplotlib import pyplot as plt
import glob


def _json_load(filename):
    with open(filename, "r") as fr:
        vars = json.load(fr)
    for k, v in vars.items():
        vars[k] = np.array(v)
    return vars


def json_load(filename):
    with open(filename, "r") as fr:
        vars = json.load(fr)
    # for k, v in vars.items():
    #     vars[k] = np.array(v)
    return vars


def write_txt(fname='123.txt', contxt='str'):
    """
    :param fname:
    :param contxt:
    :return:
    """
    if os.path.exists(fname):
        with open(fname, "a") as file:
            file.write(contxt)
    else:
        with open(fname, "w") as file:
            file.write(contxt)
    return 0


def point_img(img, x, y, thick=-1):
    """
    :param img:
    :param x: x location
    :param y: y location
    :param thick:
    :return:
    """
    cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=thick)
    print('img.shape', img.shape)
    print('img[x][y][0],img[x][y][1],img[x][y][2]', img[y][x][0], img[y][x][1], img[y][x][2])
    # img[y][x][0] = 0
    # img[y][x][1] = 0
    # img[y][x][2] = 255
    return img


def get_feature_points(img):
    """
    get feature points by Shi-Tomasi
    :param img: img mat
    :return: feature points
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=8000000,
                          qualityLevel=0.002,
                          minDistance=0,
                          blockSize=5,
                          useHarrisDetector=True,
                          )

    #img = cv2.bilateralFilter(img, 16, 50, 50)
    points = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)
    return points


def div_manual_corners(array):
    """
    :param array:
    :return:
    """
    h, w = array.shape
    h_array = np.zeros([h//2, w])
    w_array = np.zeros([h//2, w])
    for i in range(h):
        if i % 2 == 0:
            w_array[i // 2, :] = array[i, :]
        if i % 2 == 1:
            h_array[i // 2, :] = array[i, :]
    return h_array, w_array


def glob_image_dir(path='save_img1', cap='save'):
    """
    :param path:
    :param cap:
    :return:
    """
    img_paths = glob.glob(path + '/%s*' % cap)
    return img_paths


def plot_show(num=2, *args):
    '''
    for example :  plot_show(3, cuted_img, cuted_img, cuted_img, cuted_img, cuted_img)
    :param num: the number of image in each raw
    :param args:
    :return:
    '''
    raw_num = len(args)//num
    index = 0
    if len(args) % num > 0:
        index = 1
    for i in range(len(args)):
        plt.subplot(raw_num+index, num, i+1)
        plt.imshow(args[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()


def plot_show_hot(num=2, *args):
    '''
    for example :  plot_show(3, cuted_img, cuted_img, cuted_img, cuted_img, cuted_img)
    :param num: the number of image in each raw
    :param args:
    :return:
    '''
    raw_num = len(args)//num
    index = 0
    if len(args) % num > 0:
        index = 1
    for i in range(len(args)):
        plt.subplot(raw_num+index, num, i+1)
        plt.imshow(args[i], cmap=plt.cm.jet)
        plt.axis('off')
    plt.show()


def glob_image_dir(path='',  cap=''):
    """
    :param path:
    :param cap:
    :return:
    """
    img_paths = glob.glob(path+'/%s*' % cap)
    return img_paths


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


if __name__ == '__main__':
    list_test = [1, 2, 3, 45, 33, 3]
    plt.plot(list_test)
    plt.show()
