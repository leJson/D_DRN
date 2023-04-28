#! python3
"""
Created on Sat Oct 10 23:20:13 2021
@author: li jing song
"""
from torch.utils.data import  Dataset
import os
import sys
import numpy as np
import h5py
import numpy as np
import time
# import numpy_gpu as gpu
import cv2
# import pandas as pd
# import cupy as cp
# import open3d as o3d
import os
import torch
import datetime
from logger import Logger
from utils_cau import *
import open3d as o3d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))    # 当前py文件所在的路径
sys.path.append(BASE_DIR)                                # 显示py文件当前路径
print("the path is: %s" % BASE_DIR)                      # 加入到系路径里

# ply_data_train1.h5路径
data_path = '/home/ljs/datasets/pointcloud_classifier/small_dataset'
train_data = 'train1.h5'
test_data = 'test0.h5'
H5_FILE = os.path.join(data_path, train_data)
# H5_FILE = 'H:/制作的数据集/小数据集/train0.h5'


class MYData(Dataset):
    def __init__(self, data_path=None, data_name=None):
        self.data_path = data_path
        if len(data_name) > 1:
            self.data_path = os.path.join(data_path, data_name[0])
            with h5py.File(self.data_path) as f:
                self.data = f['data'][:]
                self.label = f['label'][:]
                # print(type(self.data))
                # print(type(self.label))
            for i in range(1, len(data_name)):
                self.data_path = os.path.join(data_path, data_name[i])
                with h5py.File(self.data_path) as f:
                    self.data = np.concatenate((self.data, f['data'][:]), axis=0)
                    self.label = np.concatenate((self.label, f['label'][:]), axis=0)
        else:
            self.data_path = os.path.join(data_path, data_name[0])
            with h5py.File(self.data_path) as f:
                self.data = f['data'][:]  # 读取主键'data'的值
                self.label = f['label'][:]  # 读取主键'label'的值

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        print(np.asarray(self.data[item]).shape)
        return np.asarray(self.data[item]), np.asarray(self.label[item])


class MYData_Lettuce(Dataset):
    def __init__(self, data_path=None, data_name=None, data_class='train', points_num=4096):
        self.data_path = data_path
        self.data = list()
        self.label = list()
        if len(data_name) > 1:
            self.data_path = os.path.join(data_path, data_name[0])
            with h5py.File(self.data_path) as f:
                self.data = f['data'][:]
                self.label = f['label'][:]
                # print(type(self.data))
                # print(type(self.label))
            for i in range(1, len(data_name)):
                self.data_path = os.path.join(data_path, data_name[i])
                with h5py.File(self.data_path) as f:
                    self.data = np.concatenate((self.data, f['data'][:]), axis=0)
                    self.label = np.concatenate((self.label, f['label'][:]), axis=0)
        else:
            self.data_path = os.path.join(data_path, data_name[0], 'out_%s/%s' % (points_num, data_class))
            print('self.data_path:', self.data_path)
            self.label_path = os.path.join(data_path, data_name[0], 'label/GroundTruth.json')
            contxt = self.read_Label(self.label_path)

            # print(contxt)
            # print(self.data_path)
            # print(self.label_path)
            pcd_files = sorted(glob_image_dir(self.data_path, cap='*.pcd'),  key=lambda x: int(x.split('/')[-1].split('.')[0]))
            # print(pcd_files)
            # print('loaddataer:', glob_image_dir(self.data_path, cap='*.pcd'))
            # for pcd_file in glob_image_dir(self.data_path, cap='*.pcd'):
            for pcd_file in pcd_files:
                # print('pcd_file:', pcd_file)
                image_index = pcd_file.split('/')[-1].split('.')[0]
                # print('image_index:', image_index)
                if int(image_index) > 0:
                    per_label = self.get_per_label(contxt, image_index)
                    if per_label:
                    # print('per_label:', per_label)
                    # print('index:', pcd_file.split('/')[-1].split('.')[0])
                    # print(self.read_PCD_PointClouds(pcd_file))
                        self.data.append(self.read_PCD_PointClouds(pcd_file))
                        self.label.append(per_label)

                    # print('%sper_label:'%image_index, per_label )
            # print(self.data)
            self.data = np.asarray(self.data)
            self.label = np.asarray(self.label)
            print('data_shape:', np.asarray(self.data).shape)
            print('label_shape:', np.asarray(self.label).shape)

    def read_PCD_PointClouds(self, file_path):
        # file_path = '/home/ljs/workspace/eccv/FirstTrainingData/out_4096/train/38.pcd'
        pcd = o3d.io.read_point_cloud(file_path)
        point_cloud = np.asarray(pcd.points)
        # color_cloud = np.asarray(pcd.colors)*255
        color_cloud = np.asarray(pcd.colors)
        points = np.concatenate([point_cloud, color_cloud], axis=1)
        # print('points.shape:', points.shape)
        # return points
        return point_cloud

    def read_Label(self, path):
        # path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
        contxt = json_load(path)
        contxt = contxt['Measurements']
        return contxt

    def get_per_label(self, contxt, index):
        # path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
        # index = '50'
        per_label = list()
        # print(100 * '*')
        item = 'Image%s' % str(index)
        # print('contxt:', contxt)
        # if contxt.has_key(item):
        if item in contxt:
            per_label.append(float(contxt[item]['FreshWeightShoot']))
            per_label.append(float(contxt[item]['DryWeightShoot']))
            per_label.append(float(contxt[item]['Height']))
            per_label.append(float(contxt[item]['Diameter']))
            per_label.append(float(contxt[item]['LeafArea']))
            return per_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # print(np.asarray(self.data[item]).shape)
        # print("*"*100)
        self.data[item][:, 0] = self.data[item][:, 0]*1.0 #0,0.1,0.5,0.8,1,1.5,2 : finished:0,0.1,0.5,0.8,1,1.5
        self.data[item][:, 1] = self.data[item][:, 1]*1.0
        return np.asarray(self.data[item]), np.asarray(self.label[item])


def read_h5file_keys(h5_filename):
    """
    读取H5文件里的键值
    :param h5_filename:
    :return:
    """
    with h5py.File(h5_filename) as f:
        return [item for item in f.keys()]


def main():
    keys = read_h5file_keys(H5_FILE)
    print("key is : %s" % keys)    # ['data', 'faceId', 'label', 'normal ']

    with h5py.File(H5_FILE) as f:
        data = f['data'][:]    # 读取主键'data'的值
        label = f['label'][:]   # 读取主键'label'的值


    index = 18
    print('data[index].shape:', data[index].shape)

    file_name = 'pointcloud_index%s_%s.pcd' % (index, label[index])
    file_save_path = os.path.join(data_path, 'pointcloud', file_name)
    print('file_save_path:', file_save_path)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data[index])
    # # o3d.io.write_point_cloud(os.path.join(file_name + ".pcd"), pcd)
    # o3d.io.write_point_cloud(file_save_path, pcd)

    print(data.shape)    # 2048组，每组1024个点，点为三维数据(x,y,z)
    print(data)
    # print(data)
    print(label.shape)
    # print(sorted(label))
    print(np.unique(label))


def _test():
    data_path = '/home/ljs/datasets/pointcloud_classifier/small_dataset'
    train_data = 'train1.h5'
    test_data = 'test0.h5'
    train_path = os.path.join(data_path, train_data)
    test_path = os.path.join(data_path, test_data)
    train_data = MYData(data_path=train_path)
    test_data = MYData(data_path=test_path)
    # print(a.__len__())
    print('train_data_num:', len(train_data))
    print('test_data_num:', len(test_data))


def _test_tensor():
    batch_size = 16
    arr = torch.ones(1024, dtype=int)
    arr_sum = torch.zeros(1024, dtype=int)
    for i in range(1, batch_size):
        arr_sum = torch.cat((arr_sum, arr*i))
    print(arr_sum)


def write_log(contxt):
    log_path = './log/train_res.txt'
    log = Logger(log_path)
    log.append(contxt)
    log.close()
    pass


if __name__ == '__main__':
    data_path = '/home/ljs/workspace/eccv'
    data_name = ['FirstTrainingData']
    data = MYData_Lettuce(data_path, data_name)
    # print(os.path.join(data_path, data_name[0], 'out_4096/train'))
    # write_log('sdsdd')
    # test_tensor()
    # main()
    # test()
