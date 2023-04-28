"""
Created on Sat Oct 10 23:20:13 2021
@author: li jing song
"""
import os.path as osp

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T


def get_dataset(num_points):
    data_name = '10'
    name = 'ModelNet%s' % data_name
    print('path:', osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name))
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    path = osp.join('/home/ljs/PycharmProjects/data', name)
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)
    train_dataset = ModelNet(
        path,
        name=data_name,
        train=True,
        transform=transform,
        pre_transform=pre_transform)
    test_dataset = ModelNet(
        path,
        name=data_name,
        train=False,
        transform=transform,
        pre_transform=pre_transform)
    return train_dataset, test_dataset


