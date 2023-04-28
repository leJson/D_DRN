"""
Created on Sat Oct 10 23:20:13 2021
@author: li jing song
"""

import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
# from torch_geometric.datasets import ModelNet
# import torch_geometric.transforms as T
# from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from data_handle import MYData_Lettuce
from train_eval import run
# from train_eval_origine import run
import os 
from torch import random
from torch import nn
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=8)  # default: 32
parser.add_argument('--lr', type=float, default=0.001)  # defualt:0.001
parser.add_argument('--lr_decay_factor', type=float, default=0.5) # try 0.3
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()

def seed_torch(seed=1029):
	random.seed()
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
class Dimension_Attention(nn.Module):
    def __init__(self, point_num=1024, dimension=3, reduction=16):
        super(Dimension_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.decode_incode_points = nn.Sequential(
                  nn.Conv2d(in_channels=point_num, out_channels=point_num // reduction, kernel_size=1, stride=1, bias=False),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=point_num // reduction, out_channels=point_num, kernel_size=1, stride=1, bias=False))
        self.decode_incode_dimension = nn.Sequential(
                      nn.Conv2d(in_channels=dimension, out_channels=1, kernel_size=1, stride=1, bias=False),
                      nn.ReLU(),
                      nn.Conv2d(in_channels=1, out_channels=dimension, kernel_size=1, stride=1, bias=False))

        self.sigmoid_dimension = nn.Sigmoid()
        self.sigmoid_point = nn.Sigmoid()

    def forward(self, x):
        w, h, c = x.shape
        x_unsqueeze = torch.unsqueeze(x, dim=-1)
        # x_dcode_p = self.decode_incode_points(x_unsqueeze).permute(0,2,1,3)
        x_dcode_p = x_unsqueeze.permute(0,2,1,3)
        x_pool = self.avg_pool(x_dcode_p)
        x_dcode_d = self.decode_incode_dimension(x_pool)
        x_da = x_dcode_d.reshape([w, -1, 1]).permute(0, 2, 1)
        x_da = self.sigmoid_dimension(x_da)
        out = x * x_da.expand_as(x)
        # print(out.shape)
        return out

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, out_features):
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        # self.lin2 = Lin(512, 256)
        self.lin2 = Lin(512, out_features)
        self.lin3 = Lin(256, out_features)

    def forward(self, pos, batch):

        # sa0_out = (data.x, data.pos, data.batch)
        sa0_out = (None, pos, batch)

        # print('pos.shape:', pos.shape)
        # print('batch.shape:', batch.shape)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        # x = F.relu(self.lin1(x))
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # return F.log_softmax(x, dim=-1)
        return x


class Net_LSTM(torch.nn.Module):
    def __init__(self, out_features):
        super(Net_LSTM, self).__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.input_size = 1  # feature points size or word size defualt:129
        self.out_size = 1  # the size of prediction for each word
        self.layer_num = 2
        # self.rnn_3 = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
        #                    bidirectional=True,
        #                    dropout=0.3,
        #                    # batch_first=True
        #                    )
        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                                 # bidirectional=True, # without bilstm:test=1.8  for final 50 images
                                 dropout=0.3,  # defualt:0.3
                                 batch_first=True
                                 )

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)

        self.lin1 = Lin(1024, 512)
        # self.lin2 = Lin(512, 256)
        self.lin2 = Lin(512, out_features)
        self.lin3 = Lin(256, out_features)

    def forward(self, pos, batch):

        batch_size = batch.reshape(-1, 1024).shape[0]
        # sa0_out = (data.x, data.pos, data.batch)
        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out # x: (batch, 1024)

        x = x.reshape(batch_size, 1024, -1)

        x, hc = self.rnn(x)
        x = self.avgpool(x)

        # print('after_pooling:', x.shape)
        x = x.reshape(batch_size, -1)
        # x = F.relu(self.lin1(x))
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # return F.log_softmax(x, dim=-1)
        return x

class D_DRN(torch.nn.Module):
    def __init__(self, out_features):
        super(D_DRN, self).__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.D_attention = Dimension_Attention()
        self.input_size = 1  # feature points size or word size defualt:129
        self.out_size = 1  # the size of prediction for each word
        self.layer_num = 2
        # self.rnn_3 = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
        #                    bidirectional=True,
        #                    dropout=0.3,
        #                    # batch_first=True
        #                    )
        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                                 # bidirectional=True, # without bilstm:test=1.8  for final 50 images
                                 dropout=0.3,  # defualt:0.3
                                 batch_first=True
                                 )

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)
    
        self.lin1 = Lin(1024, 512)
        # self.lin2 = Lin(512, 256)
        self.lin2 = Lin(512, out_features)
        self.lin3 = Lin(256, out_features)

    def forward(self, pos, batch):
     
        batch_size = batch.reshape(-1, 1024).shape[0]
        # sa0_out = (data.x, data.pos, data.batch)
        # DA_attention
        pos = pos.reshape(batch_size, 1024, -1)
        pos = self.D_attention(pos)
        pos = pos.reshape(-1,3)

        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out # x: (batch, 1024)

        x = x.reshape(batch_size, 1024, -1)

        x, hc = self.rnn(x)
        x = self.avgpool(x)

        # print('after_pooling:', x.shape)
        x = x.reshape(batch_size, -1)
        # x = F.relu(self.lin1(x))
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x

 

if __name__ == '__main__':
   
    seed_torch(seed=1000)
    data_path = './'
    # data_name = ['FirstTrainingData']
    # data_name = ['ori_data']
    data_name = ['lettuce']

    train_dataset = MYData_Lettuce(data_path=data_path, data_name=data_name, data_class='train', points_num=1024)
    # print('train_dataset num:', train_dataset.__len__())
    test_dataset = MYData_Lettuce(data_path=data_path, data_name=data_name, data_class='test', points_num=1024)

    out_features = 1
    model = D_DRN(out_features)
    run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
        args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)




