import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from data_handle import MYData_Lettuce
from train_eval import run
# from train_eval_origine import run
import os 
from torch import random

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


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class DRN(torch.nn.Module):
    def __init__(self, out_features):
        super(DRN, self).__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.input_size = 1  # feature points size or word size defualt:129
        self.out_size = 1  # the size of prediction for each word
        self.layer_num = 2

        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
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
        return x


if __name__ == '__main__':
   
    seed_torch(seed=1000)
    data_path = './'
    # data_name = ['ori_data']
    data_name = ['lettuce']
    train_dataset = MYData_Lettuce(data_path=data_path, data_name=data_name, data_class='train', points_num=1024)
    test_dataset = MYData_Lettuce(data_path=data_path, data_name=data_name, data_class='test', points_num=1024)
    out_features = 1
    model = DRN(out_features)

    run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
        args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)


