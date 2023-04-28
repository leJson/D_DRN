"""
改自 pytorch_geometric
Created on Sat Oct 10 23:20:13 2021
@author: li jing song
"""
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
import csv
import os

# from apex import amp
# from torch.cuda import amp
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
POINTS_NUM = 1024
CLASS_LABEL=2
np.random.seed(1000)


def csv_writer(file_path, item='Predict_Number'):
    f = open(file_path, 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['FileName', item])
    return writer, f


def get_batch(batch_size, points_num=POINTS_NUM):
    arr = torch.ones(points_num, dtype=int)
    arr_sum = torch.zeros(points_num, dtype=int)
    for i in range(1, batch_size):
        arr_sum = torch.cat((arr_sum, arr*i))
    return arr_sum
# .module.state_dict()


def save_net(fname, net):
    # torch.save(net.module.state_dict(), fname)
    torch.save(net.state_dict(), fname)
    net.train()
    if torch.cuda.is_available():
        net.cuda()


def run(train_dataset, test_dataset, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay):
    # CLASS_LABEL=0
    criterion = nn.MSELoss()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.99)
    csv_filePath = os.path.join('result',
                                '%s_test_accuracy_' % (CLASS_LABEL) + time.strftime("%d-%m-%Y-%H-%M-%S") + '.csv')
    writer_ljs, f = csv_writer(csv_filePath)
    # writer, f = csv_writer(csv_filePath)
    # apex = True
    # cuda = True
    # scaler = amp.GradScaler(enabled=cuda)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # print('test_loader:', test_loader)

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        # with amp.autocast(enabled=cuda):
        train_loss = train(model, optimizer, train_loader, device, criterion)
        model.eval()
        test_acc, csv_out, csv_label_out = test(model, test_loader, device, epoch)
        savemodel_name = os.path.join('weights', '%s_epoch.pt'%epoch)

        save_net(savemodel_name, model)
        csv_out = np.concatenate((csv_out, [test_acc.detach().cpu().numpy()], [train_loss.detach().cpu().numpy()]))
        # csv_out = np.asarray(list(csv_out).append(test_acc))
        # print('csv_out:',  csv_out)
        if epoch == 1:
            writer_ljs.writerow(csv_label_out)
        writer_ljs.writerow(csv_out)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        print('Epoch: {:03d}, Test: {:.4f}, Duration: {:.2f}'.format(
            epoch, test_acc, t_end - t_start))
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    f.close()


def train(model, optimizer, train_loader, device, criterion):
    # criterion = nn.MSELoss()

    model.train()
    total_loss = 0

    csv_out_list = torch.tensor(list()).to(device)
    csv_out_label = torch.tensor(list()).to(device)

    for data, label in train_loader:
        optimizer.zero_grad()
        batch_size = data.shape[0]
        # data = data.to(device)
        # 在线增强1: 打乱点云顺序
        data = np.asanyarray(data)
        for sample in data:
           np.random.shuffle(sample)
        data = torch.from_numpy(data)
        # print('data_shape:', data.shape)
        # print('data:', data)
        # 在线增强2:随机移动
        # np.random.seed(10)
        if np.random.randint(0, 2):

            data[:, :, 0] = -data[:, :, 0]
            if np.random.randint(0, 2):
                data[:, :, 1] = -data[:, :, 1]

        # if np.random.randint(0, 2):
        #     data[:, :, 0] = data[:, :, 0] + np.random.randint(-3, 3) * 0.01
        #     data[:, :, 1] = data[:, :, 1] + np.random.randint(-3, 3) * 0.01
        #     data[:, :, 2] = data[:, :, 2] + np.random.randint(-3, 3) * 0.01


        # print('train_data.shape', data.shape)
        data = Variable(data.float().reshape(-1, 3).to(device))


        # label = label.long().to(device)
        label = label.float().to(device)
        # label = label[:, 2:4]
        # print('label:', label[:, 2:4])
        batch = get_batch(batch_size).to(device)
        # label = Variable(label[:, 3])
        label = Variable(torch.unsqueeze(label[:, CLASS_LABEL], dim=1))
        # print('trian_label.shape:', label.shape)
        # out = copy.copy(model(data, batch))
        # with amp.autocast(enabled=True):
        out = model(data, batch)
        loss = criterion(out, label)
        # loss = F.nll_loss(out, data.y)
        total_loss += loss

        # scaler.scale(loss).backward()
        loss.backward()
        optimizer.step()
        # with amp.autocast(enabled=True):

    print('train_loss:', total_loss/len(train_loader))
    return total_loss/len(train_loader)


def test(model, test_loader, device, epoch):

    # print('epoch:', epoch)
    # correct = 0
    total_mse = 1
    csv_out_list = torch.tensor(list()).to(device)
    csv_out_label = torch.tensor(list()).to(device)
    for data, label in test_loader:
        # data = data.to(device)
        batch_size = data.shape[0]
        # print(' batch_size :',  batch_size )
        # 打乱点云顺序
        data = np.asanyarray(data)
        for sample in data:
          np.random.shuffle(sample)
        data = torch.from_numpy(data)
        data = data.float().reshape(-1, 3).to(device)
        label = label.float().to(device)
        batch = get_batch(batch_size).to(device)
        # label = label[:, 3]
        # label = Variable(label[:, 3])
        label = Variable(torch.unsqueeze(label[:, CLASS_LABEL], dim=1))
        out = copy.copy(model(data, batch))
        if epoch == 1:
            csv_label = torch.squeeze(label, dim=-1)
            csv_out_label = torch.cat((csv_out_label, csv_label))

        csv_out = torch.squeeze(out, dim=-1)
        csv_out_list = torch.cat((csv_out_list, csv_out))
        # print(csv_out)
        # csv_out
        # print(csv_out_list.cpu().numpy())
        loss = torch.nn.functional.mse_loss(out, label)
        # pred = model(data.pos, data.batch).max(1)[1]
        total_mse += loss
    test_acc = total_mse / len(test_loader)
    # model.train()
    return test_acc, csv_out_list.detach().cpu().numpy(), csv_out_label.detach().cpu().numpy()
