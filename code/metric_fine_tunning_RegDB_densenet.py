import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random

import cv2
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn.functional import one_hot
import torchvision.transforms as transforms
from torchinfo import summary
from torchvision.models.densenet import densenet169

from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.data_record import RecordBox

# Datapath
path_hr = "C:/super_resolution/data/image/HR"
path_lr = "C:/super_resolution/data/image/LR_4_noise10"
path_sr = "C:/super_resolution/data/image/SR"

path_a = "/A_set"
path_b = "/B_set"

path_train_img = "/train/images"
path_val_img = "/val/images"
path_test_img = "/test/images"

path_log = "C:/super_resolution/log/log_metric/metric_model/Reg/densenet"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dt_now = datetime.datetime.now()
time = str(dt_now.year) + "_" + str(dt_now.month) + "_" + str(dt_now.day) + "_" + str(dt_now.hour) + "_" + str(dt_now.minute) + "_" + str(dt_now.second)

# Dataset settings
class Dataset_for_Classification(data.Dataset):
    def __init__(self, **kwargs):
        self.path_img = kwargs['path_img']
        self.path_fold = kwargs['path_fold']
        self.path_data = kwargs['path_data']

        self.list_files = os.listdir(self.path_img + self.path_fold + self.path_data)

        # mixup, cutmix는 좀 더 찾아보고 넣자...
        # 일단은 없이 할 수 있는 한 augmentation 진행함
        self.transform_raw = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor()])

        self.label_list = []
        for name in self.list_files :
            label = name.split(".")[0].split("_")[4]
            if label not in self.label_list :
                self.label_list.append(label)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        _name = self.list_files[idx]
        pil_img = Image.open(self.path_img + self.path_fold + self.path_data + "/" + _name)
        label = _name.split(".")[0].split("_")[4]

        '''
        if label not in self.label_list:
            self.label_list.append(label)
        '''
        label_index = self.label_list.index(label)
        label_tensor = torch.Tensor([label_index]).to(torch.int64)
        label_onehot = one_hot(label_tensor, num_classes = len(self.label_list))

        return self.transform_raw(pil_img), label_tensor, label_onehot


if __name__ == "__main__":
    # train, valid, test dataset 설정
    dataset_train = Dataset_for_Classification(path_img=path_hr,
                                               path_fold=path_a,
                                               path_data=path_train_img)
    print(len(dataset_train))
    '''
    dataset_val = Dataset_for_Classification()

    dataset_test = Dataset_for_Classification()
    '''

    # dataset을 dataloader에 할당
    # 원래는 torch의 dataloader를 부르는게 맞지만
    # 멀티코어 활용을 위해 DataLoader_multi_worker_FIX를 import 하여 사용
    BATCH_SIZE = 25

    dataloader_train = DataLoader_multi_worker_FIX(dataset=dataset_train,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=True)

    loss_train = RecordBox(name = "loss_train", is_print = False)
    accuracy_train = RecordBox(name = "accuracy_train", is_print = False)
    lr = RecordBox(name = "learning_rate", is_print = False)

    '''
    dataloader_valid = DataLoader_multi_worker_FIX(dataset=dataset_val,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 prefetch_factor=2,
                                                 drop_last=False)
    '''
    dataloader_test = DataLoader_multi_worker_FIX(dataset=dataset_train,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  prefetch_factor=2,
                                                  drop_last=False)

    # 학습 설정
    # device, scaler, model, loss, epoch, batch_size, lr, optimizer, scheduler
    LR = 1e-5
    EPOCH = 90
    num_classes = len(dataset_train.label_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

    model = densenet169(weights = "DEFAULT")
    model.classifier = nn.Linear(in_features=1664, out_features=num_classes, bias=True)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = LR,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = 30,
                                                gamma = 0.1)

    summary(model, (BATCH_SIZE, 3, 224, 224))

    # train, valid, test
    model.train()
    size = len(dataloader_train.dataset)

    for i_epoch_raw in range(EPOCH):
        i_epoch = i_epoch_raw + 1
        print(f"<epoch {i_epoch}>")
        # train
        optimizer.zero_grad()
        for batch, i_dataloader in enumerate(dataloader_train) :
            data, label_tensor, label_onehot = i_dataloader
            label_onehot = torch.squeeze(label_onehot).to(torch.float64)
            label_tensor = torch.squeeze(label_tensor)
            data = data.to(device)
            label_onehot = label_onehot.to(device)
            label_tensor = label_tensor.to(device)
            data = data.requires_grad_(True)

            output = model(data)

            loss = criterion(output, label_onehot)
            loss_train.add_item(loss.item())

            amp_scaler.scale(loss).backward(retain_graph = False)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()

            if batch % 30 == 0:
                current = batch * len(data)
                print(f"loss: {loss.item()}  [{current}/{size}]")

            with torch.no_grad():
                _total = 0
                _correct = 0

                _, predicted = torch.max(output, 1)
                _total += label_tensor.size(0)
                _correct += (predicted == label_tensor).sum().item()

                _accuracy = 100 * (_correct / _total)

                # if batch % 30 == 0 :
                #     print(f"Ground Truth : {label_tensor}")
                #     print(f"Predicted : {predicted}")
                #     print(f"Result : {(predicted == label_tensor)}")
                accuracy_train.add_item(_accuracy)

            loss_train.update_batch()
            accuracy_train.update_batch()

        lr.add_item(scheduler.get_last_lr()[0])
        scheduler.step()
        lr.update_batch()

        lt = loss_train.update_epoch(path = path_log, is_return = True)
        at = accuracy_train.update_epoch(path = path_log, is_return = True)
        lr.update_epoch(path = path_log)

        if i_epoch % 10 == 0 :
            try :
                torch.save({
                    'epoch': i_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, path_log + f"/ckpt/ckeckpoint_{time}_epoch{i_epoch}.pt")
            except :
                os.makedirs(path_log + "/ckpt")
                torch.save({
                    'epoch': i_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, path_log + f"/ckpt/ckeckpoint_{time}_epoch{i_epoch}.pt")

        print("train : loss {}, accuracy {}%".format(lt, at))
        print("------------------------------------------------------------------------")

    '''
        # valid
        for i_dataloader in dataloader_valid:
            data, label = i_dataloader
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = model(data)
                loss = criterion(output, label)
                print(loss.item())
    '''

    # save model
    try :
        torch.save(model, path_log + f"/model/model_classification_RegDB_densenet_{time}.pt")
    except :
        os.makedirs(path_log + "/model")
        torch.save(model, path_log + f"/model/model_classification_RegDB_densenet_{time}.pt")

    # model = torch.load("C:\super_resolution\log\log_classification\metric_model\Reg\convnext\model\model_classification_RegDB_convnext_2023_2_15_13_17_49.pt")
    # model.to(device)

    # test
    model.eval()
    correct = 0
    total = 0
    for batch, i_dataloader in enumerate(dataloader_test) :
        data, label_tensor, label_onehot = i_dataloader
        label = torch.squeeze(label_tensor)
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output, 1)
            label = torch.squeeze(label)
            # if batch % 30 == 0 :
            #     print(predicted.item(), label.item())
            total += 1
            if predicted.item() == label.item() :
                correct += 1

    print(f"Test Accuracy : {100 * (correct / total)}%")