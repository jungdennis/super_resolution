import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from mp_dataloader import DataLoader_multi_worker_FIX


# Datapath
path_lr = "C:/super_resolution/image/LR"
path_hr = "C:/super_resolution/image/HR"
path_sr = "C:/super_resolution/image/SR"

path_a = "/A_set"
path_b = "/B_set"

path_train_img = "/train/images"
path_val_img = "/val/images"
path_test_img = "/test/images"

# Dataset settings
class Classification_Dataset(data.Dataset):
    def __init__(self, **kwargs):
        self.path_img = kwargs['path_img']
        self.path_fold = kwargs['path_fold']
        self.path_data = kwargs['path_data']

        self.list_files = os.listdir(self.path_img + self.path_fold + self.path_data)

        self.transform_raw = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        _name = self.list_files[idx]
        pil_img = Image.open(self.path_img + self.path_fold + self.path_data + "/" + _name)
        label = int(_name.split(".")[0].split("_")[4])

        return self.transform_raw(pil_img), label


if __name__ == "__main__":
    # 기본 설정
    # device, scaler, model, loss, epoch, batch_size, lr, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)
    model =
    model.to(device)
    criterion = torch.nn.L1Loss()
    LR = 2e-4
    EPOCH = 300
    BATCH_SIZE = 16
    optimizer = torch.optim.Adam(model.parameters()
                                 , lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer
                                                ,step_size=50
                                                ,gamma=0.5)

    # train, valid, test dataset 설정
    dataset_train = Classification_Dataset()

    dataset_val = Classification_Dataset()

    dataset_test = Classification_Dataset()

    # dataset을 dataloader에 할당
    # 원래는 torch의 dataloader를 부르는게 맞지만
    # 멀티코어 활용을 위해 DataLoader_multi_worker_FIX를 import 하여 사용
    dataloader_train = DataLoader_multi_worker_FIX(dataset=dataset_train
                                                   ,batch_size=BATCH_SIZE
                                                   ,shuffle=True
                                                   ,num_workers=2
                                                   ,prefetch_factor=2
                                                   ,drop_last=True)

    dataloader_val = DataLoader_multi_worker_FIX(dataset=dataset_val
                                                 ,batch_size=1
                                                 ,shuffle=True
                                                 ,num_workers=2
                                                 ,prefetch_factor=2
                                                 ,drop_last=False)

    dataloader_test = DataLoader_multi_worker_FIX(dataset=dataset_test
                                                  ,batch_size=1
                                                  ,shuffle=False
                                                  ,num_workers=2
                                                  ,prefetch_factor=2
                                                  ,drop_last=False)

    # train, valid, test
    for i_epoch_raw in range(EPOCH):
        i_epoch = i_epoch_raw + 1

        # train
        optimizer.zero_grad()
        for i_dataloader in dataloader_train:
            data, label = i_dataloader
            data = data.to(device)
            label = label.to(device)

            data = data.requires_grad_(True)

            output = model(data)

            loss = criterion(output, label)
            print("loss", loss.item())
            amp_scaler.scale(loss).backward(retain_graph=False)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # valid
        for i_dataloader in dataloader_val:
            data, label = i_dataloader
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = model(data)
                loss = criterion(output, label)
                print(loss.item())

        # test
        for i_dataloader in dataloader_val:
            data, label = i_dataloader
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = model(data)
                loss = criterion(output, label)
                print(loss.item())