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
from transformers import get_cosine_schedule_with_warmup

from DLCs.model_convnext import convnext_small
from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.data_record import RecordBox

# Datapath
path_hr = "C:/super_resolution/data/image_SYSU/HR"
path_lr = "C:/super_resolution/data/image_SYSU/LR"
path_sr = "C:/super_resolution/data/image_SYSU/SR"

path_a = "/A_set"
path_b = "/B_set"

path_train_img = "/train/images"
path_val_img = "/val/images"
path_test_img = "/test/images"

path_log = "C:/super_resolution/log/log_classification/make_model/SYSU/convnext"

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
            label = name.split(".")[0].split("_")[0]
            if label not in self.label_list :
                self.label_list.append(label)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        _name = self.list_files[idx]
        pil_img = Image.open(self.path_img + self.path_fold + self.path_data + "/" + _name)
        label = _name.split(".")[0].split("_")[0]

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


    '''
    dataset_val = Dataset_for_Classification()

    dataset_test = Dataset_for_Classification()
    '''

    # dataset을 dataloader에 할당
    # 원래는 torch의 dataloader를 부르는게 맞지만
    # 멀티코어 활용을 위해 DataLoader_multi_worker_FIX를 import 하여 사용
    BATCH_SIZE = 16

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
    LR = 5e-5
    EPOCH = 30
    num_classes = len(dataset_train.label_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

    model = convnext_small(pretrained = True, in_22k = True, num_classes = 21841)
    model.head = nn.Linear(in_features = 768, out_features = num_classes, bias = True)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR,
                                  weight_decay=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=EPOCH)

    # train, valid, test
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
                accuracy_train.add_item(_accuracy)

            loss_train.update_batch()
            accuracy_train.update_batch()

        lr.add_item(scheduler.get_last_lr()[0])
        scheduler.step()
        lr.update_batch()

        lt = loss_train.update_epoch(path = path_log, is_return = True)
        at = accuracy_train.update_epoch(path = path_log, is_return = True)
        lr.update_epoch(path = path_log)

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
        torch.save(model, path_log + f"/model/model_classification_SYSU_convnext_{time}.pt")
    except :
        os.makedirs(path_log + "/model")
        torch.save(model, path_log + f"/model/model_classification_SYSU_convnext_{time}.pt")

    # test
    correct = 0
    total = 0
    for i_dataloader in dataloader_test:
        data, label_tensor, label_onehot = i_dataloader
        label = torch.squeeze(label_tensor)
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output, 1)
            label = torch.squeeze(label)
            total += 1
            if predicted.item() == label.item() :
                correct += 1

    print(f"Test Accuracy : {100 * (correct / total)}%")