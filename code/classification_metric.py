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
path_hr = "C:/super_resolution/data/image/HR"
path_lr = "C:/super_resolution/data/image/LR"
path_sr = "C:/super_resolution/data/image/SR"

path_a = "/A_set"
path_b = "/B_set"

path_train_img = "/train/images"
path_val_img = "/val/images"
path_test_img = "/test/images"

path_log = "C:/super_resolution/log/log_classification/metric_log"

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
        for name in self.list_files:
            label = name.split(".")[0].split("_")[4]
            if label not in self.label_list:
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

        return self.transform_raw(pil_img), label_tensor

if __name__ == "__main__":
    # train, valid, test dataset 설정
    dataset_test = Dataset_for_Classification(path_img=path_hr,
                                              path_fold=path_a,
                                              path_data=path_test_img)

    # dataset을 dataloader에 할당
    # 원래는 torch의 dataloader를 부르는게 맞지만
    # 멀티코어 활용을 위해 DataLoader_multi_worker_FIX를 import 하여 사용
    BATCH_SIZE = 1

    dataloader_test = DataLoader_multi_worker_FIX(dataset=dataset_test,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=True)

    # loss_train = RecordBox(name = "loss_train", is_print = False)
    # accuracy_train = RecordBox(name = "accuracy_train", is_print = False)
    # lr = RecordBox(name = "learning_rate", is_print = False)

    # 학습 설정
    # device, scaler, model, loss, epoch, batch_size, lr, optimizer, scheduler
    EPOCH = 1
    num_classes = len(dataset_test.label_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

    model_path = "C:/super_resolution/log/log_classification/make_model/model"
    latest_model = os.listdir(model_path)[-1]
    model = torch.load(model_path + "/" + latest_model)
    model.head = nn.Identity()
    model.to(device)

    # train, valid, test
    size = len(dataloader_test.dataset)

    for i_epoch_raw in range(EPOCH):
        i_epoch = i_epoch_raw + 1

    center_dict = {}
    distance_same = []
    distance_diff = []

    # metric 측정을 위한 center 추출
    for i_dataloader in dataloader_test :
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad() :
            label = label.item()
            if label not in center_dict.keys():
                center = model(data)
                center_dict[label] = center
            else :
                pass

    _label_list = list(center_dict.keys())

    # metric 측정
    cnt = 1
    for center_label in _label_list :
        print(f"Start measuring metric of label {center_label} [{cnt}/{len(_label_list)}]")
        center_data = center_dict[center_label]

        for i_dataloader in dataloader_test :
            data, label = i_dataloader
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = model(data)
                label = label.item()
                distance = (output - center_data).pow(2).sum().sqrt().item()

                distance_same.append(distance) if center_label == label else distance_diff.append(distance)

        cnt = cnt + 1

    distance_same.sort()
    distance_diff.sort()
    torch.save({'distance_same' : distance_same,
                'distance_diff' : distance_diff}, path_log + "test_for_rough_convnext_model.pt")
    print(len(distance_same), len(distance_diff))

    plt.hist(distance_same, histtype = "step", color = "b")
    plt.show()
    plt.hist(distance_diff, histtype = "step", color = "g")
    # plt.title("Distribution of distance")
    # plt.legend(["same", "diff"])
    plt.show()