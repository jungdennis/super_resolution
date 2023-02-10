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
from DLCs.metric_tools import metric_histogram, calc_FAR_FRR, calc_EER, graph_FAR_FRR

# Datapath
path_hr = "C:/super_resolution/data/image_SYSU/HR"
path_lr = "C:/super_resolution/data/image_SYSU/LR"
path_sr = "C:/super_resolution/data/image_SYSU/SR"
path_img = path_hr

path_a = "/A_set"
path_b = "/B_set"
path_fold = path_a

path_train_img = "/train/images"
path_val_img = "/val/images"
path_test_img = "/test/images"

path_log = "C:/super_resolution/log/log_classification/metric_log"
path_metric = path_log + "/metric_HR_A_SYSU.pt"
# path_rate = path_log + "/FAR_FRR_LR_B.pt"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



# Dataset settings
class Dataset_for_Classification(data.Dataset):
    def __init__(self, **kwargs):
        self.path_img = kwargs['path_img']
        self.path_fold = kwargs['path_fold']
        self.path_data = kwargs['path_data']

        try:
            self.mode = kwargs['mode']
        except :
            self.mode = "all"

        self.list_files = os.listdir(self.path_img + self.path_fold + self.path_data)

        # mixup, cutmix는 좀 더 찾아보고 넣자...
        # 일단은 없이 할 수 있는 한 augmentation 진행함
        self.transform_raw = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor()])

        self.label_frame = {}
        self.label_list = []
        for name in self.list_files:
            label = name.split(".")[0].split("_")[0]
            frame = name.split(".")[0].split("_")[1]

            if label not in self.label_list :
                self.label_frame[label] = []
                self.label_list.append(label)

            self.label_frame[label].append(frame)
            self.label_frame[label].sort()


        self.img_list=[]
        if self.mode == "center" :
            for label in list(self.label_list) :
                _list = self.label_frame[label]
                center_frame = _list[0]
                name_frag = label + "_" + center_frame
                for name in self.list_files :
                    if name_frag in name :
                        self.img_list.append(name)
        elif self.mode == "all" :
            for label in list(self.label_list):
                _list = self.label_frame[label]
                for i in range(1, len(_list)) :
                    img_frame = _list[i]
                    name_frag = label + "_" + img_frame
                    for name in self.list_files :
                        if name_frag in name :
                            self.img_list.append(name)

        self.img_list = set(self.img_list)
        self.img_list = list(self.img_list)
        print(len(self.img_list))


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        _name = self.list_files[idx]
        pil_img = Image.open(self.path_img + self.path_fold + self.path_data + "/" + _name)
        label = int(_name.split(".")[0].split("_")[0])

        label_tensor = torch.Tensor([label]).to(torch.int64)

        return self.transform_raw(pil_img), label_tensor

if __name__ == "__main__":
    # train, valid, test dataset 설정
    dataset_center = Dataset_for_Classification(path_img=path_img,
                                                path_fold=path_fold,
                                                path_data=path_test_img,
                                                mode="center")
    dataset_image = Dataset_for_Classification(path_img=path_img,
                                               path_fold=path_fold,
                                               path_data=path_test_img)
    num_img = len(dataset_image)
    print(f"dataset_center : {len(dataset_center.img_list)}, dataset_image : {len(dataset_image.img_list)}")


    # dataset을 dataloader에 할당
    # 원래는 torch의 dataloader를 부르는게 맞지만
    # 멀티코어 활용을 위해 DataLoader_multi_worker_FIX를 import 하여 사용
    BATCH_SIZE = 1

    dataloader_center = DataLoader_multi_worker_FIX(dataset=dataset_center,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    prefetch_factor=2,
                                                    drop_last=True)

    dataloader_image = DataLoader_multi_worker_FIX(dataset=dataset_image,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=True)

    # 학습 설정
    # device, scaler, model, loss, epoch, batch_size, lr, optimizer, scheduler
    EPOCH = 1
    num_classes = len(list(dataset_center.label_frame.keys()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

    model_path = "C:/super_resolution/log/log_classification/make_model/SYSU/model"
    latest_model = os.listdir(model_path)[-1]
    model = torch.load(model_path + "/" + latest_model)
    model.head = nn.Identity()
    model.to(device)

    # train, valid, test
    size = len(dataloader_center.dataset)

    for i_epoch_raw in range(EPOCH):
        i_epoch = i_epoch_raw + 1

    center_list = []
    distance_same = []
    distance_diff = []

    # metric 측정을 위한 center 추출
    for i_dataloader in dataloader_center:
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            label = label.item()
            center = model(data)
            center_list.append([label, center])

    print(len(center_list))

    # metric 측정을 위한 feature 추출
    cnt_feature = 1
    cnt_pass = 0
    for i_dataloader in dataloader_image:
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            label = label.item()
            output = model(data)
            print(f"Calculating Feature and Distance... [{cnt_feature}/{num_img}]")
            cnt_feature += 1

            for center_label, center in center_list:
                distance = (output - center).pow(2).sum().sqrt().item()
                if center_label == label:
                    distance_same.append(distance)
                else:
                    distance_diff.append(distance)

    distance_same.sort()
    distance_diff.sort()

    torch.save({'distance_same' : distance_same,
                'distance_diff' : distance_diff}, path_metric)