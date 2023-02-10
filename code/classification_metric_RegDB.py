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
import torchvision.transforms as transforms

from DLCs.mp_dataloader import DataLoader_multi_worker_FIX

# Datapath
path_hr = "C:/super_resolution/data/image/HR"
path_lr = "C:/super_resolution/data/image/LR_8_noise30"
path_sr = "C:/super_resolution/data/image/SR"

path_a = "/A_set"
path_b = "/B_set"
path_fold = path_a

path_train_img = "/train/images"
path_val_img = "/val/images"
path_test_img = "/test/images"

path_log = "C:/super_resolution/log/log_classification/metric_log"
path_metric = path_log + "/metric_LR_A_Reg_8_noise30_two.pt"

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
            label = name.split(".")[0].split("_")[4]
            frame = name.split(".")[0].split("_")[3]

            if label not in self.label_list :
                self.label_frame[label] = []
                self.label_list.append(label)

            self.label_frame[label].append(frame)
            self.label_frame[label].sort()

        self.img_list=[]
        if self.mode == "center" :
            for label in list(self.label_list) :
                _list = self.label_frame[label]
                center_label = _list[0]
                name_frag = "_" + center_label + "_" + label
                for name in self.list_files :
                    if name[-len(name_frag)-4:-4] == name_frag :
                        self.img_list.append(name)
        elif self.mode == "image" :
            for label in list(self.label_list) :
                _list = self.label_frame[label]
                image_label_1 = _list[5]
                image_label_2 = _list[9]
                name_frag_1 = image_label_1 + "_" + label
                name_frag_2 = image_label_2 + "_" + label
                for name in self.list_files :
                    if name[-len(name_frag_1)-4:-4] == name_frag_1 :
                        self.img_list.append(name)
                    if name[-len(name_frag_2)-4:-4] == name_frag_2 :
                        self.img_list.append(name)
        elif self.mode == "all" :
            for label in list(self.label_list):
                _list = self.label_frame[label]
                for i in range(1, 10) :
                    img_label = _list[i]
                    name_frag = "_" + img_label + "_" + label
                    for name in self.list_files:
                        if name[-len(name_frag)-4:-4] == name_frag:
                            self.img_list.append(name)

        self.img_list = set(self.img_list)
        self.img_list = list(self.img_list)
        print(len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        _name = self.img_list[idx]
        pil_img = Image.open(self.path_img + self.path_fold + self.path_data + "/" + _name)
        label = _name.split(".")[0].split("_")[4]

        label_index = list(self.label_frame.keys()).index(label)
        label_tensor = torch.Tensor([label_index]).to(torch.int64)

        return self.transform_raw(pil_img), label_tensor


if __name__ == "__main__":
    # train, valid, test dataset 설정
    dataset_center = Dataset_for_Classification(path_img=path_lr,
                                                path_fold=path_fold,
                                                path_data=path_test_img,
                                                mode="center")
    dataset_image = Dataset_for_Classification(path_img=path_lr,
                                               path_fold=path_fold,
                                               path_data=path_test_img,
                                               mode="image")
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

    model_path = "C:/super_resolution/log/log_classification/make_model/RegDB/model"
    latest_model = os.listdir(model_path)[-1]
    model = torch.load(model_path + "/" + latest_model)
    model.head = nn.Identity()
    model.to(device)

    # train, valid, test
    size = len(dataloader_center.dataset)

    for i_epoch_raw in range(EPOCH):
        i_epoch = i_epoch_raw + 1

    center_dict = {}
    distance_same = []
    distance_diff = []

    # metric 측정을 위한 center 추출
    for i_dataloader in dataloader_center:
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            label = label.item()
            if label not in center_dict.keys():
                center = model(data)
                center_dict[label] = [label, center]

    _label_list = list(center_dict.keys())
    print(len(_label_list))


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

            for center_label in _label_list:
                center_label, center = center_dict[center_label]
                distance = (output - center).pow(2).sum().sqrt().item()
                if center_label == label:
                    distance_same.append(distance)
                else:
                    distance_diff.append(distance)


    distance_same.sort()
    distance_diff.sort()

    torch.save({'distance_same' : distance_same,
                'distance_diff' : distance_diff}, path_metric)