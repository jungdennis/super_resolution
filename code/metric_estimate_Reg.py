import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
import csv
import sys

import cv2
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms

from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.metric_tools import metric_histogram, calc_FAR_FRR_v2, calc_EER, graph_FAR_FRR

import argparse

# 단일 코드로 돌릴 때 사용
_model = "densenet"
_resolution = "HR"
_scale = 4
_noise = 30
_sr_model = "IMDN"
_mode = "all"

# Argparse Setting
parser = argparse.ArgumentParser(description = "RegDB의 Distance와 FAR, FRR, EER을 측정합니다.")

parser.add_argument("--model", required = False, choices = ["convnext", "inception", "densenet"], default = _model, help = "이미지 분류 모델 선택 (convnext, inception)")
parser.add_argument('--resolution', required = False, choices = ["HR", "LR", "SR"], default = _resolution, help = "이미지의 Resolution 선택 (HR, LR, SR)")
parser.add_argument('--scale', required = False, type = int, default = _scale, help = "LR 이미지의 Scale Factor 입력")
parser.add_argument('--noise', required = False, type = int, default = _noise, help = "LR 이미지 noise의 sigma 값 입력")
parser.add_argument('--sr', required = False, default = _sr_model, help = "SR 이미지의 알고리즘 이름 입력")
parser.add_argument("--mode", required = False, default = _mode, choices = ["all", "pick_1", "pick_2"],  help = "RegDB의 이미지 선택 모드 입력")
parser.add_argument("--csv", required = False, action = "store_true", help = "csv파일에 기록 여부 선택 (True, False)")

args = parser.parse_args()

# Mode Setting
MODEL = args.model
RESOLUTION = args.resolution
SCALE_FACTOR = args.scale
NOISE = args.noise
SR_MODEL = args.sr
MEASURE_MODE = args.mode        # all or pick_1 or pick_2
CSV = args.csv

# Datapath
if RESOLUTION == "HR" :
    path_img = "C:/super_resolution/data/image/HR"
    path_log = f"C:/super_resolution/log/log_metric/graph_and_log/Reg/{MODEL}/HR/"
    option_frag = f"HR_{MEASURE_MODE}_{MODEL}"
elif RESOLUTION == "LR" :
    path_img = f"C:/super_resolution/data/image/LR_{SCALE_FACTOR}_noise{NOISE}"
    path_log = f"C:/super_resolution/log/log_metric/graph_and_log/Reg/{MODEL}/LR_{SCALE_FACTOR}_{NOISE}/"
    option_frag = f"LR_{SCALE_FACTOR}_{NOISE}_{MEASURE_MODE}_{MODEL}"
elif RESOLUTION == "SR" :
    path_img = f"C:/super_resolution/data/image/SR_{SR_MODEL}"
    path_log = f"C:/super_resolution/log/log_metric/graph_and_log/Reg/{MODEL}/SR_{SR_MODEL}/"
    option_frag = f"SR_{SR_MODEL}_{MEASURE_MODE}_{MODEL}"

path_a = "/A_set"
path_b = "/B_set"

path_train_img = "/train/images"
path_val_img = "/val/images"
path_test_img = "/test/images"

dt_now = datetime.datetime.now()
date = str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day) + " " + str(dt_now.hour) + ":" + str(dt_now.minute)

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
        elif self.mode == "pick_1" :
            for label in list(self.label_list) :
                _list = self.label_frame[label]
                image_label_1 = _list[5]
                name_frag_1 = image_label_1 + "_" + label

                for name in self.list_files :
                    if name[-len(name_frag_1)-4:-4] == name_frag_1 :
                        self.img_list.append(name)
        elif self.mode == "pick_2" :
            for label in list(self.label_list) :
                _list = self.label_frame[label]
                image_label_1 = _list[5]
                name_frag_1 = image_label_1 + "_" + label
                image_label_2 = _list[9]
                name_frag_2 = image_label_2 + "_" + label

                for name in self.list_files :
                    if name[-len(name_frag_1)-4:-4] == name_frag_1 :
                        self.img_list.append(name)
                    if name[-len(name_frag_2) - 4:-4] == name_frag_2:
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
    # 로그 저장 설정
    if CSV :
        try :
            log_check = open(f"C:/super_resolution/log/log_metric/graph_and_log/Reg/calc_log.csv", "r")
            log_check.close()
            log = open(f"C:/super_resolution/log/log_metric/graph_and_log/Reg/calc_log.csv", "a", newline = "")
            log_write = csv.writer(log, delimiter=',')
        except :
            log = open(f"C:/super_resolution/log/log_metric/graph_and_log/Reg/calc_log.csv", "a", newline = "")
            log_write = csv.writer(log, delimiter = ',')
            log_write.writerow(["date", "option", "mean_genu", "std_genu", "mean_impo", "std_impo", "th_EER", "EER"])

    # 학습 설정
    # device, scaler, model, loss, epoch, batch_size, lr, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

    model_path = f"C:/super_resolution/code/DLCs/metric/metric_Reg_{MODEL}.pt"
    model = torch.load(model_path)

    if MODEL == "convnext" :
        model.head = nn.Identity()
    elif MODEL == "densenet" :
        model.classifier = nn.Identity()
    elif MODEL == "inception" :
        model.last_linear = nn.Identity()

    model.to(device)
    model.eval()

    loss_mse = nn.MSELoss(reduction='sum')
    loss_mse.to(device)

    '''
    Fold A
    '''
    print(f"Start Measure Metric : Fold A ({MEASURE_MODE})")
    # train, valid, test dataset 설정
    dataset_center_a = Dataset_for_Classification(path_img=path_img,
                                                  path_fold=path_a,
                                                  path_data=path_test_img,
                                                  mode="center")
    dataset_image_a = Dataset_for_Classification(path_img=path_img,
                                                 path_fold=path_a,
                                                 path_data=path_test_img,
                                                 mode=MEASURE_MODE)

    num_img_a = len(dataset_image_a)
    print(f"Fold A : dataset_center : {len(dataset_center_a.img_list)}, dataset_image : {len(dataset_image_a.img_list)}")


    # dataset을 dataloader에 할당
    # 원래는 torch의 dataloader를 부르는게 맞지만
    # 멀티코어 활용을 위해 DataLoader_multi_worker_FIX를 import 하여 사용
    dataloader_center_a = DataLoader_multi_worker_FIX(dataset=dataset_center_a,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    prefetch_factor=2,
                                                    drop_last=True)

    dataloader_image_a = DataLoader_multi_worker_FIX(dataset=dataset_image_a,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=True)



    # train, valid, test
    center_list_A = []
    distance_same_A = []
    distance_diff_A = []

    # metric 측정을 위한 center 추출
    for i_dataloader in dataloader_center_a:
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            label = label.item()
            center = model(data)
            center_list_A.append([label, center])

    print(len(center_list_A))

    # metric 측정을 위한 feature 추출
    cnt_feature = 1
    cnt_pass = 0
    for i_dataloader in dataloader_image_a:
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            label = label.item()
            output = model(data)
            print(f"Calculating Feature and Distance... [{cnt_feature}/{num_img_a}]")
            cnt_feature += 1

            for center_label, center in center_list_A:
                distance = (output - center).pow(2).sum().sqrt().item()
                if center_label == label:
                    distance_same_A.append(distance)
                else:
                    distance_diff_A.append(distance)

    distance_same_A.sort()
    distance_diff_A.sort()

    try :
        torch.save({'distance_same' : distance_same_A,
                    'distance_diff' : distance_diff_A}, path_log + f"metric_Reg_A_{option_frag}.pt")
    except :
        os.makedirs(path_log)
        torch.save({'distance_same': distance_same_A,
                    'distance_diff': distance_diff_A}, path_log + f"metric_Reg_A_{option_frag}.pt")

    '''
    Fold B
    '''
    print(f"Start Measure Metric : Fold B ({MEASURE_MODE})")
    dataset_center_b = Dataset_for_Classification(path_img=path_img,
                                                  path_fold=path_b,
                                                  path_data=path_test_img,
                                                  mode="center")
    dataset_image_b = Dataset_for_Classification(path_img=path_img,
                                                 path_fold=path_b,
                                                 path_data=path_test_img,
                                                 mode=MEASURE_MODE)

    num_img_b = len(dataset_image_b)
    print(f"Fold B : dataset_center : {len(dataset_center_b.img_list)}, dataset_image : {len(dataset_image_b.img_list)}")

    dataloader_center_b = DataLoader_multi_worker_FIX(dataset=dataset_center_b,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      prefetch_factor=2,
                                                      drop_last=True)

    dataloader_image_b = DataLoader_multi_worker_FIX(dataset=dataset_image_b,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     num_workers=2,
                                                     prefetch_factor=2,
                                                     drop_last=True)

    center_list_B = []
    distance_same_B = []
    distance_diff_B = []

    # metric 측정을 위한 center 추출
    for i_dataloader in dataloader_center_b:
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            label = label.item()
            center = model(data)
            center_list_B.append([label, center])

    print(len(center_list_B))

    # metric 측정을 위한 feature 추출
    cnt_feature = 1
    cnt_pass = 0
    for i_dataloader in dataloader_image_b:
        data, label = i_dataloader
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            label = label.item()
            output = model(data)
            print(f"Calculating Feature and Distance... [{cnt_feature}/{num_img_b}]")
            cnt_feature += 1

            for center_label, center in center_list_B:
                distance = (output - center).pow(2).sum().sqrt().item()
                if center_label == label:
                    distance_same_B.append(distance)
                else:
                    distance_diff_B.append(distance)

    distance_same_B.sort()
    distance_diff_B.sort()

    try:
        torch.save({'distance_same': distance_same_B,
                    'distance_diff': distance_diff_B}, path_log + f"metric_Reg_B_{option_frag}.pt")
    except:
        os.makedirs(path_log)
        torch.save({'distance_same': distance_same_B,
                    'distance_diff': distance_diff_B}, path_log + f"metric_Reg_B_{option_frag}.pt")

    # FAR, FRR, EER 측정
    print("Start Calulating FAR/FRR/EER")
    distance_same = distance_same_A + distance_same_B
    distance_diff = distance_diff_A + distance_diff_B

    distance_same.sort()
    distance_diff.sort()

    distance_same = np.array(distance_same)
    distance_diff = np.array(distance_diff)

    mean_genu = distance_same.mean()
    std_genu = distance_same.std()

    mean_impo = distance_diff.mean()
    std_impo = distance_diff.std()

    metric_histogram(distance_same, distance_diff, density = True,
                     title=f"Distribution of Distance (Reg_{option_frag})",
                     save_path = path_log + f"hist_Reg_{option_frag}.png")
    # print(len(distance_same), len(distance_diff))

    threshold, FAR, FRR = calc_FAR_FRR_v2(distance_same, distance_diff, save = path_log + f"FAR_FRR_Reg_{option_frag}.pt")
    EER, th = calc_EER(threshold, FAR, FRR, save = path_log + f"EER_Reg_{option_frag}.pt")

    graph_FAR_FRR(threshold, FAR, FRR, show_EER = True,
                  title = f"Graph of FAR & FRR (Reg_{option_frag})",
                  save_path = path_log + f"graph_EER_Reg_{option_frag}.png")
    graph_FAR_FRR(threshold, FAR, FRR, show_EER = True, log=True,
                  title = f"Graph of FAR & FRR (Reg_{option_frag})",
                  save_path = path_log + f"graph_EER_Reg_{option_frag}_log.png")
    if CSV :
        log_write.writerow([date, option_frag, mean_genu, std_genu, mean_impo, std_impo, th, EER])
        log.close()