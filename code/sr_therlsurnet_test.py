import os
import random
import argparse
import datetime
import csv

import torch.utils.data as data
import torchvision.transforms as transforms
import ignite
from torchvision.transforms.functional import to_pil_image

from DLCs.data_tools import pil_marginer_v3, pil_augm_lite, imshow_pil
from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.data_record import RecordBox
from DLCs.contextual_loss import *

import torch

from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from DLCs.super_resolution.model_therlsurnet import TherlSuRNet

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 단일 코드로 돌릴 때 사용
_database = "Reg"
_fold = "A"
_scale = 4
_noise = 30
_batch = 16
_csv = True

# Argparse Setting
parser = argparse.ArgumentParser(description = "TherlSuRNet model을 이용해 Super Resolution을 진행합니다. (Test)")

parser.add_argument('--database', required = False, choices = ["Reg", "SYSU"], default = _database, help = "사용할 데이터베이스 입력 (Reg, SYSU)")
parser.add_argument('--fold', required = False, choices = ["A", "B"], default = _fold, help = "학습을 진행할 fold 입력 (A, B)")
parser.add_argument('--scale', required = False, type = int, default = _scale, help = "LR 이미지의 Scale Factor 입력")
parser.add_argument('--noise', required = False, type = int, default = _noise, help = "LR 이미지 noise의 sigma 값 입력")
parser.add_argument("--csv", required = False, action='store_true', help = "csv파일에 기록 여부 선택")
parser.add_argument("--server", required = False, action = 'store_true', help = "neuron 서버로 코드 실행 여부 선택")

args = parser.parse_args()

# Mode Setting
DATABASE = args.database        # Reg or SYSU
FOLD = args.fold
SCALE_FACTOR = args.scale
NOISE = args.noise
CSV = args.csv
SR_MODEL = "TherlSuRNet"
DEVICE = "SERVER" if args.server else "LOCAL"

# # 단일 코드로 돌릴 때의 옵션
# CSV = _csv

# Datapath
if DEVICE == "SERVER" :
    path_device = "/scratch/hpc111a06/syjung/super_resolution"
elif DEVICE == "LOCAL" :
    path_device= "C:/super_resolution"

if DATABASE == "Reg" :
    path_img = path_device + "/data/image/"
elif DATABASE == "SYSU" :
    path_img = "C:/super_resolution/data/image_SYSU/"

path_hr = path_img + "HR"
path_lr = path_img + f"LR_{SCALE_FACTOR}_noise{NOISE}"
path_sr = path_img + f"SR_{SR_MODEL}"

path_fold = f"/{FOLD}_set"

path_train_img = "/train/images"
path_valid_img = "/val/images"
path_test_img = "/test/images"

path_log = path_device + f"/log/log_sr/{SR_MODEL}/{DATABASE}/{FOLD}_set"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dt_now = datetime.datetime.now()
date = str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day) + " " + str(dt_now.hour) + ":" + str(dt_now.minute)

class Dataset_for_SR(data.Dataset):
    def __init__(self, **kwargs):
        self.path_hr = kwargs['path_hr']
        self.path_lr = kwargs['path_lr']
        self.path_fold = kwargs['path_fold']
        self.path_image = kwargs['path_image']

        self.is_test = kwargs['is_test']
        if self.is_test is False:
            self.is_train = kwargs['is_train']  # (bool)

            if self.is_train:
                self.flip_hori = kwargs['flip_hori']  # (bool) 수평 반전
                self.flip_vert = kwargs['flip_vert']  # (bool) 수직 반전
            else:
                self.flip_hori = False
                self.flip_vert = False
            self.scale_factor = kwargs['scale_factor']
            self.size_hr = kwargs['size_hr']  # (w(int), h(int))
            self.size_lr = kwargs['size_lr']

        self.list_files = os.listdir(self.path_hr + self.path_fold + self.path_image)

        self.transform_raw = transforms.Compose([transforms.ToTensor()])  # (transforms.Compose) with no Norm

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        # 이미지 파일 이름 뽑아오기
        _name = self.list_files[idx]

        # Original, LR_4_noise10 이미지 불러오기
        pil_hr = Image.open(self.path_hr + self.path_fold + self.path_image + "/" + _name)
        pil_lr = Image.open(self.path_lr + self.path_fold + self.path_image + "/" + _name)

        # hr_w, hr_h = pil_hr.size
        # lr_w, lr_h = pil_lr.size

        # train일 경우 image crop, data augmentaion 진행
        if self.is_test is False:
            pil_hr_patch, pil_lr_patch = pil_marginer_v3(in_pil_hr=pil_hr,
                                                         target_size_hr=self.size_hr,
                                                         img_background=(0, 0, 0),
                                                         # (선택) 세부옵션 (각각 default 값 있음)
                                                         scaler=1.0,
                                                         is_random=self.is_train,
                                                         itp_opt_img=Image.LANCZOS,
                                                         # 선택 (LR_4_noise10 Image 관련)
                                                         in_pil_lr=pil_lr,
                                                         in_scale_factor=self.scale_factor,
                                                         target_size_lr=self.size_lr)
            # imshow_pil(pil_hr_patch)
            # imshow_pil

            pil_hr_patch, pil_lr_patch = pil_augm_lite(pil_hr_patch, pil_lr_patch,
                                                       self.flip_hori, self.flip_vert, get_info=False)

            # imshow_pil(pil_hr_patch)
            # imshow_pil(pil_lr_patch)

            return self.transform_raw(pil_hr_patch), self.transform_raw(pil_lr_patch), _name

        # valid, test일 경우 원본 데이터 그대로 사용
        else:
            return self.transform_raw(pil_hr), self.transform_raw(pil_lr), _name

if __name__ == "__main__":
    if CSV :
        try :
            log_check = open("C:/super_resolution/log/log_sr/sr_log.csv", "r")
            log_check.close()
            log = open("C:/super_resolution/log/log_sr/sr_log.csv", "a", newline = "")
            log_write = csv.writer(log, delimiter=',')
        except :
            log = open("C:/super_resolution/log/log_sr/sr_log.csv", "a", newline = "")
            log_write = csv.writer(log, delimiter = ',')
            log_write.writerow(["date", "database", "fold", "degrade", "mode", "loss", "psnr", "ssim"])

    # 기본 설정 : device, scaler, model, loss, epoch, batch_size, random_seed, lr, optimizer, scheduler
    # train 코드의 그것을 그대로 배껴주세요
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TherlSuRNet()
    model.to(device)

    criterion_l1 = torch.nn.L1Loss().to(device)
    criterion_cx = ContextualLoss(use_vgg=True).to(device)

    # Ignite를 활용한 PSNR, SSIM 계산을 위한 준비
    def ignite_eval_step(engine, batch):
        return batch

    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')

    checkpoint_path = path_log + "/checkpoint"
    ckpt_list = os.listdir(checkpoint_path)
    print(checkpoint_path + f"/{ckpt_list[-1]}")
    ckpt = torch.load(checkpoint_path + f"/{ckpt_list[-1]}")
    model.load_state_dict(ckpt["model_state_dict"])

    loss_test = RecordBox(name="loss_test", is_print=False)
    psnr_test = RecordBox(name="psnr_test", is_print=False)
    ssim_test = RecordBox(name="ssim_test", is_print=False)

    dataset_test = Dataset_for_SR(path_hr=path_hr,
                                  path_lr=path_lr,
                                  path_fold=path_fold,
                                  path_image=path_test_img,
                                  is_test=True)

    dataloader_test = DataLoader_multi_worker_FIX(dataset=dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  prefetch_factor=2,
                                                  drop_last=False)

    # test
    model.eval()
    for i_dataloader in dataloader_test:
        i_batch_hr, i_batch_lr, i_batch_name = i_dataloader
        i_batch_hr = i_batch_hr.to(device)
        i_batch_lr = i_batch_lr.to(device)

        with torch.no_grad():
            i_batch_sr = model(i_batch_lr)

            ts_hr = torch.clamp(i_batch_hr[0], min=0, max=1).to(device)
            ts_sr = torch.clamp(i_batch_sr[0], min=0, max=1).to(device)  # B C H W
            name = i_batch_name[0]

            # 이미지 저장
            pil_sr = to_pil_image(ts_sr)
            try:
                pil_sr.save(path_sr + path_fold + path_test_img + "/" + name)
            except:
                os.makedirs(path_sr + path_fold + path_test_img)
                pil_sr.save(path_sr + path_fold + path_test_img + "/" + name)

            # PSNR, SSIM 계산
            ts_hr = ts_hr.to(device)
            ts_sr = ts_sr.to(device)

            ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0),
                                                   torch.unsqueeze(ts_hr, 0)
                                                   ]])

            _psnr_test = ignite_result.metrics['psnr']
            _ssim_test = ignite_result.metrics['ssim']
            psnr_test.add_item(_psnr_test)
            ssim_test.add_item(_ssim_test)

            loss_l1 = criterion_l1(i_batch_sr, i_batch_hr)
            loss_ssim = _ssim_test
            loss_cx = criterion_cx(i_batch_sr, i_batch_hr)

            _loss_test = 10 * loss_l1 + 10 * (1 - loss_ssim) + 0.1 * loss_cx
            loss_test.add_item(_loss_test.item())

        loss_test.update_batch()
        psnr_test.update_batch()
        ssim_test.update_batch()

    _lte = loss_test.update_epoch(is_return=True, path=path_log)
    _pte = psnr_test.update_epoch(is_return=True, path=path_log)
    _ste = ssim_test.update_epoch(is_return=True, path=path_log)

    if CSV:
        log_write.writerow([date, SR_MODEL, DATABASE, FOLD, f"{SCALE_FACTOR}_{NOISE}", "test", _lte, _pte, _ste])
        log.close()

    print("<Test Result> loss {}, psnr {}, ssim {}".format(_lte, _pte, _ste))