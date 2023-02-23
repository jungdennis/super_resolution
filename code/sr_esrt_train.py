import os
import random
import argparse
import csv
import datetime

import torch.utils.data as data
import torchvision.transforms as transforms
import ignite
from torchvision.transforms.functional import to_pil_image

from DLCs.data_tools import pil_marginer_v3, pil_augm_lite, imshow_pil
from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.data_record import RecordBox
from DLCs.sr_tools import graph_loss, graph_single

import torch

from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from DLCs.super_resolution.model_esrt import ESRT

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
_epoch = 300
_batch = 16
_csv = True
_load = False

# Argparse Setting
parser = argparse.ArgumentParser(description = "ESRT model을 이용해 Super Resolution을 진행합니다. (Train)")

parser.add_argument('--database', required = False, choices = ["Reg", "SYSU"], default = _database, help = "사용할 데이터베이스 입력 (Reg, SYSU)")
parser.add_argument('--fold', required = False, choices = ["A", "B"], default = _fold, help = "학습을 진행할 fold 입력 (A, B)")
parser.add_argument('--scale', required = False, type = int, default = _scale, help = "LR 이미지의 Scale Factor 입력")
parser.add_argument('--noise', required = False, type = int, default = _noise, help = "LR 이미지 noise의 sigma 값 입력")
parser.add_argument('--epoch', required = False, type = int, default = _epoch, help = "학습을 진행할 epoch 수 입력")
parser.add_argument('--batch', required = False, type = int, default = _batch, help = "학습을 진행할 batch size 입력")
parser.add_argument("--csv", required = False, action = 'store_true', help = "csv파일에 기록 여부 선택 (True, False)")
parser.add_argument("--load", required = False, action = 'store_true', help = "이전 학습 기록 load 여부 선택 (True, False)")

args = parser.parse_args()

# Mode Setting
DATABASE = args.database        # Reg or SYSU
FOLD = args.fold
SCALE_FACTOR = args.scale
NOISE = args.noise
EPOCH = args.epoch
BATCH_SIZE = args.batch
CSV = args.csv
LOAD = args.load
SR_MODEL = "ESRT"

# # 단일 코드로 돌릴 때의 옵션
# CSV = _csv
# TEST = _test
# LOAD = _load

# Datapath
if DATABASE == "Reg" :
    path_img = "C:/super_resolution/data/image/"
elif DATABASE == "SYSU" :
    path_img = "C:/super_resolution/data/image_SYSU/"

path_hr = path_img + "HR"
path_lr = path_img + f"LR_{SCALE_FACTOR}_noise{NOISE}"
path_sr = path_img + f"SR_{SR_MODEL}"

path_fold = f"/{FOLD}_set"

path_train_img = "/train/images"
path_valid_img = "/val/images"
path_test_img = "/test/images"

path_log = f"C:/super_resolution/log/log_sr/{SR_MODEL}/{DATABASE}/{FOLD}_set"

option_frag = f"{SR_MODEL}_{DATABASE}_fold {FOLD}"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dt_now = datetime.datetime.now()
date = str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day) + " " + str(dt_now.hour) + ":" + str(dt_now.minute)

# Dataset settings
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

        # HR, LR 이미지 불러오기
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


# train, valid, test
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
            log_write.writerow(["date", "model", "database", "fold", "degrade", "mode", "loss", "psnr", "ssim"])

    # 기본 설정 : device, scaler, model, loss, epoch, batch_size, random_seed, lr, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)
    model = ESRT(upscale=4)
    model.to(device)
    criterion = torch.nn.L1Loss()
    LR = 2e-4

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=50,
                                                gamma=0.5)

    # Ignite를 활용한 PSNR, SSIM 계산을 위한 준비
    def ignite_eval_step(engine, batch):
        return batch

    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')

    loss_train = RecordBox(name="loss_train", is_print=False)
    psnr_train = RecordBox(name="psnr_train", is_print=False)
    ssim_train = RecordBox(name="ssim_train", is_print=False)

    loss_valid = RecordBox(name="loss_valid", is_print=False)
    psnr_valid = RecordBox(name="psnr_valid", is_print=False)
    ssim_valid = RecordBox(name="ssim_valid", is_print=False)

    lr = RecordBox(name="learning_rate", is_print=False)

    lr_list = []

    loss_train_list = []
    psnr_train_list = []
    ssim_train_list = []

    loss_valid_list = []
    psnr_valid_list = []
    ssim_valid_list = []

    # train, valid dataset 설정
    dataset_train = Dataset_for_SR(path_hr=path_hr,
                                   path_lr=path_lr,
                                   path_fold=path_fold,
                                   path_image=path_train_img,
                                   is_train=True,
                                   is_test=False,
                                   flip_hori=True,
                                   flip_vert=False,
                                   scale_factor=4,
                                   size_hr=(192, 192),
                                   size_lr=(48, 48))

    dataset_valid = Dataset_for_SR(path_hr=path_hr,
                                   path_lr=path_lr,
                                   path_fold=path_fold,
                                   path_image=path_valid_img,
                                   is_train=False,
                                   is_test=False,
                                   scale_factor=4,
                                   size_hr=(192, 192),
                                   size_lr=(48, 48))

    '''
    dataloader_train = torch.utils.data.DataLoader(dataset     = dataset_train
                                                  ,batch_size  = 4
                                                  ,shuffle     = True
                                                  ,num_workers = 0
                                                  ,prefetch_factor = 2
                                                  ,drop_last = True
                                                  )
    '''

    # dataset을 dataloader에 할당
    dataloader_train = DataLoader_multi_worker_FIX(dataset=dataset_train,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=True)

    dataloader_valid = DataLoader_multi_worker_FIX(dataset=dataset_valid,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=False)

    # checkpoint 저장 log 불러오기
    if LOAD :
        checkpoint_path = path_log + "/checkpoint"
        ckpt_list = os.listdir(checkpoint_path)
        ckpt = torch.load(checkpoint_path + f"/{ckpt_list[-1]}")

        i_epoch = ckpt["epoch"]

        try:
            lr_list = ckpt["lr"]
        except:
            pass

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        loss_train_list = ckpt["loss_train"]
        psnr_train_list = ckpt["psnr_train"]
        ssim_train_list = ckpt["ssim_train"]

        loss_valid_list = ckpt["loss_valid"]
        psnr_valid_list = ckpt["psnr_valid"]
        ssim_valid_list = ckpt["ssim_valid"]

    else :
        i_epoch = 0

    # train, valid
    size = len(dataloader_train.dataset)
    for i_epoch_raw in range(EPOCH):
        i_epoch += 1

        if i_epoch > EPOCH :
            break

        print(f"<epoch {i_epoch}>")

        # train
        optimizer.zero_grad()
        model.train()
        for batch, i_dataloader in enumerate(dataloader_train):
            i_batch_hr, i_batch_lr, i_batch_name = i_dataloader
            i_batch_hr = i_batch_hr.to(device)
            i_batch_lr = i_batch_lr.to(device)

            i_batch_lr = i_batch_lr.requires_grad_(True)

            i_batch_sr = model(i_batch_lr)

            _loss_train = criterion(i_batch_sr, i_batch_hr)
            loss_train.add_item(_loss_train.item())

            amp_scaler.scale(_loss_train).backward(retain_graph=False)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()

            with torch.no_grad():
                for i_batch in range(BATCH_SIZE):
                    ts_hr = torch.clamp(i_batch_hr[i_batch], min=0, max=1).to(device)
                    ts_sr = torch.clamp(i_batch_sr[i_batch], min=0, max=1).to(device)  # B C H W
                    name = i_batch_name[i_batch]

                    # 이미지 저장
                    pil_sr = to_pil_image(ts_sr)
                    try :
                        pil_sr.save(path_sr + path_fold + path_train_img + "/" + name)
                    except :
                        os.makedirs(path_sr + path_fold + path_train_img)
                        pil_sr.save(path_sr + path_fold + path_train_img + "/" + name)

                    # PSNR, SSIM 계산
                    ts_hr = ts_hr.to(device)
                    ts_sr = ts_sr.to(device)

                    ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0)
                                                           ,torch.unsqueeze(ts_hr, 0)
                                                           ]])

                    _psnr_train = ignite_result.metrics['psnr']
                    _ssim_train = ignite_result.metrics['ssim']
                    psnr_train.add_item(_psnr_train)
                    ssim_train.add_item(_ssim_train)

                if batch % 30 == 0:
                    loss = _loss_train.item()
                    current = batch * len(i_batch_lr)
                    print(f"loss: {loss}  [{current}/{size}]")

            loss_train.update_batch()
            psnr_train.update_batch()
            ssim_train.update_batch()

        lr.add_item(scheduler.get_last_lr()[0])
        scheduler.step()
        lr.update_batch()

        # valid
        model.eval()
        for i_dataloader in dataloader_valid:
            i_batch_hr, i_batch_lr, i_batch_name = i_dataloader
            i_batch_hr = i_batch_hr.to(device)
            i_batch_lr = i_batch_lr.to(device)

            with torch.no_grad():
                i_batch_sr = model(i_batch_lr)
                _loss_valid = criterion(i_batch_sr, i_batch_hr)
                loss_valid.add_item(_loss_valid.item())

                ts_hr = torch.clamp(i_batch_hr[0], min=0, max=1).to(device)
                ts_sr = torch.clamp(i_batch_sr[0], min=0, max=1).to(device)    # B C H W
                name = i_batch_name[0]

                # 이미지 저장
                pil_sr = to_pil_image(ts_sr)
                try :
                    pil_sr.save(path_sr + path_fold + path_valid_img + "/" + name)
                except :
                    os.makedirs(path_sr + path_fold + path_valid_img)
                    pil_sr.save(path_sr + path_fold + path_valid_img + "/" + name)

                #PSNR, SSIM 계산
                ts_hr = ts_hr.to(device)
                ts_sr = ts_sr.to(device)

                ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0)
                                                    ,torch.unsqueeze(ts_hr, 0)
                                                    ]])

                _psnr_valid = ignite_result.metrics['psnr']
                _ssim_valid = ignite_result.metrics['ssim']
                psnr_valid.add_item(_psnr_valid)
                ssim_valid.add_item(_ssim_valid)

            loss_valid.update_batch()
            psnr_valid.update_batch()
            ssim_valid.update_batch()

        _lt = loss_train.update_epoch(is_return=True, path = path_log)
        _pt = psnr_train.update_epoch(is_return=True, path = path_log)
        _st = ssim_train.update_epoch(is_return=True, path = path_log)
        _lv = loss_valid.update_epoch(is_return=True, path = path_log)
        _pv = psnr_valid.update_epoch(is_return=True, path = path_log)
        _sv = ssim_valid.update_epoch(is_return=True, path = path_log)
        _lr = lr.update_epoch(is_return=True,  path = path_log)

        lr_list.append(_lr)
        loss_train_list.append(_lt)
        psnr_train_list.append(_pt)
        ssim_train_list.append(_st)
        loss_valid_list.append(_lv)
        psnr_valid_list.append(_pv)
        ssim_valid_list.append(_sv)

        if i_epoch % 10 == 0 :
            if i_epoch < 100 :
                epoch_save = f"0{i_epoch}"
            else :
                epoch_save = f"{i_epoch}"

            try :
                torch.save({
                    'epoch': i_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'lr': lr_list,
                    'loss_train': loss_train_list,
                    'psnr_train': psnr_train_list,
                    'ssim_train': ssim_train_list,
                    'loss_valid': loss_valid_list,
                    'psnr_valid': psnr_valid_list,
                    'ssim_valid': ssim_valid_list,
                }, path_log + f"/checkpoint/epoch_{epoch_save}.pt")
            except :
                os.makedirs(path_log + "/checkpoint")
                torch.save({
                    'epoch': i_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'lr': lr_list,
                    'loss_train': loss_train_list,
                    'psnr_train': psnr_train_list,
                    'ssim_train': ssim_train_list,
                    'loss_valid': loss_valid_list,
                    'psnr_valid': psnr_valid_list,
                    'ssim_valid': ssim_valid_list,
                }, path_log + f"/checkpoint/epoch_{epoch_save}.pt")

        print("train : loss {}, psnr {}, ssim : {}".format(_lt, _pt, _st))
        print("valid : loss {}, psnr {}, ssim : {}".format(_lv, _pv, _sv))
        print("------------------------------------------------------------------------")

    graph_loss(loss_train_list, loss_valid_list, save = path_log + f"/loss_{option_frag}.png",
               title = f"Graph of Loss ({option_frag})")

    try:
        graph_single(lr_list, "lr", save=path_log + f"/lr_{option_frag}.png",
                     title=f"Graph of LR ({option_frag})")
    except:
        pass

    graph_single(psnr_train_list, "PSNR", save = path_log + f"/psnr_train_{option_frag}.png",
                 title = f"Graph of PSNR_train ({option_frag})", print_max = True)
    graph_single(psnr_valid_list, "PSNR", save = path_log + f"/psnr_valid_{option_frag}.png",
                 title = f"Graph of PSNR_valid ({option_frag})", print_max = True)
    graph_single(ssim_train_list, "SSIM", save = path_log + f"/ssim_train_{option_frag}.png",
                 title = f"Graph of SSIM_train ({option_frag})", print_max = True)
    graph_single(ssim_valid_list, "SSIM", save = path_log + f"/ssim_valid_{option_frag}.png",
                 title = f"Graph of SSIM_valid ({option_frag})", print_max = True)

    if CSV:
        log_write.writerow([date, SR_MODEL, DATABASE, FOLD, f"{SCALE_FACTOR}_{NOISE}", "train", min(loss_train_list), max(psnr_train_list), max(ssim_train_list)])
        log_write.writerow([date, SR_MODEL, DATABASE, FOLD, f"{SCALE_FACTOR}_{NOISE}", "valid", min(loss_valid_list), max(psnr_valid_list), max(ssim_valid_list)])
        log.close()

    print("training complete! - check the log and go to test session")