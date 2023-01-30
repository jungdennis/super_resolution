import os
import random

import torch.utils.data as data
import torchvision.transforms as transforms
import ignite
from transformers import get_cosine_schedule_with_warmup
from torchvision.transforms.functional import to_pil_image

from DLCs.data_tools import pil_marginer_v3, pil_augm_lite, imshow_pil
from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.data_record import RecordBox

import torch

from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from DLCs.super_resolution.model_bsrn import BSRN

# Data path
path_hr = "C:/super_resolution/data/image/HR"
path_lr = "C:/super_resolution/data/image/LR"
path_sr = "C:/super_resolution/data/image/SR"

path_a = "/A_set"
path_b = '/B_set'
path_fold = path_a

path_train_img = "/train/images"
path_valid_img = "/val/images"
path_test_img = "/test/images"

path_model = "/BSRN"

path_log = "C:/super_resolution/log/log_sr"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
                self.rotate = kwargs['rotate']        # (bool) 회전
            else:
                self.flip_hori = False
                self.flip_vert = False
                self.rotate = False
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

        # Original, LR 이미지 불러오기
        pil_hr = Image.open(self.path_hr + self.path_fold + self.path_image + "/" + _name)
        pil_lr = Image.open(self.path_lr + self.path_fold + self.path_image + "/" + _name)

        # hr_w, hr_h = pil_hr.size
        # lr_w, lr_h = pil_lr.size

        # train일 경우 image crop, data augmentaion 진행
        if self.is_test is False:
            pil_hr_patch, pil_lr_patch = pil_marginer_v3(in_pil_hr = pil_hr,
                                                         target_size_hr = self.size_hr,
                                                         img_background = (0, 0, 0),
                                                         # (선택) 세부옵션 (각각 default 값 있음)
                                                         scaler = 1.0,
                                                         is_random = self.is_train,
                                                         itp_opt_img = Image.LANCZOS,
                                                         # 선택 (LR Image 관련)
                                                         in_pil_lr = pil_lr,
                                                         in_scale_factor = self.scale_factor,
                                                         target_size_lr = self.size_lr)
            # imshow_pil(pil_hr_patch)
            # imshow_pil

            pil_hr_patch, pil_lr_patch = pil_augm_lite(pil_hr_patch,
                                                       pil_lr_patch,
                                                       self.flip_hori,
                                                       self.flip_vert,
                                                       get_info = False)

            # imshow_pil(pil_hr_patch)
            # imshow_pil(pil_lr_patch)

            return self.transform_raw(pil_hr_patch), self.transform_raw(pil_lr_patch), _name

        # valid, test일 경우 원본 데이터 그대로 사용
        else:
            return self.transform_raw(pil_hr), self.transform_raw(pil_lr), _name


# train, valid, test
if __name__ == "__main__":
    # 기본 설정 : device, scaler, model, loss, epoch, batch_size, random_seed, lr, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)
    model = BSRN(upscale=4)
    model.to(device)
    criterion = torch.nn.L1Loss()
    HP_LR = 1e-3
    HP_EPOCH = 400
    HP_BATCH = 16
    HP_SEED = 485
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HP_LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = HP_EPOCH)
    # random seed 고정
    random.seed(HP_SEED)
    np.random.seed(HP_SEED)
    torch.manual_seed(HP_SEED)

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
                                   rotate=True,
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

    # dataset을 dataloader에 할당
    dataloader_train = DataLoader_multi_worker_FIX(dataset=dataset_train,
                                                   batch_size=HP_BATCH,
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

    # train, valid
    size = len(dataloader_train.dataset)
    for i_epoch_raw in range(HP_EPOCH):
        i_epoch = i_epoch_raw + 1
        print("<epoch {}>".format(i_epoch))

        # train
        optimizer.zero_grad()
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
                for i_batch in range(HP_BATCH):
                    ts_hr = torch.clamp(i_batch_hr[i_batch], min=0, max=1).to(device)
                    ts_sr = torch.clamp(i_batch_sr[i_batch], min=0, max=1).to(device)  # B C H W
                    name = i_batch_name[i_batch]

                    # 이미지 저장
                    pil_sr = to_pil_image(ts_sr)
                    try:
                        pil_sr.save(path_sr + path_model + path_fold + path_train_img + "/" + name)
                    except:
                        os.makedirs(path_sr + path_model + path_fold + path_train_img)
                        pil_sr.save(path_sr + path_model + path_fold + path_train_img + "/" + name)

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
                    pil_sr.save(path_sr + path_model + path_fold + path_valid_img + "/" + name)
                except :
                    os.makedirs(path_sr + path_model + path_fold + path_valid_img)
                    pil_sr.save(path_sr + path_model + path_fold + path_valid_img + "/" + name)

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

        _lt = loss_train.update_epoch(is_return=True, path = path_log + path_model)
        _pt = psnr_train.update_epoch(is_return=True, path = path_log + path_model)
        _st = ssim_train.update_epoch(is_return=True, path = path_log + path_model)
        _lv = loss_valid.update_epoch(is_return=True, path = path_log + path_model)
        _pv = psnr_valid.update_epoch(is_return=True, path = path_log + path_model)
        _sv = ssim_valid.update_epoch(is_return=True, path = path_log + path_model)
        _lr = lr.update_epoch(is_return=True,  path = path_log + path_model)

        lr_list.append(_lr)
        loss_train_list.append(_lt)
        psnr_train_list.append(_pt)
        ssim_train_list.append(_st)
        loss_valid_list.append(_lv)
        psnr_valid_list.append(_pv)
        ssim_valid_list.append(_sv)

        try :
            torch.save({
                'epoch': i_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_train': loss_train_list,
                'psnr_train': psnr_train_list,
                'ssim_train': ssim_train_list,
                'loss_valid': loss_valid_list,
                'psnr_valid': psnr_valid_list,
                'ssim_valid': ssim_valid_list,
            }, path_log + path_model + f"/checkpoint/epoch{i_epoch}.pt")
        except :
            os.makedirs(path_log + path_model + "/checkpoint")
            torch.save({
                'epoch': i_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_train': loss_train_list,
                'psnr_train': psnr_train_list,
                'ssim_train': ssim_train_list,
                'loss_valid': loss_valid_list,
                'psnr_valid': psnr_valid_list,
                'ssim_valid': ssim_valid_list,
            }, path_log + path_model + f"/checkpoint/epoch{i_epoch}.pt")

        print("train : loss {}, psnr {}, ssim : {}".format(_lt, _pt, _st))
        print("valid : loss {}, psnr {}, ssim : {}".format(_lv, _pv, _sv))
        print("------------------------------------------------------------------------")
    print("training complete! - check the log and go to test session")