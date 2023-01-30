import os
import random

import torch.utils.data as data
import torchvision.transforms as transforms
import ignite
from torchvision.transforms.functional import to_pil_image

from DLCs.data_tools import pil_marginer_v3, pil_augm_lite, imshow_pil
from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.data_record import RecordBox

import torch

from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from DLCs.super_resolution.model_imdn import IMDN

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

path_model = "/IMDN"

path_log = "C:/super_resolution/log/log_sr"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

        # Original, LR 이미지 불러오기
        pil_hr = Image.open(self.path_hr + self.path_fold + self.path_image + "/" + _name)
        pil_lr = Image.open(self.path_lr + self.path_fold + self.path_image + "/" + _name)

        # hr_w, hr_h = pil_hr.size
        # lr_w, lr_h = pil_lr.size

        # train일 경우 image crop, data augmentaion 진행
        if self.is_test is False:
            pil_hr_patch, pil_lr_patch = pil_marginer_v3(in_pil_hr=pil_hr
                                                         , target_size_hr=self.size_hr
                                                         , img_background=(0, 0, 0)
                                                         # (선택) 세부옵션 (각각 default 값 있음)
                                                         , scaler=1.0
                                                         , is_random=self.is_train
                                                         , itp_opt_img=Image.LANCZOS
                                                         # 선택 (LR Image 관련)
                                                         , in_pil_lr=pil_lr
                                                         , in_scale_factor=self.scale_factor
                                                         , target_size_lr=self.size_lr
                                                         )
            # imshow_pil(pil_hr_patch)
            # imshow_pil

            pil_hr_patch, pil_lr_patch = pil_augm_lite(pil_hr_patch
                                                       , pil_lr_patch
                                                       , self.flip_hori
                                                       , self.flip_vert
                                                       , get_info=False
                                                       )

            # imshow_pil(pil_hr_patch)
            # imshow_pil(pil_lr_patch)

            return self.transform_raw(pil_hr_patch), self.transform_raw(pil_lr_patch), _name

        # valid, test일 경우 원본 데이터 그대로 사용
        else:
            return self.transform_raw(pil_hr), self.transform_raw(pil_lr), _name

if __name__ == "__main__":
    # 기본 설정 : device, scaler, model, loss, epoch, batch_size, random_seed, lr, optimizer, scheduler
    # train 코드의 그것을 그대로 배껴주세요
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)
    model = IMDN(upscale=4)
    model.to(device)
    criterion = torch.nn.L1Loss()
    HP_LR = 2e-4
    HP_EPOCH = 400
    HP_BATCH = 16
    HP_SEED = 485
    optimizer = torch.optim.Adam(model.parameters()
                                 , lr=HP_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer
                                                , step_size=50
                                                , gamma=0.5)

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

    # checkpoint 저장 log 불러오기
    '''
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
        })
    '''
    checkpoint_path = path_log + path_model + f"/checkpoint/epoch{HP_EPOCH}.pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    '''
    loss_train = checkpoint["loss_train"]
    psnr_train = checkpoint["psnr_train"]
    ssim_train = checkpoint["ssim_train"]
    loss_valid = checkpoint["loss_valid"]
    psnr_valid = checkpoint["psnr_valid"]
    ssim_valid = checkpoint["ssim_valid"]

    plt.plot(range(1, HP_EPOCH+1), loss_train, range(1, HP_EPOCH+1), loss_valid)
    plt.title("Loss (Fold A)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "valid"])
    plt.show()
    plt.savefig("./loss_a.png")

    plt.plot(range(1, HP_EPOCH + 1), psnr_train, range(1, HP_EPOCH + 1), psnr_valid)
    plt.title("PSNR (Fold A)")
    plt.xlabel("epoch")
    plt.ylabel("PSNR")
    plt.legend(["train", "valid"])
    plt.show()
    plt.savefig("./psnr_a.png")

    plt.plot(range(1, HP_EPOCH + 1), ssim_train, range(1, HP_EPOCH + 1), ssim_valid)
    plt.title("SSIM (Fold A)")
    plt.xlabel("epoch")
    plt.ylabel("SSIM")
    plt.legend(["train", "valid"])
    plt.show()
    plt.savefig("./ssim_a.png")
    '''

    loss_test = RecordBox(name="loss_test", is_print=False)
    psnr_test = RecordBox(name="psnr_test", is_print=False)
    ssim_test = RecordBox(name="ssim_test", is_print=False)

    dataset_test = Dataset_for_SR(path_hr=path_hr,
                                  path_lr=path_lr,
                                  path_fold=path_fold,
                                  path_image=path_test_img,
                                  is_test=True)

    dataloader_test = DataLoader_multi_worker_FIX(dataset=dataset_test
                                                  , batch_size=1
                                                  , shuffle=False
                                                  , num_workers=2
                                                  , prefetch_factor=2
                                                  , drop_last=False)

    # test
    for i_dataloader in dataloader_test:
        i_batch_hr, i_batch_lr, i_batch_name = i_dataloader
        i_batch_hr = i_batch_hr.to(device)
        i_batch_lr = i_batch_lr.to(device)

        with torch.no_grad():
            i_batch_sr = model(i_batch_lr)
            _loss_test = criterion(i_batch_sr, i_batch_hr)
            loss_test.add_item(_loss_test.item())

            ts_hr = torch.clamp(i_batch_hr[0], min=0, max=1).to(device)
            ts_sr = torch.clamp(i_batch_sr[0], min=0, max=1).to(device)  # B C H W
            name = i_batch_name[0]

            # 이미지 저장
            pil_sr = to_pil_image(ts_sr)
            pil_sr.save(path_sr + path_model + path_fold + path_test_img + "/" + name)

            # PSNR, SSIM 계산
            ts_hr = ts_hr.to(device)
            ts_sr = ts_sr.to(device)

            ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0)
                                                      , torch.unsqueeze(ts_hr, 0)
                                                   ]])

            _psnr_test = ignite_result.metrics['psnr']
            _ssim_test = ignite_result.metrics['ssim']
            psnr_test.add_item(_psnr_test)
            ssim_test.add_item(_ssim_test)

        loss_test.update_batch()
        psnr_test.update_batch()
        ssim_test.update_batch()

    _lte = loss_test.update_epoch(is_return=True, path=path_log + path_model)
    _pte = psnr_test.update_epoch(is_return=True, path=path_log + path_model)
    _ste = ssim_test.update_epoch(is_return=True, path=path_log + path_model)

    print("<Test Result>")
    print("test : loss {}, psnr {}, ssim {}".format(_lte, _pte, _ste))