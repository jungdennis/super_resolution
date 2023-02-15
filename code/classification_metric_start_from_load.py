import os
import numpy as np
import random

import torch
from DLCs.metric_tools import metric_histogram, calc_FAR_FRR_v2, calc_EER, graph_FAR_FRR, graph_ROC

import argparse

# 단일 코드로 돌릴 때 사용
_database = "Reg"
_resolution = "HR"
_scale = 4
_noise = 30
_model = "IMDN"
_mode = "all"

# Argparse Setting
parser = argparse.ArgumentParser(description = "거리 측정 파일이 있을 경우에 FAR, FRR, EER을 측정합니다.")

parser.add_argument('--database', required = True, choices = ["Reg", "SYSU"], default = _database, help = "사용할 데이터베이스 입력 (Reg, SYSU)")
parser.add_argument('--resolution', required = True, choices = ["HR", "LR", "SR"], default = _resolution, help = "이미지의 Resolution 선택 (HR, LR, SR)")
parser.add_argument('--scale', required = False, type = int, default = _scale, help = "LR 이미지의 Scale Factor 입력")
parser.add_argument('--noise', required = False, type = int, default = _noise, help = "LR 이미지 noise의 sigma 값 입력")
parser.add_argument('--model', required = False, default = _model, help = "SR 이미지의 알고리즘 이름 입력")
parser.add_argument("--mode", required = True, default = _mode, choices = ["all", "pick_1", "pick_2"],  help = "RegDB의 이미지 선택 모드 입력")

args = parser.parse_args()

# Mode Setting
DATABASE = args.database        # Reg or SYSU
RESOLUTION = args.resolution
SCALE_FACTOR = args.scale
NOISE = args.noise
MODEL = args.model
MEASURE_MODE = args.mode        # all or pick_1 or pick_2

# Datapath
if RESOLUTION == "HR" :
    path_log = f"C:/super_resolution/log/log_classification/graph_and_log/{DATABASE}/HR/"
    if DATABASE == "SYSU" :
        option_frag = "HR"
    elif DATABASE == "Reg" :
        option_frag = f"HR_{MEASURE_MODE}"
elif RESOLUTION == "LR" :
    path_log = f"C:/super_resolution/log/log_classification/graph_and_log/{DATABASE}/LR_{SCALE_FACTOR}_{NOISE}/"
    if DATABASE == "SYSU" :
        option_frag = f"LR_{SCALE_FACTOR}_{NOISE}"
    elif DATABASE == "Reg" :
        option_frag = f"LR_{SCALE_FACTOR}_{NOISE}_{MEASURE_MODE}"
elif RESOLUTION == "SR" :
    path_log = f"C:/super_resolution/log/log_classification/graph_and_log/{DATABASE}/SR_{MODEL}/"
    if DATABASE == "SYSU" :
        option_frag = f"SR_{MODEL}"
    elif DATABASE == "Reg" :
        option_frag = f"SR_{MODEL}_{MEASURE_MODE}"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    metric_save_A = torch.load(path_log + f"metric_{DATABASE}_A_{option_frag}.pt")
    distance_same_A = metric_save_A['distance_same']
    distance_diff_A = metric_save_A['distance_diff']

    metric_save_B = torch.load(path_log + f"metric_{DATABASE}_B_{option_frag}.pt")
    distance_same_B = metric_save_B['distance_same']
    distance_diff_B = metric_save_B['distance_diff']

    distance_same = distance_same_A + distance_same_B
    distance_diff = distance_diff_A + distance_diff_B

    distance_same.sort()
    distance_diff.sort()

    metric_histogram(distance_same, distance_diff, title=f"Distribution of Distance ({DATABASE}_{option_frag})", density=True,
                     save_path = path_log + f"hist2_{DATABASE}_{option_frag}.png")
    # print(min(distance_same), max(distance_same))
    # print(min(distance_diff), max(distance_diff))

    distance_same = np.array(distance_same)
    distance_diff = np.array(distance_diff)

    print(len(distance_same[distance_same == 0.0]))

    threshold, FAR, FRR = calc_FAR_FRR_v2(distance_same, distance_diff, save = path_log + f"FAR_FRR_{DATABASE}_{option_frag}.pt")
    EER, th = calc_EER(threshold, FAR, FRR, save = path_log + f"EER_{DATABASE}_{option_frag}.pt")

    graph_FAR_FRR(threshold, FAR, FRR, show_EER = True, title = f"Graph of FAR & FRR ({DATABASE}_{option_frag})",
                  save_path = path_log + f"graph_EER_{DATABASE}_{option_frag}.png")
    if DATABASE == "Reg" :
        graph_FAR_FRR(threshold, FAR, FRR, show_EER = True, log=False, title = f"Graph of FAR & FRR ({DATABASE}_{option_frag})",
                      save_path = path_log + f"graph_EER_{DATABASE}_{option_frag}_log.png")