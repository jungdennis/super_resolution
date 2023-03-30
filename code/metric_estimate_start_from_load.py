import os
import numpy as np
import random
import csv
import datetime

import torch
from DLCs.metric_tools import metric_histogram, calc_FAR_FRR_v2, calc_EER, graph_FAR_FRR, graph_ROC

import argparse

# 단일 코드로 돌릴 때 사용
_database = "Reg"
_model = "convnext"
_resolution = "HR"
_scale = 4
_noise = 30
_sr_model = "IMDN"
_mode = "all"

# Argparse Setting
parser = argparse.ArgumentParser(description = "거리 측정 파일이 있을 경우에 FAR, FRR, EER을 측정합니다.")

parser.add_argument('--database', required = False, choices = ["Reg", "SYSU"], default = _database, help = "사용할 데이터베이스 입력 (Reg, SYSU)")
parser.add_argument("--model", required = False, choices = ["convnext", "inception", "densenet"], default = _model, help = "이미지 분류 모델 선택 (convnext, inception)")
parser.add_argument('--resolution', required = False, choices = ["HR", "LR", "SR"], default = _resolution, help = "이미지의 Resolution 선택 (HR, LR, SR)")
parser.add_argument('--scale', required = False, type = int, default = _scale, help = "LR 이미지의 Scale Factor 입력")
parser.add_argument('--noise', required = False, type = int, default = _noise, help = "LR 이미지 noise의 sigma 값 입력")
parser.add_argument('--sr', required = False, default = _sr_model, help = "SR 이미지의 알고리즘 이름 입력")
parser.add_argument("--mode", required = False, default = _mode, choices = ["all", "pick_1", "pick_2"],  help = "RegDB의 이미지 선택 모드 입력")
parser.add_argument("--csv", required = False, action = "store_true", help = "csv파일에 기록 여부 선택 (True, False)")

args = parser.parse_args()

# Mode Setting
DATABASE = args.database        # Reg or SYSU
MODEL = args.model
RESOLUTION = args.resolution
SCALE_FACTOR = args.scale
NOISE = args.noise
SR_MODEL = args.sr
MEASURE_MODE = args.mode        # all or pick_1 or pick_2
CSV = args.csv

# Datapath
if RESOLUTION == "HR" :
    path_log = f"C:/super_resolution/log/log_metric/graph_and_log/{DATABASE}/{MODEL}/HR/"
    if DATABASE == "SYSU" :
        option_frag = f"HR_{MODEL}"
    elif DATABASE == "Reg" :
        option_frag = f"HR_{MEASURE_MODE}_{MODEL}"
elif RESOLUTION == "LR" :
    path_log = f"C:/super_resolution/log/log_metric/graph_and_log/{DATABASE}/{MODEL}/LR_{SCALE_FACTOR}_{NOISE}/"
    if DATABASE == "SYSU" :
        option_frag = f"LR_{SCALE_FACTOR}_{NOISE}_{MODEL}"
    elif DATABASE == "Reg" :
        option_frag = f"LR_{SCALE_FACTOR}_{NOISE}_{MEASURE_MODE}_{MODEL}"
elif RESOLUTION == "SR" :
    path_log = f"C:/super_resolution/log/log_metric/graph_and_log/{DATABASE}/{MODEL}/SR_{SR_MODEL}/"
    if DATABASE == "SYSU" :
        option_frag = f"SR_{SR_MODEL}_{MODEL}"
    elif DATABASE == "Reg" :
        option_frag = f"SR_{SR_MODEL}_{MEASURE_MODE}_{MODEL}"

dt_now = datetime.datetime.now()
date = str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day) + " " + str(dt_now.hour) + ":" + str(dt_now.minute)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    # 로그 저장 설정
    if CSV:
        try:
            log_check = open(f"C:/super_resolution/log/log_metric/graph_and_log/{DATABASE}/calc_log.csv", "r")
            log_check.close()
            log = open(f"C:/super_resolution/log/log_metric/graph_and_log/{DATABASE}/calc_log.csv", "a", newline = "")
            log_write = csv.writer(log, delimiter=',')
        except:
            log = open(f"C:/super_resolution/log/log_metric/graph_and_log/{DATABASE}/calc_log.csv", "a", newline = "")
            log_write = csv.writer(log, delimiter=',')
            log_write.writerow(["date", "option", "mean_genu", "std_genu", "mean_impo", "std_impo", "th_EER", "EER"])

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

    distance_same = np.array(distance_same)
    distance_diff = np.array(distance_diff)

    mean_genu = distance_same.mean()
    std_genu = distance_same.std()

    mean_impo = distance_diff.mean()
    std_impo = distance_diff.std()

    metric_histogram(distance_same, distance_diff, density = True, xlim = [0, 65],
                     title=f"Distribution of Distance ({DATABASE}_{option_frag})",
                     save_path = path_log + f"hist_{DATABASE}_{option_frag}.png")
    #
    # threshold, FAR, FRR = calc_FAR_FRR_v2(distance_same, distance_diff, save = path_log + f"FAR_FRR_{DATABASE}_{option_frag}.pt")
    # EER, th = calc_EER(threshold, FAR, FRR, save = path_log + f"EER_{DATABASE}_{option_frag}.pt")
    #
    # graph_FAR_FRR(threshold, FAR, FRR, show_EER = True,
    #               title = f"Graph of FAR & FRR ({DATABASE}_{option_frag})",
    #               save_path = path_log + f"graph_EER_{DATABASE}_{option_frag}.png")
    # if DATABASE == "Reg" :
    #     graph_FAR_FRR(threshold, FAR, FRR, show_EER = True, log=False,
    #                   title = f"Graph of FAR & FRR ({DATABASE}_{option_frag})",
    #                   save_path = path_log + f"graph_EER_{DATABASE}_{option_frag}_log.png")

    if CSV :
        log_write.writerow([date, option_frag, mean_genu, std_genu, mean_impo, std_impo, th, EER])
        log.close()