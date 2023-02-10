import os
import numpy as np
import random

import torch
from DLCs.metric_tools import metric_histogram, calc_FAR_FRR, calc_EER, graph_FAR_FRR, graph_ROC

# Datapath
path_log = "C:/super_resolution/log/log_classification/metric_log"
path_rate = path_log + "/FAR_FRR_HR_SYSU.pt"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    metric_save_A = torch.load(path_log + "/metric_HR_A_SYSU.pt")
    distance_same_A = metric_save_A['distance_same']
    distance_diff_A = metric_save_A['distance_diff']

    metric_save_B = torch.load(path_log + "/metric_HR_B_SYSU.pt")
    distance_same_B = metric_save_B['distance_same']
    distance_diff_B = metric_save_B['distance_diff']

    distance_same = distance_same_A + distance_same_B
    distance_diff = distance_diff_A + distance_diff_B

    distance_same.sort()
    distance_diff.sort()

    metric_histogram(distance_same, distance_diff, title="Distribution of Distance (HR, SYSU)", density = True)
    # print(len(distance_same), len(distance_diff))
    # print(min(distance_same), max(distance_same))
    # print(min(distance_diff), max(distance_diff))
    #
    distance_same = np.array(distance_same)
    distance_diff = np.array(distance_diff)
    #
    # print(distance_same)
    # print(len(distance_same[distance_same == 0]))
    #
    threshold, FAR, FRR = calc_FAR_FRR(distance_same, distance_diff, save = path_rate)
    # rate_save = torch.load(path_rate)
    # threshold = rate_save["threshold"]
    # FAR = rate_save['FAR']
    # FRR = rate_save['FRR']
    EER, th = calc_EER(threshold, FAR, FRR)
    graph_FAR_FRR(threshold, FAR, FRR, show_EER = True, title = "Graph of FAR & FRR (HR, SYSU)")
    graph_FAR_FRR(threshold, FAR, FRR, show_EER = True, log=False, title="Graph of FAR & FRR (HR, SYSU)")
    print(f"EER : {EER}, threshold : {th}")
    torch.save({"EER" : EER, "th" : th}, path_log + "/EER_HR_SYSU.pt")