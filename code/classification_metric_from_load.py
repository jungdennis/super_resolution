import os
import numpy as np
import random

import torch
from DLCs.metric_tools import metric_histogram, calc_FAR_FRR, calc_EER, graph_FAR_FRR

# Datapath
path_log = "C:/super_resolution/log/log_classification/metric_log"
path_metric = path_log + "/metric_HR_B"
path_rate = path_log + "/FAR_FRR_HR_B.pt"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    metric_save = torch.load(path_metric)
    distance_same = metric_save['distance_same']
    distance_diff = metric_save['distance_diff']
    metric_histogram(distance_same, distance_diff, title="Distribution of metric (HR, Fold B)")

    # threshold, FAR, FRR = calc_FAR_FRR(distance_same, distance_diff, save = path_rate)
    rate_save = torch.load(path_rate)
    threshold = rate_save["threshold"]
    FAR = rate_save['FAR']
    FRR = rate_save['FRR']
    EER = calc_EER(distance_same, distance_diff)
    graph_FAR_FRR(threshold, FAR, FRR, show_EER = True, title = "Graph of FAR & FRR (HR, Fold A)")
    graph_FAR_FRR(threshold, FAR, FRR, show_EER=True, log=False, title="Graph of FAR & FRR (HR, Fold A)")
    print(EER)