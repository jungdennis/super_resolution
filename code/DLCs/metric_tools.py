import os
import numpy as np
import matplotlib.pyplot as plt

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def metric_histogram(list_correct, list_wrong, xlim=None, density=False, title="Distribution of Distance"):
    plt.hist(list_correct, histtype="step", color="b", bins = int(max(list_correct)*2), density=density)
    plt.hist(list_wrong, histtype="step", color="g", bins = int(max(list_wrong)*2), density=density)
    plt.title(title)
    plt.xlabel("Distance")
    if density is False :
        plt.ylabel("Count")
        plt.yscale("log")
    else :
        plt.ylabel("Probability")
    plt.legend(["genuine", "imposter"])
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()

def calc_FAR_FRR(list_correct, list_wrong, range = None, save = None) :
    if range is None:
        threshold = np.arange(0, max(list_wrong), 0.00001)
    else:
        threshold = np.arange(range[0], range[1], 0.00001)

    total_FA = len(list_correct)
    total_FR = len(list_wrong)

    list_correct = np.array(list_correct)
    list_wrong = np.array(list_wrong)

    FAR = []
    FRR = []

    cnt = 0
    for th in threshold :
        if cnt % 1000 == 0 :
            print(f"Calculating FAR/FFR... {100 * cnt/len(threshold)}%")

        cnt_FA = len(list_correct[list_correct >= th])
        cnt_FR = len(list_wrong[list_wrong < th])

        FAR.append(cnt_FA / total_FA)
        FRR.append(cnt_FR / total_FR)

        cnt += 1
        # cnt_FA = 0
        # cnt_FR = 0
        
    if save is not None :
        torch.save({'threshold': threshold,
            'FAR': FAR,
            "FRR" : FRR}, save)
            
    return threshold, FAR, FRR

def calc_EER(threshold, FAR, FRR, for_graph = False) :
    _x = []
    for i in range(len(threshold)) :
        if FAR[i] > FRR[i] :
            pass
        elif FAR[i] == FRR[i] :
            EER = FAR[i]
            _x = [threshold[i]]
            print(_x)
            break
        elif FAR[i] < FRR[i] :
            print(f"i : {threshold[i]}, {FAR[i]}, {FRR[i]}")
            print(f"i-1 : {threshold[i-1]}, {FAR[i-1]}, {FRR[i-1]}")
            if FAR[i-1] != FAR[i] :
                EER = (FAR[i] * FRR[i-1]) - (FAR[i-1] - FRR[i]) / ((FAR[i] - FRR[i]) - (FAR[i-1] - FRR[i - 1]))
            elif FAR[i-1] == FAR[i] :
                EER = FAR[i-1]
            _x = [threshold[i-1], threshold[i]]
            print(_x)
            break
    
    if for_graph :
        return EER, sum(_x) / len(_x)
    else :
        return EER
            
def graph_FAR_FRR(threshold, FAR, FRR, show_EER = False, xlim = None, ylim = None, log = True, title = "Graph of FAR & FRR") :
    plt.plot(threshold, FAR, color="b")
    plt.plot(threshold, FRR, color="g")
    if show_EER :
        EER, x = calc_EER(threshold, FAR, FRR, for_graph = True)
        plt.scatter(x, EER, marker ="o", color = "r")
        
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Rate")
    if log :
        plt.yscale("log")
    
    if show_EER :
        plt.legend(["FAR", "FRR", "EER"])
    else :
        plt.legend(["FAR", "FRR"])
    
    if xlim is not None :
        plt.xlim(xlim)  
    if ylim is not None :
        plt.ylim(xlim) 
    
    plt.show()

if __name__ == "__main__" :
    path_log = "C:/super_resolution/log/log_classification/metric_log"

    metric_save_A = torch.load(path_log + "/metric_HR_A.pt")
    distance_same_A = metric_save_A['distance_same']
    distance_diff_A = metric_save_A['distance_diff']

    metric_save_B = torch.load(path_log + "/metric_HR_B.pt")
    distance_same_B = metric_save_B['distance_same']
    distance_diff_B = metric_save_B['distance_diff']

    distance_same = distance_same_A + distance_same_B
    distance_diff = distance_diff_A + distance_diff_B

    distance_same.sort()
    distance_diff.sort()

    metric_histogram(distance_same, distance_diff, title="Distribution of Distance (Original)", density=True)

    metric_save_A = torch.load(path_log + "/metric_LR_A.pt")
    distance_same_A = metric_save_A['distance_same']
    distance_diff_A = metric_save_A['distance_diff']

    metric_save_B = torch.load(path_log + "/metric_LR_B.pt")
    distance_same_B = metric_save_B['distance_same']
    distance_diff_B = metric_save_B['distance_diff']

    distance_same = distance_same_A + distance_same_B
    distance_diff = distance_diff_A + distance_diff_B

    distance_same.sort()
    distance_diff.sort()

    metric_histogram(distance_same, distance_diff, title="Distribution of Distance (LR)", density=True)
    
    