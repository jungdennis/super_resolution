import os
import numpy as np
import matplotlib.pyplot as plt

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def metric_histogram(list_correct, list_wrong, xlim = None, density = False, title = "Distribution of metric") :
    plt.hist(list_correct, histtype="step", color="b", density=density)
    plt.hist(list_wrong, histtype="step", color="g", density=density)
    plt.title(title)
    plt.xlabel("Metric")
    plt.ylabel("Density")
    plt.legend(["same", "different"])
    if xlim is not None :
        plt.xlim(xlim)
    plt.show()

def calc_FAR_FRR(list_correct, list_wrong, range = None, save = None) :
    if range is None:
        threshold = np.arange(0, max(list_wrong), 0.000001)
    else:
        threshold = np.arange(range[0], range[1], 0.000001)
    cnt_total = len(list_correct) + len(list_wrong)

    FAR = []
    FRR = []

    cnt = 0
    for th in threshold :
        if cnt % 1000 == 0 :
            print(f"Calculating FAR/FFR... {100 * cnt/len(threshold)}%")

        cnt_FA = len(list_correct[list_correct >= th])
        cnt_FR = len(list_wrong[list_wrong < th])

        FAR.append(cnt_FA / cnt_total)
        FRR.append(cnt_FR / cnt_total)

        cnt += 1
        
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
            break
        elif FAR[i] < FRR[i] :
            EER = (FAR[i] * FRR[i-1]) - (FAR[i-1] - FRR[i]) / ((FAR[i] - FRR[i]) - (FAR[i-1] - FRR[i - 1]))
            _x = [threshold[i-1], threshold[i]]
            break
    
    if for_graph :
        return EER, sum(_x) / len(_x)
    else :
        return EER
            
def graph_FAR_FRR(threshold, FAR, FRR, show_EER = False, xlim = None, ylim = None, title = "Graph of FAR & FRR (HR)") :
    plt.plot(threshold, FAR, color="b")
    plt.plot(threshold, FRR, color="g")
    if show_EER :
        EER, x = calc_EER(threshold, FAR, FRR, for_graph = True)
        plt.scatter(x, EER, marker ="o", Color = "r")
        
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Rate")
    
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
    save_path = path_log + "/FAR_FRR_SR.pt"
    
    data = torch.load("C:\super_resolution\log\log_classification\metric_logtest_for_rough_convnext_model.pt")
    distance_same = np.array(data["distance_same"])
    distance_diff = np.array(data["distance_diff"])

    print(len(distance_same), len(distance_diff))

    # 히스토그램 로그스케일로 한번 해보기
    # plt.hist(distance_same, histtype="step", color="b")
    # plt.hist(distance_diff, histtype="step", color="g")
    # plt.title("Distribution of distance (HR)")
    # plt.xlabel("Distance")
    # plt.ylabel("Count")
    # plt.legend(["same", "diff"])
    # plt.show()
    #
    # 
    
    
    # threshold, FAR, FRR = calc_FAR_FRR(list_correct = distance_same,
    #                                    list_wrong = distance_diff,
    #                                    save = save_path)

    save_data = torch.load(save_path)
    threshold = save_data["threshold"]
    FAR = save_data["FAR"]
    FRR = save_data["FRR"]
    graph_FAR_FRR(threshold, FAR, FRR)
    
    EER = calc_EER(threshold, FAR, FRR)
    graph_FAR_FRR(threshold, FAR, FRR, show_EER = True)
    print(EER)
    
    