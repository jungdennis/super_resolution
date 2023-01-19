import os
import numpy as np
import matplotlib.pyplot as plt

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

path_log = "C:/super_resolution/log/log_classification/metric_log"
def FAR_FRR_for_distance(list_correct, list_wrong, range = None) :
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
        '''
        cnt_FA = 0
        cnt_FR = 0

        for data in list_correct :
            if data >= th :
                cnt_FA += 1

        for data in list_wrong :
            if data < th :
                cnt_FR += 1
        '''

        cnt_FA = len(list_correct[list_correct >= th])
        cnt_FR = len(list_wrong[list_wrong < th])

        FAR.append(cnt_FA / cnt_total)
        FRR.append(cnt_FR / cnt_total)

        cnt += 1

    return threshold, FAR, FRR



if __name__ == "__main__" :
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
    # plt.hist(distance_same, histtype="step", color="b", density=True)
    # plt.hist(distance_diff, histtype="step", color="g", density=True)
    # plt.title("Distribution of distance (HR)")
    # plt.xlabel("Distance")
    # plt.ylabel("Density")
    # plt.legend(["same", "diff"])
    # plt.show()

    threshold, FAR, FRR = FAR_FRR_for_distance(distance_same, distance_diff)

    torch.save({'threshold': threshold,
                'FAR': FAR,
                "FRR" : FRR}, path_log + "/FAR_FRR_SR.pt")

    # save_data = torch.load(path_log + "/FAR_FRR_SR.pt")
    # threshold = save_data["threshold"]
    # FAR = save_data["FAR"]
    # FRR = save_data["FRR"]      # 다시 저장 코드 돌리면 FRR로 바꿀 것!
    #
    # plt.plot(threshold, FAR, color="b")
    # plt.plot(threshold, FRR, color="g")
    # plt.title("Graph of FAR & FRR")
    # plt.xlabel("Threshold")
    # plt.ylabel("Rate")
    # plt.ylim([0.00027, 0.000275])
    # plt.xlim([22, 23])
    # plt.legend(["FAR", "FRR"])
    # plt.show()
    #
    # for i in range(len(threshold)) :
    #     if FAR[i] > FRR[i] :
    #         pass
    #     elif FAR[i] == FRR[i] :
    #         EER = FAR[i]
    #         break
    #     elif FAR[i] < FRR[i] :
    #         EER = (FAR[i] * FRR[i-1]) - (FAR[i-1] - FRR[i]) / ((FAR[i] - FRR[i]) - (FAR[i-1] - FRR[i - 1]))
    #         break
    #
    # print(EER)