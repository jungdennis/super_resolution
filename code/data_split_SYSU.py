import os
import random
import cv2

SEED = 485
random.seed(SEED)

path_load = "C:/super_resolution/data/SYSU-MM01"
path_save = "C:/super_resolution/data/SYSU_split"

if __name__ == "__main__" :
    # 사람 라벨 데이터 만들기
    people_num_3 = os.listdir(path_load + "/cam3")
    people_num_6 = os.listdir(path_load + "/cam6")

    people_label_3 = []
    people_label_6 = []

    for num in people_num_3 :
        people_label_3.append("3" + num)
    for num in people_num_6 :
        people_label_6.append("6" + num)

    people_label = people_label_3 + people_label_6
    random.shuffle(people_label)

    print(len(people_label_3), len(people_label_6), len(people_label))

    # 사람 라벨 train/valid/test로 나누기
    '''
    <fold A>
    train : 354명 * 20장
    valid : 39명 * 20장
    test : 392명 * 20장
    
    <fold B>
    train : 353명 * 20장
    valid : 39명 * 20장
    test : 393명 * 20장
    '''

    people_label_train_a = people_label[:354]
    people_label_valid_a = people_label[354:393]
    people_label_test_a = people_label[393:]

    people_label_train_b = people_label_test_a[:353]
    people_label_valid_b = people_label_test_a[353:]
    people_label_test_b = people_label_train_a + people_label_valid_a

    print(len(people_label_train_a), len(people_label_valid_a), len(people_label_test_a))
    print(len(people_label_train_b), len(people_label_valid_b), len(people_label_test_b))

    for people in people_label_train_a :
        path_fold = "/A_set"
        path_data = "/train/images"

        if people[0] == "3" :
            path_cam = "/cam3"
        elif people[0] == "6" :
            path_cam = "/cam6"

        path_img = path_load + path_cam + f"/{people[1:]}"
        img_list = os.listdir(path_img)
        if len(img_list) > 20 :
            print(f"20장 초과의 인원이 있습니다 : {people}, {len(img_list)}장")
        elif len(img_list) < 20 :
            print(f"20장 미만의 인원이 있습니다 : {people}, {len(img_list)}장")

        for name in img_list :
            img = cv2.imread(path_img + f"/{name}")
            try :
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)
            except :
                os.makedirs(path_save + path_fold + path_data)
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)

    print(len(os.listdir(path_save + path_fold + path_data)))

    for people in people_label_valid_a :
        path_fold = "/A_set"
        path_data = "/val/images"

        if people[0] == "3" :
            path_cam = "/cam3"
        elif people[0] == "6" :
            path_cam = "/cam6"

        path_img = path_load + path_cam + f"/{people[1:]}"
        img_list = os.listdir(path_img)
        if len(img_list) > 20 :
            print(f"20장 초과의 인원이 있습니다 : {people}, {len(img_list)}장")
        elif len(img_list) < 20 :
            print(f"20장 미만의 인원이 있습니다 : {people}, {len(img_list)}장")

        for name in img_list :
            img = cv2.imread(path_img + f"/{name}")
            try :
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)
            except :
                os.makedirs(path_save + path_fold + path_data)
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)

    print(len(os.listdir(path_save + path_fold + path_data)))

    for people in people_label_test_a :
        path_fold = "/A_set"
        path_data = "/test/images"

        if people[0] == "3" :
            path_cam = "/cam3"
        elif people[0] == "6" :
            path_cam = "/cam6"

        path_img = path_load + path_cam + f"/{people[1:]}"
        img_list = os.listdir(path_img)
        if len(img_list) > 20 :
            print(f"20장 초과의 인원이 있습니다 : {people}, {len(img_list)}장")
        elif len(img_list) < 20 :
            print(f"20장 미만의 인원이 있습니다 : {people}, {len(img_list)}장")

        for name in img_list :
            img = cv2.imread(path_img + f"/{name}")
            try :
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)
            except :
                os.makedirs(path_save + path_fold + path_data)
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)

    print(len(os.listdir(path_save + path_fold + path_data)))

    for people in people_label_train_b :
        path_fold = "/B_set"
        path_data = "/train/images"

        if people[0] == "3" :
            path_cam = "/cam3"
        elif people[0] == "6" :
            path_cam = "/cam6"

        path_img = path_load + path_cam + f"/{people[1:]}"
        img_list = os.listdir(path_img)
        if len(img_list) > 20 :
            print(f"20장 초과의 인원이 있습니다 : {people}, {len(img_list)}장")
        elif len(img_list) < 20 :
            print(f"20장 미만의 인원이 있습니다 : {people}, {len(img_list)}장")

        for name in img_list :
            img = cv2.imread(path_img + f"/{name}")
            try :
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)
            except :
                os.makedirs(path_save + path_fold + path_data)
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)

    print(len(os.listdir(path_save + path_fold + path_data)))

    for people in people_label_valid_b:
        path_fold = "/B_set"
        path_data = "/val/images"

        if people[0] == "3":
            path_cam = "/cam3"
        elif people[0] == "6":
            path_cam = "/cam6"

        path_img = path_load + path_cam + f"/{people[1:]}"
        img_list = os.listdir(path_img)
        if len(img_list) > 20 :
            print(f"20장 초과의 인원이 있습니다 : {people}, {len(img_list)}장")
        elif len(img_list) < 20 :
            print(f"20장 미만의 인원이 있습니다 : {people}, {len(img_list)}장")

        for name in img_list :
            img = cv2.imread(path_img + f"/{name}")
            try:
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)
            except:
                os.makedirs(path_save + path_fold + path_data)
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)

    print(len(os.listdir(path_save + path_fold + path_data)))

    for people in people_label_test_b:
        path_fold = "/B_set"
        path_data = "/test/images"

        if people[0] == "3":
            path_cam = "/cam3"
        elif people[0] == "6":
            path_cam = "/cam6"

        path_img = path_load + path_cam + f"/{people[1:]}"
        img_list = os.listdir(path_img)
        if len(img_list) > 20 :
            print(f"20장 초과의 인원이 있습니다 : {people}, {len(img_list)}장")
        elif len(img_list) < 20 :
            print(f"20장 미만의 인원이 있습니다 : {people}, {len(img_list)}장")

        for name in img_list :
            img = cv2.imread(path_img + f"/{name}")
            try:
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)
            except:
                os.makedirs(path_save + path_fold + path_data)
                cv2.imwrite(path_save + path_fold + path_data + f"/{people}_{name}", img)

    print(len(os.listdir(path_save + path_fold + path_data)))