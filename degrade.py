import os
import cv2
import numpy as np
import csv
import datetime

main_path = "C:/super_resolution/"
load_path = "C:/super_resolution/data_split_open/RegDB/"
save_path = "C:/super_resolution/image/"

# option 변수 설정
resize_target = 256
scale_factor = 4
blur_sigma = 3
noise_sigma = 10

def degrade(fold, set, name) :
    # 원본 이미지 불러오기
    read_path = load_path + fold + "/" + set + "/images/" + name
    img_in = cv2.imread(read_path)

    # 256*256으로 resize 하여 HR 만들기
    size_resize = (resize_target, resize_target)
    img_resize = cv2.resize(img_in, size_resize, cv2.INTER_LANCZOS4)

    save_path_hr = save_path + "HR/" + fold + "/" + set + "/images/" + name
    cv2.imwrite(save_path_hr, img_resize)

    # 4배율로 줄여 LR 만들기
    '''
    Low Resolution 이미지 만들기
    1. Gaussian Blur를 적용
    2. 4배율로 Downsize
    3. Noise를 첨가
    '''

    # Gaussian Blur
    img_lr = cv2.GaussianBlur(img_resize, (0, 0), blur_sigma)

    # Downsize
    img_lr = cv2.resize(img_lr, (0, 0), fx=(1 / scale_factor), fy=(1 / scale_factor))

    # Gaussian Noise - Thermal Image는 일종의 GrayScale로 볼 수 있으므로 Gray Noise 적용 예정
    noise_size = resize_target // scale_factor
    noise = np.zeros((noise_size, noise_size))

    for i in range(noise_size):
        for j in range(noise_size):
            make_noise = np.random.normal() * noise_sigma
            noise[i][j] = make_noise

    img_b, img_g, img_r = cv2.split(img_lr)

    img_noise_b = np.uint8(np.clip(img_b + noise, 0, 255))
    img_noise_g = np.uint8(np.clip(img_g + noise, 0, 255))
    img_noise_r = np.uint8(np.clip(img_r + noise, 0, 255))

    img_out = cv2.merge((img_noise_b, img_noise_g, img_noise_r))

    save_path_sr = save_path + "LR/" + fold + "/" + set + "/images/" + name
    cv2.imwrite(save_path_sr, img_out)

# 이미지 리스트 불러오기
fold_list = os.listdir(load_path)
set_list = os.listdir(load_path + fold_list[0])
img_dict = {}

for fold in fold_list :
    fold_dict = {}
    img_dict[fold] = fold_dict

    for set in set_list :
        list_path = load_path + fold + "/" + set + "/images/"
        fold_dict[set] = os.listdir(list_path)

# HR, SR 이미지 생성 및 저장
for fold in fold_list :
    for set in set_list :
        img_list = img_dict[fold][set]
        for img in img_list :
            print(fold, set, img)
            degrade(fold, set, img)

# 옵션 변수 csv 파일에 저장
option_csv_path = main_path + "option_degrade.csv"

option_write = open(option_csv_path, 'a', encoding = 'utf-8', newline = '')
writer = csv.writer(option_write)

dt_now = datetime.datetime.now()
time = str(dt_now.date()) + " " + str(dt_now.hour) + ":" + str(dt_now.minute) + ":" + str(dt_now.second)

option = [time, resize_target, scale_factor, blur_sigma, noise_sigma]
writer.writerow(option)
option_write.close()