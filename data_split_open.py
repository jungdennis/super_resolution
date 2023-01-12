import os
from PIL import Image
import cv2

path_subset1 = "C:/super_resolution/PersonReID Data/RegDB/subset1/Thermal Images"
path_subset2 = "C:/super_resolution/PersonReID Data/RegDB/subset2/Thermal Images"
path_target = "C:/super_resolution/data_split_open/RegDB"

# subset 별 이미지 리스트 뽑아오기
img_list_1 = os.listdir(path_subset1)
img_list_2 = os.listdir(path_subset2)

# id와 frame 정보를 담아둘 list 생성

id_img_1 = []
id_info_1 = {}
for img in img_list_1 :
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_1.append([img_id, img])      # 뽑아낸 ID와 frame 번호를 저장
for id, name in id_img_1 :
    try :
        id_info_1[id].append(name)
    except :
        id_info_1[id] = []
        id_info_1[id].append(name)

id_img_2 = []
id_info_2 = {}
for img in img_list_2 :
    img_name = img.split(".")[0]

    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_2.append([img_id, img])
for id, name in id_img_2 :
    try :
        id_info_2[id].append(name)
    except :
        id_info_2[id] = []
        id_info_2[id].append(name)

# Data Split
'''
Open World Setting
1. 전체 인원을 기준으로 사람을 나눔
2. 각 인원에 해당하는 전체 frame이 들어감
3. 필요한 데이터 개수
   train : 165명 * 10장
   valid : 41명 * 10장
   test : 206명 * 10장
'''
'''
len_train = 165
len_valid = 41

# train/valid/test id 나누기
id_train_a = []
id_valid_a = []
id_test_a = []
id_train_b = []
id_valid_b = []
id_test_b = []

id_a = list(id_info_1.keys())
id_b = list(id_info_2.keys())

id_train_a = id_a[0:len_train]
id_train_b = id_b[0:len_train]

id_valid_a = id_a[len_train:]
id_valid_b = id_b[len_train:]

id_test_a = id_b.copy()
id_test_b = id_a.copy()

# 각 id 별 train/valid/test frame 나누기
img_train_a = []
img_valid_a = []
img_test_a = []
img_train_b = []
img_valid_b = []
img_test_b = []

for id in id_train_a :
    subset_no = 1
    _img_list = id_info_1[id]
    for img in _img_list :
        img_train_a.append((img, subset_no))

for id in id_valid_a :
    subset_no = 1
    _img_list = id_info_1[id]
    for img in _img_list :
        img_valid_a.append((img, subset_no))

for id in id_test_a :
    subset_no = 2
    _img_list = id_info_2[id]
    for img in _img_list :
        img_test_a.append((img, subset_no))

for id in id_train_b :
    subset_no = 2
    _img_list = id_info_2[id]
    for img in _img_list :
        img_train_b.append((img, subset_no))

for id in id_valid_b:
    subset_no = 2
    _img_list = id_info_2[id]
    for img in _img_list:
        img_valid_b.append((img, subset_no))

for id in id_test_b:
    subset_no = 1
    _img_list = id_info_1[id]
    for img in _img_list:
        img_test_b.append((img, subset_no))

print(len(img_train_a), len(img_valid_a), len(img_test_a))
print(len(img_train_b), len(img_valid_b), len(img_test_b))

# 사진 폴더에 저장하기
def move_data(set, img_train, img_valid, img_test) :
    if set == "A" or set == "B" :
        path_save = path_target + "/" + set+ "_set"
        print("save_path :", path_save)
        for file_name, subset_no in img_train :
            path_load = path_subset1 if subset_no == 1 else path_subset2
            img = Image.open(path_load + "/" + file_name)
            img.save(path_save + "/train/images/" + file_name)
        print("Train Data Move Complete")

        for file_name, subset_no in img_valid :
            path_load = path_subset1 if subset_no == 1 else path_subset2
            img = Image.open(path_load + "/" + file_name)
            img.save(path_save + "/val/images/" + file_name)
        print("Valid Data Move Complete")

        for file_name, subset_no in img_test :
            path_load = path_subset1 if subset_no == 1 else path_subset2
            img = Image.open(path_load + "/" + file_name)
            img.save(path_save + "/test/images/" + file_name)
        print("Test Data Move Complete")
    else :
        print("Input only A or B")

move_data("A", img_train_a, img_valid_a, img_test_a)
print("A set complete")
move_data("B", img_train_b, img_valid_b, img_test_b)
print("B set complete")
'''