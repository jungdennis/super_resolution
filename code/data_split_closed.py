import os
from PIL import Image
import cv2

path_subset1 = "C:/super_resolution/PersonReID Data/Reg/subset1/Thermal Images"
path_subset2 = "C:/super_resolution/PersonReID Data/Reg/subset2/Thermal Images"
path_target = "C:/super_resolution/project_use_open/Reg"

# subset 별 이미지 리스트 뽑아오기
img_list_1 = os.listdir(path_subset1)
img_list_2 = os.listdir(path_subset2)

# id와 frame 정보를 담아둘 list 생성
id_frame = []
for img in img_list_1 :
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    img_frame = img_name.split("_")[3]

    id_frame.append([img_id, img_frame])      # 뽑아낸 ID와 frame 번호를 저장

for img in img_list_2 :
    img_name = img.split(".")[0]

    img_id = img_name.split("_")[4]
    img_frame = img_name.split("_")[3]

    id_frame.append([img_id, img_frame])

# 각 ID 별 frame 정보 기록
people_info = {}

for id, frame in id_frame :
    try :
        people_info[id].append(frame)
    except :
        people_info[id] = []
        people_info[id].append(frame)

# Data Split
'''
Closed World Setting
1. 각 인원 당 들어가 있는 frame 수(10장씩)을 기준으로 나눔
2. 모든 사람들이 들어가 있음
3. 필요한 데이터 개수
   train : 412명 * 4장
   valid : 412명 * 1장
   test : 412명 * 5장
'''

len_train = 4
len_valid = 1

people_id = list(people_info.keys())

# 각 id 별 train/valid/test frame 나누기
img_train_a = []
img_valid_a = []
img_train_b = []
img_valid_b = []

for id in people_id :
    frame_list = people_info[id].copy()
    img_list = img_list_1 if int(id) % 2 == 0 else img_list_2
    subset_no = 1 if int(id) % 2 == 0 else 2

    for i in range(len_train) :
        frame = frame_list.pop(0)
        img_frag = "_" + frame + "_" + id
        for img in img_list :
            if img[-len(img_frag)-4:-4] == img_frag :
                img_train_a.append((img, subset_no))

    for i in range(len_valid) :
        frame = frame_list.pop(0)
        img_frag = "_" + frame + "_" + id
        for img in img_list :
            if img[-len(img_frag)-4:-4] == img_frag :
                img_valid_a.append((img, subset_no))

    for i in range(len_train) :
        frame = frame_list.pop(0)
        img_frag = "_" + frame + "_" + id
        for img in img_list :
            if img[-len(img_frag)-4:-4] == img_frag :
                img_train_b.append((img, subset_no))

    for i in range(len_valid) :
        frame = frame_list.pop(0)
        img_frag = "_" + frame + "_" + id
        for img in img_list :
            if img[-len(img_frag)-4:-4] == img_frag :
                img_valid_b.append((img, subset_no))

img_test_a = img_train_b + img_valid_b
img_test_b = img_train_a + img_valid_a

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