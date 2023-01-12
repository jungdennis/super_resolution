import os

path_target = "C:/super_resolution/data_split_open/RegDB"

train_a = os.listdir(path_target + "/A_set/train/images")
valid_a = os.listdir(path_target + "/A_set/val/images")
test_a = os.listdir(path_target + "/A_set/test/images")
train_b = os.listdir(path_target + "/B_set/train/images")
valid_b = os.listdir(path_target + "/B_set/val/images")
test_b = os.listdir(path_target + "/B_set/test/images")

id_img_train_a = []
id_info_train_a = {}
for img in train_a :
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_train_a.append([img_id, img])      # 뽑아낸 ID와 frame 번호를 저장
for id, name in id_img_train_a :
    try :
        id_info_train_a[id] += 1
    except :
        id_info_train_a[id] = 0
        id_info_train_a[id] += 1

id_img_valid_a = []
id_info_valid_a = {}
for img in valid_a :
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_valid_a.append([img_id, img])      # 뽑아낸 ID와 frame 번호를 저장
for id, name in id_img_valid_a :
    try :
        id_info_valid_a[id] += 1
    except :
        id_info_valid_a[id] = 0
        id_info_valid_a[id] += 1

id_img_test_a = []
id_info_test_a = {}
for img in test_a :
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_test_a.append([img_id, img])      # 뽑아낸 ID와 frame 번호를 저장
for id, name in id_img_test_a :
    try :
        id_info_test_a[id] += 1
    except :
        id_info_test_a[id] = 0
        id_info_test_a[id] += 1

id_img_train_b = []
id_info_train_b = {}
for img in train_b:
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_train_b.append([img_id, img])  # 뽑아낸 ID와 frame 번호를 저장
for id, name in id_img_train_b:
    try:
        id_info_train_b[id] += 1
    except:
        id_info_train_b[id] = 0
        id_info_train_b[id] += 1

id_img_valid_b = []
id_info_valid_b = {}
for img in valid_b:
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_valid_b.append([img_id, img])  # 뽑아낸 ID와 frame 번호를 저장
for id, name in id_img_valid_b:
    try:
        id_info_valid_b[id] += 1
    except:
        id_info_valid_b[id] = 0
        id_info_valid_b[id] += 1

id_img_test_b = []
id_info_test_b = {}
for img in test_b:
    # 파일명 뽑아내기
    img_name = img.split(".")[0]

    # ID 번호와 frame 번호 뽑아내기
    img_id = img_name.split("_")[4]
    # img_frame = img_name.split("_")[3]

    id_img_test_b.append([img_id, img])  # 뽑아낸 ID와 frame 번호를 저장
for id, name in id_img_test_b:
    try:
        id_info_test_b[id] += 1
    except:
        id_info_test_b[id] = 0
        id_info_test_b[id] += 1

print(len(id_info_train_a), id_info_train_a)
print(len(id_info_valid_a), id_info_valid_a)
print(len(id_info_test_a), id_info_test_a)
print(len(id_info_train_b), id_info_train_b)
print(len(id_info_valid_b), id_info_valid_b)
print(len(id_info_test_b), id_info_test_b)