import glob
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import json


def Ff(X):
    FX = 7.787 * X + 0.137931
    index = X > 0.008856
    FX[index] = torch.pow(X[index], 1.0 / 3.0)
    return FX


def myRGB2Lab(img):
    X = (0.412453 * img[:, 0, :, :] + 0.357580 * img[:, 1, :, :] + 0.180423 * img[:, 2, :, :]) / 0.950456
    Y = (0.212671 * img[:, 0, :, :] + 0.715160 * img[:, 1, :, :] + 0.072169 * img[:, 2, :, :])
    Z = (0.019334 * img[:, 0, :, :] + 0.119193 * img[:, 1, :, :] + 0.950227 * img[:, 2, :, :]) / 1.088754

    F_X = Ff(X)
    F_Y = Ff(Y)
    F_Z = Ff(Z)

    L = 903.3 * Y
    index = Y > 0.008856
    L[index] = 116 * F_Y[index] - 16
    a = 500 * (F_X - F_Y)
    b = 200 * (F_Y - F_Z)

    return torch.stack([L, a, b], dim=1)


def calculate_Lab_ab(img):
    Lab_img = myRGB2Lab(img)
    a = Lab_img[:, 1, :, :]
    b = Lab_img[:, 2, :, :]
    a = torch.mean(torch.flatten(a, 1), dim=1)
    b = torch.mean(torch.flatten(b, 1), dim=1)

    return float(a), float(b)


root = "/data1/user12/PPR0K_all_files_11161_zip"
retoucher = 'b'
train_input_files = sorted(glob.glob(os.path.join(root, "train/source" + "/*.tif")))
train_input_dir = os.path.join(root, "train/source")
train_target_files = sorted(glob.glob(os.path.join(root, "train/target_" + retoucher + "/*.tif")))
train_mask_files = sorted(glob.glob(os.path.join(root, "train/masks_360p" + "/*.png")))

path_LQ_dict = {}
for path in train_target_files:
    path_LQ_dict[path.split('/')[-1].split('_')[0]] = []
for paths in train_input_files:
    file_index = paths.split('/')[-1].split('_')[0]
    file = paths.split('/')[-1]
    path_LQ_dict[file_index].append(file)

img_list_a = {}
img_list_b = {}
for index in range(len(train_input_files)):
    img_name = os.path.split(train_input_files[index])[-1]
    img_list_a[img_name] = []
    img_list_b[img_name] = []

    for k in path_LQ_dict[img_name.split('_')[0]]:
        if k == img_name:
            continue
        img_gt = Image.open(os.path.join(root, "train/target_" + retoucher + '/' + k))
        img_gt = TF.to_tensor(img_gt).unsqueeze(0)
        img_gt_a, img_gt_b = calculate_Lab_ab(img_gt)
        img_list_a[img_name].append(img_gt_a)
        img_list_b[img_name].append(img_gt_b)

json_str_a = json.dumps(img_list_a)
new_dict_a = json.loads(json_str_a)

json_str_b = json.dumps(img_list_b)
new_dict_b = json.loads(json_str_b)

with open("record_gt_b_a.json", "w") as f:
    json.dump(new_dict_a, f)
with open("record_gt_b_b.json", "w") as f:
    json.dump(new_dict_b, f)
