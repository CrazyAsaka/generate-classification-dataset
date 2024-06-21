import cv2
import numpy as np
import random
import math
import os
from tqdm import tqdm
from utils import Augmentation

count = 0


# 将黑变白，非黑变白
def getMask(src):
    srcImg = np.copy(src)
    srcImg[np.all(srcImg == [0, 0, 0], axis=-1)] = [120, 120, 120]
    srcImg[np.any(srcImg != [120, 120, 120], axis=-1)] = [0, 0, 0]
    srcImg[np.any(srcImg != [0, 0, 0], axis=-1)] = [255, 255, 255]

    # dst = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    return srcImg


def pinImg2Bg(src, bg, srcCoords, bgCoords):
    addition = np.ones_like(src)
    src = cv2.add(addition, src)
    m = cv2.getPerspectiveTransform(srcCoords, bgCoords)
    blackBgSrc = cv2.warpPerspective(src, m, (src.shape[1], src.shape[0]))
    # cv2.imshow('b', blackBgSrc)
    maskImg = getMask(blackBgSrc)

    g = cv2.bitwise_and(maskImg, bg)
    # cv2.imshow('g', g)
    dst = cv2.bitwise_xor(g, blackBgSrc)
    return dst


def get_coord(bg):
    global count
    h,w,_ = bg.shape
    center = np.array([w / 2, h / 2])
    r = random.uniform(0,0.1)
    angle = count % 360
    theta = angle / 360.0 * math.pi
    count += 10
    if count >= 360:
        count = 0
    ml = min(w, h) // (2+r)
    # theta = random.uniform(0, 2*math.pi)  # Anticlockwise
    vec_1 = np.array([ml * math.cos(theta), ml * math.sin(theta)])
    vec_1t = np.array([-ml * math.cos(theta), -ml * math.sin(theta)])
    vec_2 = np.array([ml * math.cos(theta + 1/2*math.pi), ml * math.sin(theta + 1/2*math.pi)])
    vec_2t = np.array([-ml * math.cos(theta + 1/2*math.pi), -ml * math.sin(theta + 1/2*math.pi)])
    p1 = center + vec_1
    p1_t = center + vec_1t
    p2 = center + vec_2
    p2_t = center + vec_2t
    # p1 = list(map(int, p1))
    # p1_t = list(map(int, p1_t))
    # p2 = list(map(int, p2))
    # p2_t = list(map(int, p2_t))

    # cv2.circle(bg, p1, 3, (0,0,255), 2)
    # cv2.circle(bg, p1_t, 3, (0, 0, 255), 2)
    # cv2.circle(bg, p2, 3, (0, 0, 255), 2)
    # cv2.circle(bg, p2_t, 3, (0, 0, 255), 2)
    return np.float32([p1,p2,p1_t,p2_t])


def generate_img(src_img_path, bg_path, new_h, new_w):
    src_img = cv2.imread(src_img_path)
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (new_w, new_h))
    src = cv2.resize(src_img, (bg.shape[1], bg.shape[0]))
    dst_coords = get_coord(bg)
    src_coords = np.float32([[bg.shape[1] - 1, bg.shape[0] - 1],
                           [0, bg.shape[0] - 1],
                           [0, 0],
                           [bg.shape[1] - 1, 0]])
    genImg = pinImg2Bg(src, bg, src_coords, dst_coords)
    return genImg


def get_img_paths(root) -> dict:
    old_dataset = {}
    for cls in os.listdir(root):
        old_dataset[f"{cls}"] = []  # 初始化每个类别的空列表
        for img_path in os.listdir(f"{root}/{cls}"):
            old_dataset[f"{cls}"].append(f"{root}/{cls}/{img_path}")
    return old_dataset


def get_bg_paths(root) -> list:
    bg_paths = []
    for bg_path in os.listdir(f"{root}"):
        bg_paths.append(f"{root}/{bg_path}")
    return bg_paths


def get_classes(root) -> list:
    classes = []
    for cls in os.listdir(f"{root}"):
        classes.append(f"{cls}")
    return classes


def create_data(dataset_root, create_num, imgs_root, bgs_path, h, w, initial_code=0, use_augm=True):
    contents = os.listdir(imgs_root)
    classes = [item for item in contents if os.path.isdir(os.path.join(imgs_root, item))]
    print(classes)
    if os.path.exists(f'{dataset_root}') is not True:
        os.makedirs(f'{dataset_root}')
    for cls in classes:
        if os.path.exists(f'{dataset_root}/{cls}') is not True:
            os.makedirs(f'{dataset_root}/{cls}')
    dataset = get_img_paths(imgs_root)
    bgs = get_bg_paths(bgs_path)
    for class_ in classes:
        t = initial_code
        with tqdm(total=create_num-t) as pbar:
            while t < create_num:
                try:
                    img = dataset[f"{class_}"][t % len(dataset[f"{class_}"])]
                    if img is None:
                        continue
                    random.shuffle(bgs)
                    bg = bgs[0]
                    # img = img[:, :, ::-1]
                    t += 1
                    pbar.update(1)
                    gen = generate_img(img, bg, h, w)
                    if use_augm is True:
                        my_aug = Augmentation.MyImgAugmentation()
                        gen = my_aug.applyAugmentation(gen)
                    cv2.imwrite(f'{dataset_root}/{class_}/{t}.jpg', gen)
                except:
                    pass
                    # print(img)


if __name__ == '__main__':

    src_root = './Datasets_2'
    bg_root = './bg'
    dataset_root = './dataset'
    create_num = int(360/10)*160*2  # 每类都创建n个图片
    create_data(dataset_root, 100, src_root, bg_root, 128, 128, initial_code=20)
    # create_data(dataset_root, 10, src_root, bg_root, 128, 128, initial_code=0, use_augm=True)
    # gen = generate_img(src_paths[0][0], bg_paths[0], 80, 80)
    # cv2.imshow('gen', gen)
    # cv2.waitKey(0)
