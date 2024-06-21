import random

import cv2
import numpy as np
import tensorflow as tf
import imutils
import os
from tqdm import tqdm


def gamma_table(gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table


class MyImgAugmentation:
    def __init__(self,
                 useRotation=False, rotationRange=15,  # [-x, x]度内任意旋转
                 useSVNoise=False,SRange=(-2,2),VRange=(10,50),
                 usePerspectTrans=False,perspectScale=0.05,
                 useGaussianNoise=False,gaussianMean=0,gaussianSigma=0.8,
                 useExposure=False, exposureRatio=2.10,  # 1.95
                 useSharpness=False, sharpness=5
                 ):
        self.useRotation = useRotation
        self.useSVNoise = useSVNoise
        self.usePerspectTrans = usePerspectTrans
        self.useGaussianNoise = useGaussianNoise
        self.useExposure = useExposure
        self.useSharpness = useSharpness

        self.rotationRange = rotationRange
        self.SRange = SRange
        self.VRange = VRange
        self.perspectScale = perspectScale
        self.gaussianMean = gaussianMean
        self.gaussianSigma = gaussianSigma
        self.exposureRatio=exposureRatio
        self.sharpness = sharpness

    def applyAugmentation(self, img):
        img1 = np.copy(img)
        if self.useSharpness is True:
            img1 = self.adjust_sharpness(img1)
        if self.useExposure is True:
            img1 = self.adjust_exposure(img1)
        if self.useSVNoise is True:
            img1 = self.randomSV(img1)
        if self.useGaussianNoise is True:
            img1 = self.randomGaussianNoise(img1)
        if self.usePerspectTrans is True:
            img1 = self.randomPerspectiveTransform(img1)
        if self.useRotation is True:
            img1 = self.randomRotation(img1)
        img1 = self.gamma_transform(img1,np.random.uniform(0.3,3,[1]))
        return img1

    def randomRotation(self, img):
        randDirect = random.choice([-1, 1])
        randAngle = random.randint(0, self.rotationRange)
        # print(randDirect, randAngle)
        if randDirect == 1:
            return imutils.rotate(img, randAngle)
        if randDirect == -1:
            return imutils.rotate_bound(img, randAngle)

    def randomSV(self, img):
        # 将 BGR 图像转换为 HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 随机生成 S 通道和 V 通道的噪声
        s_noise = np.random.randint(self.SRange[0], self.SRange[1], size=hsv_image.shape[:2])
        v_noise = np.random.randint(self.VRange[0], self.VRange[1], size=hsv_image.shape[:2])

        # 将噪声添加到 S 通道和 V 通道
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + s_noise, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + v_noise, 0, 255)

        # 将 HSV 图像转换回 BGR
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return bgr_image

    def randomPerspectiveTransform(self, img):
        # 图像尺寸
        height, width = img.shape[:2]

        # 随机生成透视变换的四个点
        tl_x, tl_y = np.random.randint(0, self.perspectScale * width),\
            np.random.randint(0, self.perspectScale * height)
        tr_x, tr_y = np.random.randint((1 - self.perspectScale) * width, width),\
            np.random.randint(0, self.perspectScale * height)
        bl_x, bl_y = np.random.randint(0, self.perspectScale * width),\
            np.random.randint((1 - self.perspectScale) * height, height)
        br_x, br_y = np.random.randint((1 - self.perspectScale) * width, width),\
            np.random.randint((1 - self.perspectScale) * height,height)
        # 定义透视变换前后的四个点坐标
        pts_before = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
        pts_after = np.float32([[tl_x, tl_y], [tr_x, tr_y], [bl_x, bl_y], [br_x, br_y]])
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts_before, pts_after)
        # 对图像进行透视变换
        transformed_image = cv2.warpPerspective(img, M, (width, height))
        return transformed_image

    def randomGaussianNoise(self, img):
        # 生成随机高斯噪声
        noise = np.random.normal(self.gaussianMean, self.gaussianSigma, img.shape).astype(np.uint8)
        # 添加噪声到图像
        noisy_image = cv2.add(img, noise)
        return noisy_image

    def adjust_exposure(self, img):
        adjusted_image = cv2.convertScaleAbs(img, alpha=self.exposureRatio, beta=0)
        return adjusted_image

    def adjust_sharpness(self, img):
        blurred_image = cv2.GaussianBlur(img, (self.sharpness, self.sharpness), 0)
        return blurred_image

    def gamma_transform(self, img, gamma):
        is_gray = img.ndim == 2 or img.shape[1] == 1
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        illum = np.power(illum, gamma)
        v = illum * 255.
        v[v > 255] = 255
        v[v < 0] = 0
        hsv[..., 2] = v.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


def applyAug2DataPath(imgFilePath, myAug:MyImgAugmentation, applyRate=0.3):
    # img_paths = []
    for img_path in tqdm(os.listdir(imgFilePath)):
        # img_paths.append(f"{imgFilePath}/{img_path}")
        try:
            rd = random.randint(1, int(1 / applyRate))
            if rd == 1:
                img = cv2.imread(f"{imgFilePath}/{img_path}")
                # cv2.imshow('img', img)
                img = myAug.applyAugmentation(img)
                cv2.imwrite(f"{imgFilePath}/{img_path}", img)
        except:
            raise ValueError("maybe applyRate is more than 1!")


def calculate_exposure(image):
    # 将图像从BGR色彩空间转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的平均亮度值
    mean_value = cv2.mean(gray)[0]

    return mean_value


def adjust_exposure(source_image, target_image):
    # 计算源图像和目标图像的曝光度
    source_exposure = calculate_exposure(source_image)
    target_exposure = calculate_exposure(target_image)

    # 计算曝光度的调整比例
    ratio = source_exposure / target_exposure

    # 对目标图像进行曝光度调整
    adjusted_image = cv2.convertScaleAbs(target_image, alpha=ratio, beta=0)

    return adjusted_image


if __name__ == '__main__':
    myAug = MyImgAugmentation(useSharpness=True)
    applyAug2DataPath(r"D:\pycharm\object_locating\dataset_2\train\images", myAug, applyRate=0.025)
    # src = cv2.imread(r'D:\pycharm\object_locating\Datasets\ambulance_008.jpg')
    # dst = cv2.imread(r'D:\pycharm\object_locating\2.jpg')
    #
    # adj = adjust_exposure(dst, src)
    #
    # cv2.imshow('dst', adj)
    # cv2.waitKey(0)

    # img = cv2.imread(r'D:\pycharm\object_locating\Datasets\ambulance_008.jpg')
    # myAug = MyImgAugmentation(useExposure=True, exposureRatio=3.0)
    # img1 = myAug.applyAugmentation(img)
    # cv2.imshow('im', img1)
    # cv2.waitKey(0)

    # labels = np.load(r"D:\pycharm\object_locating\dataset\test\labels.npy")
    #
    # for label in labels:
    #     judge_conf = label[24:]
    #     pos = []
    #     box_r = 32
    #     for idx, point in enumerate(judge_conf):
    #         if point >= 0.5:
    #             x_offset, y_offset = label[2 * idx], label[2 * idx + 1]
    #             x, y = idx % 4 * 32 + x_offset * box_r + box_r / 2, idx // 4 * 32 + y_offset * box_r + box_r / 2  # 相对格子中心
    #             pos.append([x, y])
    #     print(np.round(pos))
    # print(labels[0])
