import errno

import cv2, os
import argparse

from skimage import exposure, img_as_float

from generate_multi_plate import MultiPlateGenerator
import random
from PIL import ImageFont, ImageDraw, Image, ImageFilter, ImageEnhance
import numpy as np
#char_dict = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
char_dict = "鄂鄂京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
char_alp = "ABCDEFGHJKLMNPQRSTUVWXYZ"
char_all = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789012345678901234567890123456789"
cnt = 2000
yellow = 7
green = 8


# 随机透视变换
factor_up_down = 6
factor_left_right = 18
factor_true = 5


# 模糊程度高
# gaussianradius = [2,2,2,2,2,2,2,2,1,3]
# 模糊程度低
gaussianradius = [0,1,1,1,1,0,1,0,1,0]
lightadius = [1,2,2,1,2,1,2,1,2,2]
flagadius = random.randint(0,2)

def parse_args():
    parser = argparse.ArgumentParser(description='中国车牌生成器')
    parser.add_argument('--double', action='store_true', default=False, help='是否双层车牌')
    parser.add_argument('--bg-color', default='yellow', help='车牌底板颜色')
    parser.add_argument('--plate-number', default='湘AD9999', help='车牌号码')
    args = parser.parse_args()
    return args


def rand_reduce(val):
    return int(np.random.random() * val)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def add_single_channel_noise(single):
    """ 添加高斯噪声
    :param single: 单一通道的图像数据
    """
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + rand_reduce(4), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = diff * noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst

def add_noise(img):
    """添加噪声"""
    img[:, :, 0] = add_single_channel_noise(img[:, :, 0])
    img[:, :, 1] = add_single_channel_noise(img[:, :, 1])
    img[:, :, 2] = add_single_channel_noise(img[:, :, 2])
    return img



def tfactor(img):
    """
    添加饱和度、亮度的噪声
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    # hsv[:, :, 1] = hsv[:, :, 1] * (0.5 + np.random.random() * 0.5)
    # hsv[:, :, 2] = hsv[:, :, 2] * (0.6 + np.random.random() * 0.4)

    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.7 + np.random.random() * 0.3)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def img_salt_pepper_noise(src, percetage):
    """
    增加椒盐噪音
    """
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg

def get_random_plate_number():
    text = ''
    for j in range(yellow):
        if j == 0:
            first = random.randint(0, 32)
            first = char_dict[first]
            text = text + str(first)
        elif j == 1:
            second = random.randint(0, 23)
            second = char_alp[second]
            text = text + str(second)
        else:
            other = random.randint(0, 63)
            other = char_all[other]
            text = text + str(other)
    return text



def rand_perspective_transfer(img, flagbackbound, factor=None, size=None):
    """ 添加投影映射畸变
    :param img: 输入图像的numpy
    :param factor: 畸变的参数
    :param size: 图片的目标尺寸，默认维持不变
    """
    if factor is None:
        factor = factor_true
    if size is None:
        size = (img.shape[1], img.shape[0])
    shape = size
    # 源图像四个顶点坐标
    pts1 = np.float32([[0, 0], [shape[0]-1, 0], [shape[0]-1, shape[1]-1], [0, shape[1]-1]])
    # 目标图像上四个顶点的坐标
    leftupX = rand_reduce(factor_left_right)
    leftupY = rand_reduce(factor_up_down)
    rightupX = shape[0] - 1 - rand_reduce(factor_left_right)
    rightupY = rand_reduce(factor_up_down)
    rightbottomX = shape[0] - 1 - rand_reduce(factor_left_right)
    rightbottomY = shape[1] - 1 - rand_reduce(factor_up_down)
    leftbottomX = rand_reduce(factor_left_right)
    leftbottomY = shape[1] - 1 - rand_reduce(factor_up_down)

    leftX = min(leftupX,leftbottomX)
    upY = min(leftupY,rightupY)
    rightX = max(rightupX,rightbottomX)
    bottomY = max(rightbottomY,leftbottomY)

    pts2 = np.float32([[leftupX, leftupY],
                        [rightupX, rightupY],
                        [rightbottomX, rightbottomY],
                        [leftbottomX, leftbottomY]])
    

    # print(pts1)
    # print(pts2)
    # 获取 3x3的投影映射/透视变换 矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 利用投影映射矩阵，进行透视变换
    if flagbackbound == 1:
        white_image = np.zeros((80,240,3), np.uint8)
        white_image[:,:,:] = 245
        dst = cv2.warpPerspective(img, matrix, (240,80), white_image, borderMode=cv2.BORDER_TRANSPARENT)
    else:
        black_image = np.zeros((80,240,3), np.uint8)
        black_image[:,:,:] = 10
        dst = cv2.warpPerspective(img, matrix, (240,80), black_image, borderMode=cv2.BORDER_TRANSPARENT)

    dst = dst[upY:bottomY, leftX:rightX]
    dst = cv2.resize(dst, (240, 80), interpolation=cv2.INTER_AREA)
    return dst, matrix, size


# 进行透视变换
def transformation(img, flagbackbound):
    # 水平角度
    horizontal_sight_directions = ('left', 'mid', 'right')
    # 垂直角度
    vertical_sight_directions = ('up', 'mid', 'down')

    horizontal_sight_direction = horizontal_sight_directions[random.randint(0, 2)]
    vertical_sight_direction = vertical_sight_directions[random.randint(0, 2)]

    # 对图片进行视角变换
    # img = sight_transfer(img, horizontal_sight_direction, vertical_sight_direction)
    img, matrix, size = rand_perspective_transfer(img, flagbackbound)
    # template_image = cv2.warpPerspective(template_image, matrix, size)
    return img



if __name__ == '__main__':
    args = parse_args()
    print(args)

    generator = MultiPlateGenerator('plate_model', 'font_model')


# =====================================================================================================================
    for i in range(cnt):
        print("========"+ str(i) + "========")
        flagindex = np.ones((80,240,3))
        light = random.randint(0,9)
        R = random.randint(42, 140)
        if R < 60:
            R = random.randint(42, 140)

        if R > 100:
            G = random.randint(R-40-10, R-40+20)
        else:
            G = random.randint(R-28-10, R-28+10)

        B = random.randint(0, 30)

        platenumber = get_random_plate_number()
        img = generator.generate_plate_special(platenumber, args.bg_color, args.double)
        img = cv2.resize(img, (240, 80), Image.ANTIALIAS)
        for p in range(80):
            for q in range(240):
                if (img[p][q][0] < 30 and img[p][q][1] < 30 and img[p][q][2] < 30):
                    flagindex[p][q][0] = 0
                    img[p][q][2] = R + random.randint(-20,20)
                    img[p][q][1] = min(G + random.randint(-12,12),img[p][q][2]-20)
                    if img[p][q][1] < 1:
                        img[p][q][1] = 6
                    img[p][q][0] = max(B + random.randint(-10,10), 1)

        # 增加高斯模糊
        # 增加高斯模糊
        img = Image.fromarray(img)
        radiusinput = random.randint(0, 9)
        img = img.filter(ImageFilter.GaussianBlur(radius=gaussianradius[radiusinput]))

        if (lightadius[light] == 1):
            enh_bri = ImageEnhance.Brightness(img)  # 增加亮度 但是有问题
            contrast = random.uniform(0.85, 1.40)
            img = enh_bri.enhance(contrast)
        if (lightadius[light] == 2):
            enh_bri = ImageEnhance.Brightness(img)  # 减弱亮度 但是有问题
            contrast = random.uniform(0.55, 1.20)
            img = enh_bri.enhance(contrast)

        # 增加hsv处理
        img = np.array(img)
        # img = tfactor(img)

        # for p in range(80):
        #     for q in range(240):
        #         if (flagindex[p][q][0] == 0):
        #             img[p][q][2] = R + random.randint(-20,20)
        #             img[p][q][1] = min(G + random.randint(-12,12),img[p][q][2]-20)
        #             if img[p][q][1] < 1:
        #                 img[p][q][1] = 6
        #             img[p][q][0] = max(B + random.randint(-10,10), 1)

        img = tfactor(img)


        # 设置黑白填充背景，0为黑，1为白
        flagbackbound = random.randint(0,1)
        # 进行透视变换
        img = transformation(img, flagbackbound)
        # 增加高斯噪音
        # img = add_noise(img)
        imgImage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        #增加图片亮度
        # if (lightadius[light] == 1):
        #     enh_bri = ImageEnhance.Brightness(imgImage)  # 增加亮度 但是有问题
        #     contrast = random.uniform(1.0, 1.5)
        #     imgImage = enh_bri.enhance(contrast)
        # if (lightadius[light] == 2):
        #     enh_bri = ImageEnhance.Brightness(imgImage)  # 减弱亮度 但是有问题
        #     contrast = random.uniform(0.6, 1.0)
        #     imgImage = enh_bri.enhance(contrast)


        # 增加照片旋转角度
        result = random.uniform(-4, 4)
        img3 = imgImage.convert('RGBA')
        img3 = img3.rotate(result)
        if flagbackbound == 1:
            fff = Image.new('RGBA', img3.size, (245,)*4)
        else:
            fff = Image.new('RGBA', img3.size, (10,)*4)
        img3 = Image.composite(img3, fff, img3)
        newimg = img3.convert("RGB")
        newimg = np.array(newimg)[:, :, ::-1]
        img = cv2.resize(newimg, (240, 80), interpolation=cv2.INTER_AREA)

        img = np.array(img)
        img = img_salt_pepper_noise(img, float(random.randint(0, 2) / 100.0))
        print(platenumber)
        cv2.imencode('.png', img)[1].tofile('F:/SmartSite/plate_generator_master/yellowplate4/' + platenumber + '.png')

