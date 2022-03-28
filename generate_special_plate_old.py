import errno

import cv2, os
import argparse

from skimage import exposure, img_as_float

from generate_multi_plate import MultiPlateGenerator
import random
from PIL import ImageFont, ImageDraw, Image, ImageFilter, ImageEnhance
import numpy as np
char_dict = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
char_alp = "ABCDEFGHJKLMNPQRSTUVWXYZ"
char_alp2 = "ABCDEFGHJKDDDDDDDDDDFFFFFFFFFF"
char_digit = "0123456789"
cnt = 20
yellow = 7
green = 8
# 模糊程度高
# gaussianradius = [2,2,2,2,2,2,2,2,1,3]
# 模糊程度低
gaussianradius = [2,2,2,1,2,1,2,1,2,1]
lightadius = [2,2,2,2,2,3,3,2,3,2]
flagadius = random.randint(0,2)

def parse_args():
    parser = argparse.ArgumentParser(description='中国车牌生成器')
    parser.add_argument('--double', action='store_true', default=False, help='是否双层车牌')
    parser.add_argument('--bg-color', default='green_car', help='车牌底板颜色')
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
    hsv[:, :, 0] = hsv[:, :, 0] * (0.7 + np.random.random() * 0.3)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.3 + np.random.random() * 0.7)
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
    # for j in range(yellow):
    #     if j == 0:
    #         first = random.randint(0, 30)
    #         first = char_dict[first]
    #         text = text + str(first)
    #     elif j == 1:
    #         second = random.randint(0, 23)
    #         second = char_alp[second]
    #         text = text + str(second)
    #     else:
    #         other = random.randint(0, 53)
    #         other = char_all[other]
    #         text = text + str(other)
    # return text

    for j in range(green):
        if j == 0:
            first = random.randint(0, 30)
            first = char_dict[first]
            text = text + str(first)
        elif j == 1:
            second = random.randint(0, 23)
            second = char_alp[second]
            text = text + str(second)
        elif j == 2:
            third = random.randint(0, 29)
            third = char_alp2[third]
            text = text + str(third)
        else:
            other = random.randint(0, 9)
            other = char_digit[other]
            text = text + str(other)
    return text



if __name__ == '__main__':
    args = parse_args()
    print(args)

    generator = MultiPlateGenerator('plate_model', 'font_model')

    # for i in range(cnt):
    #     RR = random.randint(200, 255)
    #     GG = random.randint(130, 200)
    #     BB = random.randint(1, 80)
    #     R = random.randint(20, 70)
    #     G = random.randint(20, 70)
    #     B = random.randint(20, 70)
    #     platenumber = get_random_plate_number()
    #     img = generator.generate_plate_special(platenumber, args.bg_color, args.double)
    #     img = cv2.resize(img, (240, 80), Image.ANTIALIAS)
    #     for p in range(80):
    #         for q in range(240):
    #             if(img[p][q][0]<10 and img[p][q][1]>185 and img[p][q][1]<195 and img[p][q][2]>245):
    #                 img[p][q][0] = BB
    #                 img[p][q][1] = GG
    #                 img[p][q][2] = RR
    #             if (img[p][q][0] < 10 and img[p][q][1] < 10 and img[p][q][2] < 10):
    #                 img[p][q][0] = B
    #                 img[p][q][1] = G
    #                 img[p][q][2] = R
    #     img = Image.fromarray(img)
    #     radiusinput = random.randint(0, 9)
    #     img = img.filter(ImageFilter.GaussianBlur(radius=gaussianradius[radiusinput]))
    #     img = np.array(img)
    #     img = img_salt_pepper_noise(img, float(random.randint(2, 12) / 100.0))
    #     print("========"+ str(i) + "========")
    #     cv2.imencode('.png', img)[1].tofile('F:\\chinese_license_plate_generator-master\\newnewplateyellowyes\\' + platenumber + '.png')


# =====================================================================================================================
    for i in range(cnt):
        # RR = random.randint(200, 255)
        # GG = random.randint(130, 200)
        # BB = random.randint(1, 80)
        light = random.randint(0,9)
        R = random.randint(50, 120)
        G = random.randint(50, 120)
        B = random.randint(50, 120)
        platenumber = get_random_plate_number()
        img = generator.generate_plate_special(platenumber, args.bg_color, args.double)
        img = cv2.resize(img, (240, 80), Image.ANTIALIAS)
        # for p in range(80):
        #     for q in range(240):
        #         if (img[p][q][0] < 10 and img[p][q][1] < 10 and img[p][q][2] < 10):
        #             img[p][q][0] = B
        #             img[p][q][1] = G
        #             img[p][q][2] = R


        # 增加高斯模糊
        img = Image.fromarray(img)
        radiusinput = random.randint(0, 9)
        img = img.filter(ImageFilter.GaussianBlur(radius=gaussianradius[radiusinput]))

        # 增加hsv处理
        img = np.array(img)
        img = tfactor(img)

        # 增加高斯噪音
        img = add_noise(img)
        imgImage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 增加图片亮度
        # if (lightadius[light] == 1):
        #     enh_bri = ImageEnhance.Brightness(imgImage)  # 增加亮度 但是有问题
        #     contrast = random.uniform(1.15, 1.3)
        #     imgImage = enh_bri.enhance(contrast)
        # if (lightadius[light] == 2):
        #     enh_bri = ImageEnhance.Brightness(imgImage)  # 减弱亮度 但是有问题
        #     contrast = random.uniform(0.6, 0.8)
        #     imgImage = enh_bri.enhance(contrast)

        # 增加照片旋转角度
        result = random.uniform(-6, 6)
        img3 = imgImage.convert('RGBA')
        img3 = img3.rotate(result)
        fff = Image.new('RGBA', img3.size, (250,)*4)
        img3 = Image.composite(img3, fff, img3)
        newimg = img3.convert("RGB")
        newimg = np.array(newimg)[:, :, ::-1]
        img = cv2.resize(newimg, (240, 80), interpolation=cv2.INTER_AREA)

        # img = np.array(img)
        # img = img_salt_pepper_noise(img, float(random.randint(2, 12) / 100.0))
        print("========"+ str(i) + "========")
        cv2.imencode('.png', img)[1].tofile('/Users/travis/PycharmProjects/plate_generator_master/greenplate/' + platenumber + '.png')


    # =====================================================================================================================
    # for i in range(cnt):
    #     RR = random.randint(175, 200)
    #     GG = random.randint(110, 130)
    #     BB = random.randint(1, 20)
    #     light = random.randint(0, 9)
    #     if flagadius == 0:
    #         R = random.randint(10, 40)
    #         G = random.randint(10, 40)
    #         B = random.randint(10, 40)
    #     else:
    #         R = random.randint(60, 125)
    #         G = random.randint(20, 50)
    #         B = random.randint(0, 30)
    #     platenumber = get_random_plate_number()
    #     img = generator.generate_plate_special(platenumber, args.bg_color, args.double)
    #     img = cv2.resize(img, (240, 80), Image.ANTIALIAS)
    #     for p in range(80):
    #         for q in range(240):
    #             if (img[p][q][0] < 10 and img[p][q][1] > 185 and img[p][q][1] < 195 and img[p][q][2] > 245):
    #                 img[p][q][0] = BB
    #                 img[p][q][1] = GG
    #                 img[p][q][2] = RR
    #             if (img[p][q][0] < 10 and img[p][q][1] < 10 and img[p][q][2] < 10):
    #                 img[p][q][0] = B
    #                 img[p][q][1] = G
    #                 img[p][q][2] = R
    #     img = Image.fromarray(img)
    #     radiusinput = random.randint(0, 9)
    #     img = img.filter(ImageFilter.GaussianBlur(radius=gaussianradius[radiusinput]))
    #     img = np.array(img)
    #     imgImage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #
    #     if (lightadius[light] == 1):
    #         # enh_con = ImageEnhance.Contrast(imgImage)
    #         # contrast = random.uniform(1.2, 1.4)
    #         # imgImage = enh_con.enhance(contrast)
    #
    #         enh_bri = ImageEnhance.Brightness(imgImage)  # 增加亮度 但是有问题
    #         contrast = random.uniform(1.1, 1.5)
    #         imgImage = enh_bri.enhance(contrast)
    #     if (lightadius[light] == 2):
    #         # enh_con = ImageEnhance.Contrast(imgImage)
    #         # contrast = random.uniform(0.58, 0.85)
    #         # imgImage = enh_con.enhance(contrast)
    #         enh_bri = ImageEnhance.Brightness(imgImage)  # 减弱亮度 但是有问题
    #         contrast = random.uniform(0.85, 1.1)
    #         imgImage = enh_bri.enhance(contrast)
    #
    #     result = random.uniform(-3.5, 3.5)
    #     img3 = imgImage.rotate(result)
    #     newimg = img3.convert("RGB")
    #     newimg = np.array(newimg)[:, :, ::-1]
    #     img = cv2.resize(newimg, (240, 80), interpolation=cv2.INTER_AREA)
    #     # img = np.array(img)
    #     # img = img_salt_pepper_noise(img, float(random.randint(2, 12) / 100.0))
    #     print("========" + str(i) + "========")
    #     cv2.imencode('.png', img)[1].tofile(
    #         'F:\\chinese_license_plate_generator-master\\newplateyellow\\' + platenumber + '.png')
