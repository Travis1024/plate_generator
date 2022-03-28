import errno
import cv2, os
import argparse
from skimage import exposure, img_as_float
from generate_multi_plate import MultiPlateGenerator
import random
from PIL import ImageFont, ImageDraw, Image, ImageFilter, ImageEnhance
import numpy as np
import math


char_dict = "鄂鄂鄂鄂鄂鄂京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
char_alp = "ABCDEFGHJKLMNPQRSTUVWXYZ"
char_alp2 = "ABCDEFGHJKDDFF"
char_alp3 = "ABCDEFGHJKLMNPQRSTUVWXYZ012345678901234567890123456789"
char_digit = "0123456789"
cnt = 10000
yellow = 7
green = 8

# 透视变换角度,最大角度
angle_horizontal = 15
angle_vertical = 15
angle_up_down = 15
angle_left_right = 40

# 随机透视变换
factor_up_down = 6
factor_left_right = 12
factor_true = 5

# 模糊程度高
# gaussianradius = [2,2,2,2,2,2,2,2,1,3]
# 模糊程度低
gaussianradius = [1,1,1,1,0,1,1,1,2,0]
lightadius = [1,2,2,1,2,1,2,1,2,2]
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
    for j in range(green):
        if j == 0:
            first = random.randint(0, 36)
            first = char_dict[first]
            text = text + str(first)
        elif j == 1:
            second = random.randint(0, 23)
            second = char_alp[second]
            text = text + str(second)
        elif j == 2:
            third = random.randint(0, 13)
            third = char_alp2[third]
            text = text + str(third)
        elif j == 3:
            fourth = random.randint(0, 53)
            fourth = char_alp3[fourth]
            text = text + str(fourth)
        else:
            other = random.randint(0, 9)
            other = char_digit[other]
            text = text + str(other)
    return text


def up_down_transfer(img, is_down=True, angle=None):
    """ 上下视角，默认下视角
    :param img: 正面视角原始图片
    :param is_down: 是否下视角
    :param angle: 角度
    :return:
    """
    if angle is None:
        angle = rand_reduce(angle_up_down)

    shape = img.shape
    size_src = (shape[1], shape[0])
    # 源图像四个顶点坐标
    pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
    # 计算图片进行投影倾斜后的位置
    interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
    # 目标图像上四个顶点的坐标
    if is_down:
        pts2 = np.float32([[interval, 0], [0, size_src[1]],
                            [size_src[0] - interval, 0], [size_src[0], size_src[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size_src[1]],
                            [size_src[0], 0], [size_src[0] - interval, size_src[1]]])
    # 获取 3x3的投影映射/透视变换 矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, matrix, size_src, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return dst, matrix, size_src

def left_right_transfer(img, is_left=True, angle=None):
    """ 左右视角，默认左视角
    :param img: 正面视角原始图片
    :param is_left: 是否左视角
    :param angle: 角度
    :return:
    """
    if angle is None:
        angle = rand_reduce(angle_left_right)

    shape = img.shape
    size_src = (shape[1], shape[0])
    # 源图像四个顶点坐标
    pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
    # 计算图片进行投影倾斜后的位置
    interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
    # 目标图像上四个顶点的坐标
    if is_left:
        pts2 = np.float32([[0, 0], [0, size_src[1]],
                            [size_src[0], interval], [size_src[0], size_src[1] - interval]])
    else:
        pts2 = np.float32([[0, interval], [0, size_src[1] - interval],
                            [size_src[0], 0], [size_src[0], size_src[1]]])
    # 获取 3x3的投影映射/透视变换 矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, matrix, size_src)
    return dst, matrix, size_src


def vertical_tilt_transfer(img, is_left_high=True):
    """ 添加按照指定角度进行垂直倾斜(上倾斜或下倾斜，最大倾斜角度self.angle_vertical一半）
    :param img: 输入图像的numpy
    :param is_left_high: 图片投影的倾斜角度，左边是否相对右边高
    """
    angle = rand_reduce(angle_vertical)

    shape = img.shape
    size_src = [shape[1], shape[0]]
    # 源图像四个顶点坐标
    pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])

    # 计算图片进行上下倾斜后的距离，及形状
    interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[1]))
    size_target = (int(math.cos((float(angle) / 180) * math.pi) * shape[1]), shape[0] + interval)
    # 目标图像上四个顶点的坐标
    if is_left_high:
        pts2 = np.float32([[0, 0], [0, size_target[1] - interval],
                            [size_target[0], interval], [size_target[0], size_target[1]]])
    else:
        pts2 = np.float32([[0, interval], [0, size_target[1]],
                            [size_target[0], 0], [size_target[0], size_target[1] - interval]])

    # 获取 3x3的投影映射/透视变换 矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, matrix, size_target)
    return dst, matrix, size_target

def horizontal_tilt_transfer(img, is_right_tilt=True):
    """ 添加按照指定角度进行水平倾斜(右倾斜或左倾斜，最大倾斜角度self.angle_horizontal一半）
    :param img: 输入图像的numpy
    :param is_right_tilt: 图片投影的倾斜方向（右倾，左倾）
    """
    angle = rand_reduce(angle_horizontal)
        
    shape = img.shape
    size_src = [shape[1], shape[0]]
    # 源图像四个顶点坐标
    pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
    
    # 计算图片进行左右倾斜后的距离，及形状
    interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
    size_target = (shape[1] + interval, int(math.cos((float(angle) / 180) * math.pi) * shape[0]))
    # 目标图像上四个顶点的坐标
    if is_right_tilt:
        pts2 = np.float32([[interval, 0], [0, size_target[1]],
                            [size_target[0], 0], [size_target[0] - interval, size_target[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size_target[1]],
                            [size_target[0] - interval, 0], [size_target[0], size_target[1]]])
    
    # 获取 3x3的投影映射/透视变换 矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, matrix, size_target)
    return dst, matrix, size_target


def sight_transfer(images, horizontal_sight_direction, vertical_sight_direction):
    """ 对图片进行视角变换
    :param images: 图片列表
    :param horizontal_sight_direction: 水平视角变换方向
    :param vertical_sight_direction: 垂直视角变换方向
    :return:
    """
    flag = 0
    # 左右视角
    if horizontal_sight_direction == 'left':
        flag += 1
        images, matrix, size = left_right_transfer(images, is_left=True)
    elif horizontal_sight_direction == 'right':
        flag -= 1
        images, matrix, size = left_right_transfer(images, is_left=False)
    else:
        pass
    # 上下视角
    if vertical_sight_direction == 'down':
        flag += 1
        images, matrix, size = up_down_transfer(images, is_down=True)
    elif vertical_sight_direction == 'up':
        flag -= 1
        images, matrix, size = up_down_transfer(images, is_down=False)
    else:
        pass
    
    # 左下视角 或 右上视角
    if abs(flag) == 2:
        images, matrix, size = vertical_tilt_transfer(images, is_left_high=True)
        images, matrix, size = horizontal_tilt_transfer(images, is_right_tilt=True)
    # 左上视角 或 右下视角
    elif abs(flag) == 1:
        images, matrix, size = vertical_tilt_transfer(images, is_left_high=False)
        images, matrix, size = horizontal_tilt_transfer(images, is_right_tilt=False)
    else:
        pass
    
    return images

def rand_perspective_transfer(img, factor=None, size=None):
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
    white_image = np.zeros((80,240,3), np.uint8)
    white_image[:,:,:] = 255
    dst = cv2.warpPerspective(img, matrix, (240,80), white_image, borderMode=cv2.BORDER_TRANSPARENT)
    dst = dst[upY:bottomY, leftX:rightX]
    return dst, matrix, size


# 进行透视变换
def transformation(img):
    # 水平角度
    horizontal_sight_directions = ('left', 'mid', 'right')
    # 垂直角度
    vertical_sight_directions = ('up', 'mid', 'down')

    horizontal_sight_direction = horizontal_sight_directions[random.randint(0, 2)]
    vertical_sight_direction = vertical_sight_directions[random.randint(0, 2)]

    # 对图片进行视角变换
    # img = sight_transfer(img, horizontal_sight_direction, vertical_sight_direction)
    img, matrix, size = rand_perspective_transfer(img)
    # template_image = cv2.warpPerspective(template_image, matrix, size)
    return img




if __name__ == '__main__':
    args = parse_args()
    print(args)
    generator = MultiPlateGenerator('plate_model', 'font_model')

    for i in range(cnt):
        print("========"+ str(i) + "========")
        flagindex = np.ones((80,240,3))
        light = random.randint(0,9)
        R = random.randint(5, 50)
        G = random.randint(5, 105)
        if G < 56:
            G = random.randint(5, 105)
        
        if G > 40:
            B = random.randint(G-40, G+10)
        elif G <= 40 and G > 20:
            B = random.randint(G-20, G+5)
        else:
            B = random.randint(G-10, G+5)
        
        if B < 1:
            B = 3

        # 生成车牌号码
        platenumber = get_random_plate_number()
        img = generator.generate_plate_special(platenumber, args.bg_color, args.double)
        img = cv2.resize(img, (240, 80), Image.ANTIALIAS)

        for p in range(80):
            for q in range(240):
                if (img[p][q][0] < 30 and img[p][q][1] < 30 and img[p][q][2] < 30):
                    flagindex[p][q][0] = 0
                    img[p][q][0] = max(B + random.randint(-8,8), 2)
                    img[p][q][1] = max(G + random.randint(-15,15),5)
                    img[p][q][2] = max(R + random.randint(-10,10), 5)
        # 增加高斯模糊
        img = Image.fromarray(img)
        radiusinput = random.randint(0, 9)
        img = img.filter(ImageFilter.GaussianBlur(radius=gaussianradius[radiusinput]))

        # 增加hsv处理
        img = np.array(img)
        img = tfactor(img)

        for p in range(80):
            for q in range(240):
                if (flagindex[p][q][0] == 0):
                    img[p][q][2] = random.randint(2,min(img[p][q][1]+2,50))

        # 进行透视变换
        img = transformation(img)

        # 增加高斯噪音
        # img = add_noise(img)
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
        result = random.uniform(-4, 4)
        img3 = imgImage.convert('RGBA')
        img3 = img3.rotate(result)
        fff = Image.new('RGBA', img3.size, (250,)*4)
        img3 = Image.composite(img3, fff, img3)
        newimg = img3.convert("RGB")
        newimg = np.array(newimg)[:, :, ::-1]
        img = cv2.resize(newimg, (240, 80), interpolation=cv2.INTER_AREA)

        img = np.array(img)
        img = img_salt_pepper_noise(img, float(random.randint(0, 2) / 100.0))

        print(platenumber)

        cv2.imencode('.png', img)[1].tofile('F:/SmartSite/plate_generator_master/greenplateclearness/' + platenumber + '.png')

