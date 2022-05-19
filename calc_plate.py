# -*- coding: utf-8 -*-

import os
import random
import shutil

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘",
             "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
             "X", "Y", "Z"]
ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
       "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

dic={}

class BatchRenamePics(object):
    def __init__(self, path):
        # 设置起始路径path
        self.path = path

    def calc(self):
        number = 0
        for file in os.listdir(self.path):
            filepath = str(file)
            for i in range(7):
                if filepath[i] not in dic:
                    dic[filepath[i]] = 1
                else:
                    dic[filepath[i]]+=1
            print(number)
            print(filepath)
            number = number + 1


if __name__ == '__main__':
    # 设置起始路径path
    path = r'F:/SmartSite/plate_generator_master/yellowplate4/'
    # 创建实例对象
    pics = BatchRenamePics(path)
    # 调用实例方法
    pics.calc()
    a = sorted(dic.items())
    print(a)