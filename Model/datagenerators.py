import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class Dataset(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr