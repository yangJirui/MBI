# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import gdal


def with_threshold():

    mbi = np.load('../data/res/mbi_raw.npy')
    msi = np.load("../data/res/msi_raw.npy")

    # 1. threshold with mbi
    mbi_t = 40
    binary_mbi = np.zeros_like(mbi, dtype=np.uint8)
    binary_mbi[mbi > mbi_t] = 255

    # 2. threshold with msi

    msi_t = 40
    binary_msi = np.zeros_like(msi, dtype=np.uint8)
    binary_msi[msi>msi_t] = 255

    plt.subplot(121)
    plt.imshow(binary_mbi, 'gray')
    plt.subplot(122)
    plt.imshow(binary_msi, 'gray')
    plt.show()


def save_combined_res(path, save_name):
    mbi = np.load('../data/res/mbi_raw.npy')
    mbi = np.array(mbi, dtype=np.uint16)

    msi = np.load('../data/res/msi_raw.npy')
    msi = np.array(msi, dtype=np.uint16)

    # get geo-info
    img_obj = gdal.Open(path)
    im_geotrans = img_obj.GetGeoTransform()  # 仿射矩阵
    im_proj = img_obj.GetProjection()  # 地图投影信息
    raw_img = img_obj.ReadAsArray(0, 0, 650, 650)  # [C, H, W]

    combined_img = np.concatenate((raw_img, np.expand_dims(mbi, axis=0),
                                   np.expand_dims(msi, axis=0)), axis=0)

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create('../data/res/'+save_name, 650, 650, 10, gdal.GDT_UInt16)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    for i in range(10):
        dataset.GetRasterBand(i + 1).WriteArray(combined_img[i])


if __name__ == '__main__':
    # with_threshold()
    save_combined_res('../data/Four_Vegas_img96.tif', save_name='combine_mbi_msi.tif')