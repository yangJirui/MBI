# -*- coding: utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt

# import osgeo

def get_brightness_img(img):
    # img is in [H, W, C] shape
    brightness_img = np.max(img, axis=-1)

    return brightness_img


def get_liner_se(theta, scale):

    rect = np.zeros((scale, scale), dtype=np.uint8)

    if theta == 0:
        rect = np.ones((scale, ), dtype=np.uint8)
    if theta == 90:
        np.ones((scale, 1), dtype=np.uint8)
    if theta == 45:
        for i in range(scale):
            rect[i, scale-1-i] = 1
    if theta == 135:
        for i in range(scale):
            rect[i, i] =1

    # rect = cv2.getStructuringElement(cv2.MORPH_CROSS, (scale, scale))
    return rect


def white_tophat_linerSE(brightness_img, se):

    white_tophat = cv2.morphologyEx(brightness_img,op=cv2.MORPH_TOPHAT, kernel=se)

    return white_tophat


def black_tophat_linerSE(brightness_img, se):

    black_tophat = cv2.morphologyEx(brightness_img, op=cv2.MORPH_BLACKHAT, kernel=se)
    # print black_tophat.dtype
    return black_tophat

def get_DMP(brightness_img, tophat_func, theta, scale, delta_scale):

    tophat_0 = tophat_func(brightness_img, se=get_liner_se(theta, scale=scale))
    tophat_1 = tophat_func(brightness_img, se=get_liner_se(theta, scale=(scale + delta_scale)) )

    tophat_0 = np.array(tophat_0, dtype=np.float32)
    tophat_1 = np.array(tophat_1, dtype=np.float32)
    DMP = np.abs(tophat_1 - tophat_0)

    return DMP


def get_mbi_and_msi(img, scale_min, scale_max, delta_scale):

    thetas =[0, 45, 90, 135]
    scales = range(scale_min, scale_max+1, delta_scale)
    # print scales
    brightness_img = get_brightness_img(img)

    MBI_dmp_list = []
    MSI_dmp_list = []
    for theta in thetas:
        for scale in scales:
            mbi_DMP = get_DMP(brightness_img=brightness_img, tophat_func=white_tophat_linerSE, theta=theta,
                          scale=scale, delta_scale=delta_scale)

            msi_DMP = get_DMP(brightness_img=brightness_img, tophat_func=black_tophat_linerSE, theta=theta,
                              scale=scale, delta_scale=delta_scale)
            # print mbi_DMP.dtype
            MBI_dmp_list.append(mbi_DMP)
            MSI_dmp_list.append(msi_DMP)

    MBI = sum(MBI_dmp_list) /(len(thetas)*len(scales)*1.0)
    MSI = sum(MSI_dmp_list) /(len(thetas)*len(scales)*1.0)
    return MBI, MSI

def laShen(img):
    Imax = np.max(img)
    Imin = np.min(img)
    MAX = 255
    MIN = 0
    img_cs = (img - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
    return img_cs

def get_building_extraction(MBI):
    # _, MBI = cv2.threshold(MBI, thresh=10, maxval=np.max(MBI), type=0)
    # _, msi = cv2.threshold(msi, thresh=8, maxval=np.max(msi), type=0)
    # MBI[MBI<10] = 0
    return MBI


def get_shadow_extraction(MSI):
    # _, mbi = cv2.threshold(mbi, thresh=10, maxval=np.max(mbi), type=0)
    # _, MSI = cv2.threshold(MSI, thresh=10, maxval=np.max(MSI), type=0)
    # laShen(MSI)

    # MSI[MSI< 0.5] = 0
    return MSI


    
if __name__ == '__main__':

    # thetas = [0, 90, 45, 135]
    # for theta in thetas:
    #     print get_liner_se(theta, 2)
    #     print 20*"_"
    import gdal
    img_8 = cv2.imread('../data/vegas2.jpg')[:, :, ::-1]
    img_obj = gdal.Open('../data/vegas2.tif')
    img = img_obj.ReadAsArray(0, 0, 650, 650)
    img = np.transpose(img, [1, 2, 0])
    # print img.dtype
    img = np.array(img, dtype=np.float32)
    bright_ness = get_brightness_img(img)
    mbi, msi = get_mbi_and_msi(img, scale_min=2, scale_max=52, delta_scale=5)

    building_extraction = get_building_extraction(MBI=mbi)

    shadow_extraction = get_shadow_extraction(MSI=msi)


    # dist = np.abs(building_extraction - shadow_extraction)
    dist = np.abs(building_extraction - shadow_extraction)

    result = np.zeros_like(dist, dtype=np.uint8)

    Tb_high = 5
    Tb_low = 0.5
    D_high = 35
    D_low = 0.5

    result[(building_extraction>=Tb_high) & (dist<D_high)] = 255
    result[(building_extraction<Tb_high) &(building_extraction>Tb_low) &(dist<D_low)] = 255
    
    plt.subplot(141)
    plt.imshow(img_8)

    plt.subplot(142)
    plt.imshow(result, cmap='gray')

    plt.subplot(143)
    plt.imshow(building_extraction, cmap='gray')

    plt.subplot(144)
    plt.imshow(shadow_extraction, cmap='gray')
    # plt.colorbar()
    plt.show()






