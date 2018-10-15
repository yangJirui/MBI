# -*- coding:utf-8 -*-
import gdal
import numpy as np
import skimage.morphology as MM
import matplotlib.pyplot as plt
import cv2

from utils import get_liner_se


def read_tif(path, H, W, band_list=[4, 2, 1]):
    '''
    band list is [1~8]. Among them, [2, 3, 5] is [B, G, R]
    raw_img [0, 1, 2, 3, 4, 5, 6, 7, 8]
    is [coastal, blue, green, yellow, red(4), red edge, near-IR1, near-IR2], dytpe is uint16
    [7] is the first NearIR
    :param path:
    :param band_list:
    :return:
    '''
    img_obj = gdal.Open(path)
    raw_img = img_obj.ReadAsArray(0, 0, H, W)  # 163 # [C, H, W]
    vis_bands = []
    print("the raw dtype is :{} ...But change it to float64.".format(raw_img.dtype))

    raw_img = np.asarray(raw_img, dtype=np.float64)
    print(raw_img.dtype)

    for band_id in band_list:
        tmp_band = raw_img[band_id]
        tmp_band = np.array(tmp_band*255.0/np.max(tmp_band), dtype=np.uint8)
        vis_bands.append(tmp_band)
    return raw_img, np.stack(vis_bands, axis=2)


def get_brightness(raw_img, selected_bands):
    '''
    :param raw_img: [C, H, W]
    :param selected_bands:
    :return:
    '''
    valid_bands = raw_img[selected_bands, :, :]
    brightness_img = np.max(valid_bands, axis=0)

    viewed_brightness_img = np.array(brightness_img * 255.0/np.max(brightness_img),
                                     dtype=np.uint8)

    return brightness_img, viewed_brightness_img


def white_hat_reconstruction(img, se):

    # 1. opening by reconstruction
    seed = MM.erosion(img, selem=se)
    mask = img

    opening_by_reconstruction = MM.reconstruction(seed=seed, mask=mask,
                                                  method='dilation',
                                                  selem=se)

    # 2. white hat reconstruction
    white_hat = img - opening_by_reconstruction

    return white_hat


def black_hat_reconstruction(img, se):
    # 1. closing by reconstruction
    dilation_img = MM.dilation(img, selem=se)

    seed = dilation_img
    mask = img

    closing_by_reconstruction = MM.reconstruction(seed=seed, mask=mask, method="erosion", selem=se)

    # 2. black hat reconstruction
    black_hat = closing_by_reconstruction - img

    return black_hat


def get_mbi(img):

    directions = [0, 45, 90, 135]
    sizes = range(52, 2-5, -5)

    total_dmp = np.zeros_like(img, dtype=np.float64)

    for d in directions:
        now_se = get_liner_se(d, scale=57)
        print (now_se)
        old_white_hat = white_hat_reconstruction(img, se=now_se)
        tmp_dmp = np.zeros_like(img, dtype=np.float64)

        for s in sizes:
            if s % 2 == 0:
                s +=1
            now_white_hat = white_hat_reconstruction(img, se=get_liner_se(d, scale=s))
            tmp_dmp += np.abs(old_white_hat - now_white_hat)
            old_white_hat = now_white_hat
        total_dmp += tmp_dmp
    mbi = total_dmp/(4*11.)

    viewed_mbi = np.asarray(mbi*255.0/np.max(mbi), dtype=np.uint8)
    return mbi, viewed_mbi


def get_msi(img):

    directions = [0, 45, 90, 135]
    sizes = range(52, 2-5, -5)

    total_dmp = np.zeros_like(img, dtype=np.float64)

    for d in directions:
        now_se = get_liner_se(d, scale=57)
        print (now_se)
        old_black_hat = black_hat_reconstruction(img, se=now_se)
        tmp_dmp = np.zeros_like(img, dtype=np.float64)

        for s in sizes:
            if s % 2==0:
                s +=1
            now_black_hat = black_hat_reconstruction(img, se=get_liner_se(d, scale=s))
            tmp_dmp += np.abs(old_black_hat - now_black_hat)
            old_black_hat = now_black_hat
        total_dmp += tmp_dmp
    msi = total_dmp/(4*11.)

    viewed_msi = np.asarray(msi*255.0/np.max(msi), dtype=np.uint8)
    return msi, viewed_msi


def get_NDVI(img):
    print("the raw_img dtype is {}, now we change it to float32".format(img.dtype))
    img = np.array(img, dtype=np.float32)

    NIR, R = img[-2], img[4]

    NDVI = (NIR - R)/(NIR + R)

    viewed_ndvi = np.array(NDVI * 255.0/np.max(NDVI), dtype=np.uint8)
    return NDVI, viewed_ndvi


def vis_some_results(img_name, save_res=False):
    raw_img, viewed_rgb = read_tif('../data/%s' % img_name,
                                   650, 650)
    # 1. get brightness img
    bright_img, viewed_brightImg = get_brightness(raw_img, selected_bands=[0, 1, 2, 3, 4])

    # 2. get NDVI
    NDVI, viewed_ndvi = get_NDVI(raw_img)

    # 3. get mbi and msi
    mbi, viewed_mbi = get_mbi(bright_img)
    msi, viewed_msi = get_msi(bright_img)

    # print (viewed_msi)
    plt.subplot(231)
    plt.imshow(viewed_rgb)
    plt.title("RGB")

    plt.subplot(232)
    plt.imshow(viewed_brightImg, 'gray')
    plt.title("brightImg")

    plt.subplot(233)
    plt.imshow(viewed_ndvi, "gray")
    plt.title("NDVI")

    plt.subplot(234)
    plt.imshow(viewed_mbi, 'gray')
    plt.title("MBI")

    plt.subplot(235)
    plt.imshow(viewed_msi, 'gray')
    plt.title("MSI")

    if save_res:
        if img_name.endswith((".tif", ".png", ".jpg")):
            img_name = ".".join(img_name.split(".")[:-1])
            print("img_name is :: ", img_name)
        np.save("../data/res/raw_data/%s_brightImg.npy" % img_name, bright_img)
        np.save("../data/res/raw_data/%s_NDVI.npy" % img_name, NDVI)
        np.save("../data/res/raw_data/%s_msi.npy" % img_name, msi)
        np.save("../data/res/raw_data/%s_mbi.npy" % img_name, mbi)

        cv2.imwrite('../data/res/viewed_data/%s_msi.png' % img_name, viewed_msi)
        cv2.imwrite('../data/res/viewed_data/%s_mbi.png' % img_name, viewed_mbi)
        cv2.imwrite("../data/res/viewed_data/%s_brightImg.png" % img_name, viewed_brightImg)
        cv2.imwrite("../data/res/viewed_data/%s_NDVI.png" % img_name, viewed_ndvi)
        plt.savefig("../data/res/viewed_data/%s_compare.png" % img_name)
    else:
        plt.show()


if __name__ == '__main__':
    print(222)
    vis_some_results(img_name="Four_Vegas_img96.tif",
                     save_res=True)




