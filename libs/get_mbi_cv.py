import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import black_tophat
from skimage.morphology import white_tophat

def read_tif(path, bands):

    img_obj = gdal.Open(path)
    raw_img = img_obj.ReadAsArray(0, 0, 650, 650) #[C, H, W]

    vis_bands = []
    for band_id in range(bands):
        tmp_band = raw_img[band_id]
        print tmp_band.shape
        tmp_band = np.array(tmp_band*255.0/np.max(tmp_band), dtype=np.uint8)
        vis_bands.append(tmp_band)
    return raw_img, np.stack(vis_bands, axis=2)


def get_brightness(raw_img):

    return np.max(raw_img, axis=0)  # raw img is in [c, h, w]


def get_liner_se(theta, scale, dtype):

    rect = np.zeros((scale, scale), dtype=dtype)
    print theta, scale
    if theta == 0:
        rect = np.ones((scale, 1), dtype=dtype)
    elif theta == 90:
        np.ones((scale, 1), dtype=dtype)
    elif theta == 45:
        for i in range(scale):
            rect[i, scale-1-i] = 1
    elif theta == 135:
        for i in range(scale):
            rect[i, i] = 1
    else:
        raise ValueError(' theta must in (0, 90, 45, 135)')

    return rect


def white_tophat_linerSE(brightness_img, se):

    white_tophat = cv2.morphologyEx(brightness_img, op=cv2.MORPH_TOPHAT, kernel=se, iterations=2)

    return white_tophat


def black_tophat_linerSE(brightness_img, se):

    black_tophat = cv2.morphologyEx(brightness_img, op=cv2.MORPH_BLACKHAT, kernel=se, iterations=2)
    print 'black', black_tophat.dtype

    return black_tophat


def get_DMP(brightness_img, tophat_func, theta, scale, delta_scale):

    DTYPE = np.uint8
    tophat_0 = tophat_func(brightness_img, se=get_liner_se(theta, scale=scale, dtype=DTYPE))

    tophat_1 = tophat_func(brightness_img, se=get_liner_se(theta, scale=scale+delta_scale, dtype=DTYPE))

    print "tophat_0 dytpe", tophat_0.dtype

    tophat_0 = np.array(tophat_0, dtype=np.float32)

    tophat_1 = np.array(tophat_1, dtype=np.float32)

    DMP = np.abs(tophat_1 - tophat_0)
    return DMP


def get_mbi_and_msi(brightness_img, scale_min, scale_max, delta_scale):

    thetas = [0, 45, 90, 135]
    scales = range(scale_min, scale_max + 1, delta_scale)
    print scales

    MBI_dmp_list = []
    MSI_dmp_list = []
    for theta in thetas:
        for scale in scales:
            mbi_DMP = get_DMP(brightness_img=brightness_img, tophat_func=white_tophat_linerSE, theta=theta,
                              scale=scale, delta_scale=delta_scale)

            msi_DMP = get_DMP(brightness_img=brightness_img, tophat_func=black_tophat_linerSE, theta=theta,
                              scale=scale, delta_scale=delta_scale)
            print 'dmp dtype', mbi_DMP.dtype

            MBI_dmp_list.append(mbi_DMP)
            MSI_dmp_list.append(msi_DMP)

    MBI = sum(MBI_dmp_list) / (len(thetas) * len(scales) * 1.0)
    MSI = sum(MSI_dmp_list) / (len(thetas) * len(scales) * 1.0)
    return MBI, MSI


if __name__ == '__main__':

    raw_img, vis_img = read_tif('../data/vegas2.tif', 3)
    brightness_img = get_brightness(raw_img)

    brightness_img = np.array(brightness_img, dtype=np.float32)
    brightness_img = np.array(brightness_img*255.0/np.max(brightness_img), dtype=np.uint8)
    MBI, MSI = get_mbi_and_msi(brightness_img, scale_min=2, scale_max=52, delta_scale=5)
    # np.save('MBI.npy', MBI)
    # np.save("MSI.npy", MSI)
    plt.subplot(141)
    plt.imshow(vis_img)

    plt.subplot(142)
    plt.imshow(brightness_img, cmap='gray')

    plt.subplot(143)
    plt.imshow(MBI, cmap='gray')

    plt.subplot(144)
    plt.imshow(MSI, cmap='gray')
    print brightness_img.dtype
    plt.show()
    #
    # MBI = np.load('MBI.npy')
    # MSI = np.load('MSI.npy')
    print MBI.shape, MBI.shape

    thre_MBI = np.zeros_like(MBI)
    thre_MBI[MBI>20] = 255

    # thre_MBI = cv2.morphologyEx(thre_MBI, op=cv2.MORPH_ERODE, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    a = np.stack((MBI, MBI, MBI), 2)
    a = np.array(a*255.0/np.max(a), dtype=np.uint8)
    print a.shape
    ret, thre_MBI = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU)

    print "threshold MBI shape and dtype", thre_MBI.shape, thre_MBI.dtype

    thre_MSI = np.zeros_like(MSI)
    thre_MSI[MSI>20] = 255
    plt.figure()
    plt.subplot(121)
    plt.imshow(thre_MBI)

    plt.subplot(122)
    plt.imshow(thre_MSI)
    plt.show()