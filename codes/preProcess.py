import gdal
import numpy as np
import skimage.morphology as MM
import matplotlib.pyplot as plt

from codes.utils import get_liner_se

def read_tif(path, H, W, band_list=[2, 3, 5]):
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


def get_mbi(img):

    directions = [0, 45, 90, 135]
    sizes = range(52, 2-5, -5)

    total_dmp = np.zeros_like(img, dtype=np.float64)

    for d in directions:
        old_white_hat = white_hat_reconstruction(img, se=get_liner_se(d, scale=57))
        tmp_dmp = np.zeros_like(img, dtype=np.float64)

        for s in sizes:
            now_white_hat = white_hat_reconstruction(img, se=get_liner_se(d, scale=s))
            tmp_dmp += np.abs(old_white_hat - now_white_hat)
            old_white_hat = now_white_hat
        total_dmp += tmp_dmp
    mbi = total_dmp/(4*11.)

    viewed_mbi = mbi*255.0/np.max(mbi)
    return mbi, viewed_mbi


def vis_some_results():

    raw_img, viewed_rgb = read_tif('C:\python\python工程\MBI\data\Four_Vegas_img96.tif',
                                   650, 650)

    bright_img, viewed_brightImg = get_brightness(raw_img, selected_bands=[0, 1, 2, 3, 4])

    mbi, viewed_mbi = get_mbi(bright_img)

    plt.subplot(131)
    plt.imshow(viewed_rgb)
    plt.subplot(132)
    plt.imshow(viewed_brightImg)

    plt.subplot(133)
    plt.imshow(viewed_mbi)

if __name__ == '__main__':
    print(222)
    vis_some_results()



