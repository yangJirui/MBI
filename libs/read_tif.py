import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import white_tophat
from scipy.ndimage.morphology import black_tophat
from scipy import ndimage
import cv2

def read_tif(path, band_list):
    '''
    band list is [1~8]. Among them, [2, 3, 5] is [B, G, R]
    [7] is the first NearIR
    :param path:
    :param band_list:
    :return:
    '''
    img_obj = gdal.Open(path)
    H =650
    raw_img = img_obj.ReadAsArray(0, 0, H, H) # 163 #[C, H, W]
    print raw_img.shape
    vis_bands = []
    for band_id in band_list:
        tmp_band = raw_img[band_id]
        # print tmp_band.shape
        tmp_band = np.array(tmp_band*255.0/np.max(tmp_band), dtype=np.uint8)
        vis_bands.append(tmp_band)
    return raw_img, np.stack(vis_bands, axis=2)

def get_NDVI(raw_img):

    R = raw_img[-2]
    IR = raw_img[-1]

    NDVI = (IR - R)*1.0/(IR+R)
    return NDVI
def get_brightness(raw_img, axis=-1):

    return np.max(raw_img, axis=axis) # raw img is in [c, h, w]


def get_liner_se(theta, scale):

    rect = np.zeros((scale, scale), dtype=np.float32)
    print theta, scale
    if theta == 0:
        rect = np.ones((1, scale), dtype=np.float32)
    elif theta == 90:
        np.ones((scale, 1), dtype=np.float32)
    elif theta == 45:
        for i in range(scale):
            rect[i, scale-1-i] = 1
    elif theta == 135:
        for i in range(scale):
            rect[i, i] = 1
    else:
        raise ValueError(' theta must in (0, 90, 45, 135)')

    return rect


def get_DMP(brightness_img, tophat_func, theta, scale, delta_scale):

    tophat_0 = tophat_func(brightness_img, structure=get_liner_se(theta, scale=scale))

    tophat_1 = tophat_func(brightness_img, structure=get_liner_se(theta, scale=scale+delta_scale))

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
            mbi_DMP = get_DMP(brightness_img=brightness_img, tophat_func=white_tophat, theta=theta,
                              scale=scale, delta_scale=delta_scale)

            msi_DMP = get_DMP(brightness_img=brightness_img, tophat_func=black_tophat, theta=theta,
                              scale=scale, delta_scale=delta_scale)
            print 'dmp dtype', mbi_DMP.dtype

            MBI_dmp_list.append(mbi_DMP)
            MSI_dmp_list.append(msi_DMP)

    MBI = sum(MBI_dmp_list) / (len(thetas) * len(scales) * 1.0)
    MSI = sum(MSI_dmp_list) / (len(thetas) * len(scales) * 1.0)
    return MBI, MSI

def open_reconstruction(img):
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    import skimage.morphology as MM
    se = get_liner_se(45, 50)
    # seed = np.copy(img)
    # seed[1:-1, 1:-1] = img.min()
    seed = ndimage.grey_erosion(img, structure=se)

    plt.imshow(seed, cmap='gray')
    mask = img
    plt.figure()
    dilated = MM.reconstruction(seed, mask, method='dilation')
    tophat = img - dilated
    print np.max(tophat)
    result = np.zeros_like(tophat)
    result[tophat>20] = 255
    plt.subplot(121)
    plt.imshow(tophat, cmap='gray')

    plt.subplot(122)
    plt.imshow(result, cmap='gray')
    plt.show()
    # imageE = MM.binary_erosion(img, selem=se)
    # # imageE = MM.erosion(img, selem=se)
    # # plt.subplot(131)
    # plt.figure()
    # plt.imshow(imageE)
    # plt.title('erodedImg')
    #
    # plt.show()
    #
    # plt.figure()
    # img_recon = MM.reconstruction(seed=imageE, mask=img)
    # # img_recon = MM.reconstruction(imageE, img)
    # print img_recon
    # plt.imshow(img - img_recon, cmap='gray')
    # plt.title('imageOpeningRecon')
    # plt.show()


if __name__ == '__main__':
    '''
    raw_img [0, 1, 2, 3, 4, 5, 6, 7, 8]
    is [coastal, blue, green, yellow, red(4), red edge, near-IR1, near-IR2], dytpe is uint16
    
    vis_img = [blue, green, red] # [1, 2, 4]BGR
    '''
    # raw_img, vis_img = read_tif('/home/yjr/PycharmProjects/MBI/data/Shanghai_img473.tif', band_list=[1, 2, 4])
    raw_img = cv2.imread('/home/yjr/PycharmProjects/MBI/data/shanghai.jpg')
    # vis_img = raw_img
    calu_brightness_img = raw_img#[:5]
    brightness_img = get_brightness(calu_brightness_img, axis=-1)
    # brightness_img = np.array(brightness_img*255.0/np.max(brightness_img), dtype=np.uint8)
    open_reconstruction(brightness_img)
    # brightness_img = np.array(brightness_img, dtype=np.float32)
    # MBI, MSI = get_mbi_and_msi(brightness_img, scale_min=2, scale_max=52, delta_scale=5)
    # np.save('MBI.npy', MBI)
    # np.save("MSI.npy", MSI)
    # plt.subplot(141)
    # plt.imshow(vis_img[:, :, ::-1])
    # #
    # plt.subplot(142)
    # plt.imshow(brightness_img, cmap='gray')
    # #
    # plt.subplot(143)
    # plt.imshow(MBI, cmap='gray')
    # #
    # plt.subplot(144)
    # plt.imshow(MSI, cmap='gray')


    # print brightness_img.dtype
    # plt.show()
    # # #
    # MBI = np.load('MBI.npy')
    # MSI = np.load('MSI.npy')
    # print np.max(MBI), np.min(MSI)
    # # # MBI = np.array(MBI*255.0/np.max(MBI), dtype=np.uint8)
    # # # MSI = np.array(MSI * 255.0 / np.max(MSI), dtype=np.uint8)
    # # # # print MBI.shape, "MBI max and min", MBI.shape, np.max(MBI), np.min(MBI)
    # # # # print MSI.shape, "MSI max and min", MSI.shape, np.max(MSI), np.min(MSI)
    # thre_MBI = MBI.copy()
    # # # thre_MBI[MBI>30] = 255
    # # # thre_MBI[MBI<=30] = 0
    # # # # print "threshold MBI shape and dtype", thre_MBI.shape, thre_MBI.dtype
    # # # #
    # # # # # thre_MSI = MSI[MSI > 2].reshape(650, 650)
    # thre_MSI = MSI.copy()
    # # # # # thre_MSI[MSI>2.0] = 255
    # # # # # thre_MSI[MSI<=2.0] = 0
    # # #
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(thre_MBI, cmap='gray')
    # # #
    # plt.subplot(132)
    # plt.imshow(thre_MSI, cmap='gray')
    # # plt.show()
    # #
    # plt.subplot(133)
    # result = np.zeros_like(MBI)
    # #
    # D = np.abs(MBI - MSI)
    # print "D max and min", np.max(D), np.min(D)
    # # #
    # Tb_high = 2
    # D_high = 35
    # Tb_low = 0.5
    # D_low = 10
    # result[(MBI>=Tb_high)&(D<D_high)] = 255
    # result[(Tb_low<=MBI)&(MBI<Tb_high) & (D<D_low)] = 255
    # # #
    # plt.imshow(result, cmap='gray')
    #
    # # res2 = watershed(MBI)
    #
    # # markers, result = watershed_cv(MBI)
    # #
    # # markers_msi, resutl_msi = watershed_cv(MSI)
    # #
    # # print markers
    # # print 20*"_+"
    # # print result
    # # plt.subplot(133)
    # # plt.imshow(markers, cmap='gray')
    # plt.show()