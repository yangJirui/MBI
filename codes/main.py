# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import gdal

from utils import get_img_objects
from geometric_utils import get_minRect
from cal_distance import get_shadows, filter_mbi
from postProcess import get_gi
import matplotlib.pyplot as plt


TG = 1.1

def extract_building_in_aimg(rgb_path, img_name):
    if img_name.endswith((".tif", ".png", ".jpg")):
        img_name = ".".join(img_name.split(".")[:-1])
        print("img_name is :: ", img_name)
    rgb_img = cv2.imread(rgb_path)
    bright_img = np.load("../data/res/raw_data/%s_brightImg.npy" % img_name)
    msi = np.load("../data/res/raw_data/%s_msi.npy" % img_name)
    NDVI = np.load('../data/res/raw_data/%s_NDVI.npy' % img_name)
    mbi = np.load("../data/res/raw_data/%s_mbi.npy" % img_name)
    # seg_res = cv2.imread("../data/res/export/%s_SegRes.png" % img_name)
    # seg_res = cv2.imread("/home/yjr/PycharmProjects/MBI_win/image-segmentation/out3.png")
    seg_res = cv2.imread('/home/yjr/PycharmProjects/MBI_win/image-segmentation/seg_res/TIF.png')

    mbi_object_list = get_img_objects(seg_res, mbi, is_binary=False)
    shadow_objects = get_shadows(bright_img, msi=msi, NDVI=NDVI)

    building_objects, shadow_objects = filter_mbi(mbi_object_list, shadow_objects)


    # postProcess
    mask = np.zeros(shape=(650, 650, 3), dtype=np.uint8)  # to identify buildings and shadows

    for a_object in building_objects["high_mbi"]:
        for row, column in a_object:
            mask[row, column] = (255, 0, 0)
    for a_object in building_objects["low_mbi"]:
        for row, column in a_object:
            mask[row, column] = (0, 128, 0)

    # 1.filter NDVI > T1()
    mask[NDVI > 0.15] = (0, 0, 0)

    # for _, a_object in shadow_objects:
    #     for row, column in a_object:
    #         mask[row, column] = (0, 0, 0)
            # cv2.drawContours(mask, [rect], -1, (0, 0, 255), 2)

    building_mask = np.array(np.max(mask, axis=-1) >= 128, dtype=np.uint8) * 255
    buildings_before_gi = get_img_objects(building_mask, building_mask)

    # 2. filter with GI

    detections = []
    for _, a_building in buildings_before_gi:
        gi, rect = get_gi(a_building)
        if gi < TG:
            # that is not building
            for row, column in a_building:
                building_mask[row, column] = 0  # should be zero
        else:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            detections.append(box)
            cv2.drawContours(rgb_img, [box], -1, (30, 0, 255), 2)
    # plt.subplot(121)
    # plt.imshow(building_mask, 'gray')
    # plt.subplot(122)
    # plt.imshow(rgb_img[:, :, ::-1])

    # plt.show()
    cv2.imwrite('../data/res/extracted_buildings/imgs/%s_mask.png' % img_name, building_mask)
    cv2.imwrite('../data/res/extracted_buildings/imgs/%s_box.png' % img_name, rgb_img)
    np.save("../data/res/extracted_buildings/rects/%s_rects.npy" % img_name, np.array(detections))
    return detections


def extract_building_for_many_imgs(imglist_path):
    ROOT_PATH = "/home/yjr/DataSet/SpaceNet"

    def get_name(all_name):
        name = all_name.strip().split("PanSharpen_")[1].split(".jpg")[0]
        root_name = name.split("_img")[0]

        return root_name, name

    with open(imglist_path) as f:
        img_list = f.readlines()

    for i, a_name in enumerate(img_list):
        dataset_name, name = get_name(a_name)
        img_path = os.path.join("/home/yjr/DataSet/SpaceNet/test_imgs", "RGB-PanSharpen_%s.jpg" % name)
        extract_building_in_aimg(rgb_path=img_path, img_name=name)
        print (i, name)
if __name__ == '__main__':
    # extract_building_in_aimg(rgb_path='/home/yjr/PycharmProjects/MBI_win/MBI/data/vegas.jpg',
    #                          img_name='Four_Vegas_img96')
    extract_building_for_many_imgs("/home/yjr/DataSet/SpaceNet/AOall_pascal/test_imgs_list.txt")