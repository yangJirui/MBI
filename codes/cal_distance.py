# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import get_img_objects
from geometric_utils import get_minRect
from geometric_utils import distance_of_two_rect

TS = 0.5  # 30
TB_HIGH = 2 # 70
TB_LOW = 0.5  # 35
D_HIGH = 20
D_LOW = 10


def get_shadows(bright_img, msi, NDVI):
    msi_threshold = TS
    bright_threshold = 10000
    ndvi_threshold = 0.15
    binary_res = np.zeros_like(bright_img, dtype=np.uint8)

    binary_res[(msi >= msi_threshold) &
               (bright_img <= bright_threshold) &
               (NDVI <= ndvi_threshold)] = 255
    object_lists = get_img_objects(binary_res, binary_res)
    return object_lists


def filter_mbi(mbi_object_list, shadow_objects):

    building_objects = {"high_mbi": [],
                        "low_mbi": []}
    for a_object in mbi_object_list:
        mean, object_pixels = a_object
        if mean > TB_LOW:
            object_rect = get_minRect(object_pixels)
            min_dist = 10000
            for a_shadow in shadow_objects:
                _, shadow_pixels = a_shadow
                shadow_rect = get_minRect(shadow_pixels)
                min_dist = min(min_dist, distance_of_two_rect(shadow_rect, object_rect))

            if mean > TB_HIGH and min_dist < D_HIGH:
                building_objects["high_mbi"].append(object_pixels)
            elif min_dist < D_LOW:
                building_objects["low_mbi"].append(object_pixels)
    return building_objects, shadow_objects


def test_getShadows(img_name):

    if img_name.endswith((".tif", ".png", ".jpg")):
        img_name = ".".join(img_name.split(".")[:-1])
        print("img_name is :: ", img_name)
    bright_img = np.load("../data/res/raw_data/%s_brightImg.npy" % img_name)
    msi = np.load("../data/res/raw_data/%s_msi.npy" % img_name)
    NDVI = np.load('../data/res/raw_data/%s_NDVI.npy' % img_name)

    shadows_list = get_shadows(bright_img, msi=msi, NDVI=NDVI)

    filter_mask = np.zeros_like(msi, dtype=np.uint8)
    rect_list = []
    for a_object in shadows_list:
        mean, object_container = a_object
        for x, y in object_container:
            filter_mask[x, y] = 255
        box = get_minRect(object_container)
        rect_list.append(box)
    color_mask = np.stack((filter_mask, filter_mask, filter_mask), axis=2)
    print(color_mask.shape)
    for a_box in rect_list:
        cv2.drawContours(color_mask, [a_box], -1, (255, 0, 0), 2)
    plt.imshow(color_mask)
    plt.show()


def test_filter_mbi(img_name):
    if img_name.endswith((".tif", ".png", ".jpg")):
        img_name = ".".join(img_name.split(".")[:-1])
        print("img_name is :: ", img_name)

    building_objects, shadow_objects = filter_mbi(img_name)
    mask = np.zeros(shape=(650, 650, 3), dtype=np.uint8)

    for a_object in building_objects["high_mbi"]:
        rect = get_minRect(a_object)
        for row, column in a_object:
            mask[row, column] = (255, 0, 0)
        # cv2.drawContours(mask, [rect], -1, (255, 0, 0), 2)
    for a_object in building_objects["low_mbi"]:
        rect = get_minRect(a_object)
        for row, column in a_object:
            mask[row, column] = (0, 128, 0)
        # cv2.drawContours(mask, [rect], -1, (0, 128, 0), 2)
    for _, a_object in shadow_objects:
        rect = get_minRect(a_object)
        for row, column in a_object:
            mask[row, column] = (0, 0, 0)
        # cv2.drawContours(mask, [rect], -1, (0, 0, 255), 2)

    building_mask = np.array(np.max(mask, axis=-1) >=128, dtype=np.uint8) * 255
    print(building_mask.shape)
    final_buildings = get_img_objects(building_mask, building_mask)

    viewed_img = cv2.imread('../data/res/viewed_data/%s_brightImg.png' % img_name)
    if len(viewed_img.shape) < 3:
        viewed_img = np.stack((viewed_img, viewed_img, viewed_img), axis=2)
    for _, a_building in final_buildings:
        box = get_minRect(a_building)
        cv2.drawContours(viewed_img, [box], -1, (0, 0, 255), 2)
    plt.subplot(131)
    plt.imshow(mask)
    plt.subplot(132)
    plt.imshow(building_mask, "gray")
    plt.subplot(133)
    plt.imshow(viewed_img)
    plt.show()


if __name__ == '__main__':
    # test_getShadows("Four_Vegas_img96")
    test_filter_mbi("Four_Vegas_img96")


