import numpy as np
import copy

import cv2

def get_liner_se(theta, scale):

    rect = np.zeros((scale, scale), dtype=np.float32)
    if theta == 0:
        rect = np.ones((1,scale), dtype=np.float32)
    elif theta == 90:
        rect = np.ones((scale, 1), dtype=np.float32)
    elif theta == 45:
        for i in range(scale):
            rect[i, scale-1-i] = 1
    elif theta == 135:
        for i in range(scale):
            rect[i, i] = 1
    else:
        raise ValueError(' theta must in (0, 90, 45, 135)')

    return rect


def region_grow(raw_img, mask, row, column, object_container, scaned_val=-1, seed_val=255):

    # RecursionError: maximum recursion depth exceeded in comparison
    # python 最大递归深度是989. 所以不适合用递归写,用栈重新写

    # rows, columns = raw_img.shape[0], raw_img.shape[1]
    #
    # # add to container, get_region sum
    # object_container.append((row, column))
    # region_sum = raw_img[row, column] * 1.0
    # mask[row][column] = scaned_val
    # if (row - 1 >= 0) and mask[row-1, column] == seed_val:
    #     region_sum += region_grow(raw_img, mask, row-1, column, object_container, scaned_val, seed_val)
    # if (row + 1 < rows) and mask[row+1, column] == seed_val:
    #     region_sum += region_grow(raw_img, mask, row+1, column, object_container, scaned_val, seed_val)
    # if (column - 1 >= 0) and mask[row, column-1] == seed_val:
    #     region_sum += region_grow(raw_img, mask, row, column-1, object_container, scaned_val, seed_val)
    # if (column + 1 < columns) and mask[row, column+1] == seed_val:
    #     region_sum += region_grow(raw_img, mask, row, column+1, object_container, scaned_val, seed_val)

    rows, columns = raw_img.shape[0], raw_img.shape[1]
    region_sum = 0
    stack = [(row, column)]
    while len(stack) != 0:
        y, x = stack.pop(-1)
        region_sum += raw_img[y, x] * 1.0
        object_container.append((y, x))
        if (y - 1 >= 0) and mask[y-1, x] == seed_val:
            stack.append((y-1, x))
            mask[y-1, x] = scaned_val
        if (y + 1 < rows) and mask[y+1, x] == seed_val:
            stack.append((y+1, x))
            mask[y+1, x] = scaned_val
        if (x - 1 >= 0) and mask[y, x-1] == seed_val:
            stack.append((y, x-1))
            mask[y, x-1] = scaned_val
        if (x + 1 < columns) and mask[y, x+1] == seed_val:
            stack.append((y, x+1))
            mask[y, x+1] = scaned_val

    return region_sum


def get_img_objects(raw_mask, raw_img, filter_pixels=50, is_binary=True):
    '''
    :param img:
    :param filter_pixels:
    :param is_binary: if True, only value-255 is considered as objects
    :return: object_list [(mean, [(y,x), (y, x)...]), (mean2, [..]),...]
    '''
    object_lists = []
    rows, columns = raw_mask.shape[0], raw_mask.shape[1]
    if is_binary:
        scaned_val = 0
        old_seed_val = 255
        mask = copy.deepcopy(raw_mask)
        for row in range(rows):
            for column in range(columns):
                if mask[row, column] != scaned_val:
                    this_object = []
                    region_sum = region_grow(raw_img, mask, row, column, this_object, seed_val=old_seed_val,
                                             scaned_val=scaned_val)
                    if len(this_object) > filter_pixels:
                        region_mean = region_sum / len(this_object)
                        object_lists.append((region_mean, this_object))
    else:
        scaned_val = -1
        old_seed_val = -1
        if len(raw_mask.shape) > 1:
            raw_mask = raw_mask[:, :, 0]
        mask = np.array(raw_mask, dtype=np.int32)
        for row in range(rows):
            for column in range(columns):
                if mask[row, column] != scaned_val and mask[row, column] != old_seed_val:
                    this_object = []
                    old_seed_val = mask[row, column]
                    region_sum = region_grow(raw_img, mask, row, column, this_object, seed_val=old_seed_val,
                                             scaned_val=scaned_val)
                    if len(this_object) > filter_pixels:
                        region_mean = region_sum / len(this_object)
                        object_lists.append((region_mean, this_object))
    print("the num of total objects is : ", len(object_lists))
    del mask
    return object_lists


def get_minRect(object_container):
    '''
    :param object_container:  a list contains all the points(y, x)
    :return:
    (x1, y1)
    +----------+(x2, y2)
    |          |
    +----------+(x3, y3)
    (x0, y0)
    '''
    object_container = np.asarray(object_container)
    y, x = object_container[:, 0], object_container[:, 1]

    object_xy = np.stack((x, y), axis=1)
    rect = cv2.minAreaRect(object_xy)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box




