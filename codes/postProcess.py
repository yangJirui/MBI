# -*- coding: utf-8 -*-

import gdal
import numpy as np
import cv2
from geometric_utils import get_minRect

def get_gi(building_object, alpha=10.0):
    '''

    :param building_object:
    :return:
    '''
    rect = get_minRect(building_object, return_type="rect")
    rect_h, rect_w = rect[1]
    # print ("rect_h: %f, rect_w: %f" % (rect_h, rect_w))
    rect_fit = len(building_object) / (rect_h*rect_w*1.0)

    if rect_h > rect_w:
        length, width = rect_h, rect_w
    else:
        length, width = rect_w, rect_h
    len_width_ratio = length*1.0/width

    GI = alpha * rect_fit / len_width_ratio

    return GI, rect


