# -*- coding: utf-8 -*-

import cv2
import numpy as np

import matplotlib.pyplot as plt

def get_minRect(object_container):
    '''
    :param object_container:  a list contains all the points
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


def check_cross(a, b, c, d):
    '''
    a_____b  is a line
    c_______d is the another line
    {(ca X cb)*(da X db) <0 } && {(ac X ad)*(bc X bd)<0}
    http://fins.iteye.com/blog/1522259
    '''
    vector_ca, vector_cb = a - c, b - c
    vector_da, vector_db = a - d, b - d

    ca_cross_cb = np.cross(vector_ca, vector_cb)
    da_cross_db = np.cross(vector_da, vector_db)
    if ca_cross_cb * da_cross_db >= 0:
        return False
    else:
        vector_ac, vector_ad = c - a, d - a
        vector_bc, vector_bd = c - b, d -b
        ac_cross_ad = np.cross(vector_ac, vector_ad)
        bc_cross_bd = np.cross(vector_bc, vector_bd)
        if ac_cross_ad * bc_cross_bd < 0:
            return True
        else:
            return False


def get_cross_point(line1, line2):

    line1_cross_line2 = np.cross(line1, line2)

    line1_cross_line2 = line1_cross_line2 *1.0 / line1_cross_line2[-1]

    return line1_cross_line2[:2]


def distance_of_two_rect(rect1, rect2):
    '''
    :param rect1: use four points
    :param rect2:
    :return:
    (x1, y1)
    +----------+(x2, y2)
    |          |
    +----------+(x3, y3)
    (x0, y0)
    '''

    # 1. get line between two centers, line = point1 X point2
    c1 = (rect1[0] + rect1[2]) / 2.0
    c2 = (rect2[0] + rect2[2]) / 2.0
    c1_expand = np.append(c1, 1.0)
    c2_expand = np.append(c2, 1.0)
    line_c1c2 = np.cross(c1_expand, c2_expand)

    # 2. check line c1_c2 weather has cross_point between each bound of rect1
    cross_points = []
    for i in range(4):
        a, b = rect1[i % 4], rect1[(i+1)%4]
        c, d = rect2[i % 4], rect2[(i+1) %4]
        if check_cross(a, b, c1, c2):
            a_expand = np.append(a, 1.0)
            b_expand = np.append(b, 1.0)
            line_ab = np.cross(a_expand, b_expand)
            cross_point = get_cross_point(line_c1c2, line_ab)
            cross_points.append(cross_point)
        if check_cross(c, d, c1, c2):
            c_expand = np.append(c, 1.0)
            d_expand = np.append(d, 1.0)
            line_cd = np.cross(c_expand, d_expand)
            cross_point = get_cross_point(line_c1c2, line_cd)
            cross_points.append(cross_point)
    # print("the total num of cross points is : ", len(cross_points))

    if len(cross_points) < 2:
        # print("rect1 include rect2 OR rect2 include rect1")
        return 0
    elif len(cross_points) == 2:
        return np.linalg.norm(cross_points[0] - cross_points[1])
    else:
        raise ValueError("too many cross points")



if __name__ == '__main__':
    rect1 = ((200, 200), (30, 40), 0)
    rect2 = ((200, 300), (40, 80), 0)

    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)
    box1 = np.int0(box1)
    box2 = np.int0(box2)

    mask = np.zeros((650, 650, 3), dtype=np.uint8)
    cv2.drawContours(mask, [box1], -1, (0, 128, 0), 2)
    cv2.drawContours(mask, [box2], -1, (0, 0, 128), 2)
    cv2.circle(mask, (200, 200), 2, (0, 128, 0), 2)
    cv2.circle(mask, (200, 300), 2, (0, 0, 128), 2)

    print(distance_of_two_rect(box1, box2))
    plt.imshow(mask)
    plt.show()




