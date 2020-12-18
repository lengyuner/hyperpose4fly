from enum import Enum
import numpy as np
class UserdefPart(Enum):
    Head = 0
    Center = 1
    Tail = 2
    LWing = 3
    RWing  = 4
    Background = 5
UserdefLimb = list(zip(
    [0, 1, 1, 1, 2, 2],
    [1, 2, 3, 4, 3, 4]))

#TODO(JZ)creat automatically color to be added
UserdefColor = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0]]
# CocoColor = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
#               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
#               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


# # convert kpts from opps to mscoco
# from_opps_converter = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
#                        14: 2, 15: 1, 16: 4, 17: 3}
# # convert kpts from mscoco to opps
# to_opps_converter = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16, 5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4, 11: 11, 12: 8, 13: 12,
#                      14: 9, 15: 13, 16: 10}


# convert kpts from opps to mscoco
from_opps_converter = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
# convert kpts from mscoco to opps
to_opps_converter = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

def opps_input_converter(userdef_kpts):
    # cvt_kpts=np.zeros(shape=[19,2])

    cvt_kpts = np.zeros(shape=[len(UserdefPart), 2])#TODO(JZ)  len()+1
    transform = np.array(
        # list(zip([0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
        #          [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3])))

        list(zip([0, 1, 2, 3, 4],
                 [0, 1, 2, 3, 4])))
    xs = userdef_kpts[0::3]
    ys = userdef_kpts[1::3]
    vs = userdef_kpts[2::3]
    lost_idx = np.where(vs <= 0)[0]
    xs[lost_idx] = -1000
    ys[lost_idx] = -1000
    cvt_xs = (xs[transform[:, 0]] + xs[transform[:, 1]]) / 2
    cvt_ys = (ys[transform[:, 0]] + ys[transform[:, 1]]) / 2
    cvt_kpts[:-1, :] = np.array([cvt_xs, cvt_ys]).transpose()
    # adding background point
    cvt_kpts[-1:, :] = -1000
    return cvt_kpts


def opps_output_converter(kpt_list):
    kpts = []
    for coco_idx in list(to_opps_converter.keys()):
        model_idx = to_opps_converter[coco_idx]
        x, y = kpt_list[model_idx]
        if (x < 0 or y < 0):
            kpts += [0.0, 0.0, 0.0]
        else:
            kpts += [x, y, 2.0]
    return kpts


# convert kpts from ppn to mscoco
from_ppn_converter = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
                      14: 2, 15: 1, 16: 4, 17: 3}
# convert kpts from mscoco to ppn
to_ppn_converter = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16, 5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4, 11: 11, 12: 8, 13: 12,
                    14: 9, 15: 13, 16: 10}


def ppn_input_converter(coco_kpts):
    transform = np.array(
        list(zip([0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
                 [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3])))
    xs = coco_kpts[0::3]
    ys = coco_kpts[1::3]
    vs = coco_kpts[2::3]
    lost_idx = np.where(vs <= 0)[0]
    xs[lost_idx] = -1000
    ys[lost_idx] = -1000
    cvt_xs = (xs[transform[:, 0]] + xs[transform[:, 1]]) / 2
    cvt_ys = (ys[transform[:, 0]] + ys[transform[:, 1]]) / 2
    cvt_kpts = np.array([cvt_xs, cvt_ys]).transpose()
    return cvt_kpts


def ppn_output_converter(kpt_list):
    kpts = []
    for coco_idx in list(to_ppn_converter.keys()):
        model_idx = to_ppn_converter[coco_idx]
        x, y = kpt_list[model_idx]
        if (x < 0 or y < 0):
            kpts += [0.0, 0.0, 0.0]
        else:
            kpts += [x, y, 1.0]
    return kpts

