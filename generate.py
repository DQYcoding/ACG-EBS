import numpy as np
import random


def generate_randone_local(well_num_hori, well_num_ver, allbd_wfeat_hori, allbd_unwfeat_hori, allbd_wfeat_ver, ecD):
    hori_p = np.empty((1, 0))
    ver_p = np.empty((1, 0))
    all_p = np.empty((1, 0))
    if well_num_hori > 0:
        hori_p1 = np.zeros((1, well_num_hori * 2))
        hori_p2 = np.zeros((1, well_num_hori * 2))
        for i in range(well_num_hori):
            wi_low_hori = allbd_wfeat_hori[0, 2 * i]
            wj_low_hori = allbd_wfeat_hori[0, 2 * i + 1]
            wi_up_hori = allbd_wfeat_hori[1, 2 * i]
            wj_up_hori = allbd_wfeat_hori[1, 2 * i + 1]
            hori_p1[0, 2*i] = random.randint(wi_low_hori, wi_up_hori)
            hori_p1[0, 2*i + 1] = random.randint(wj_low_hori, wj_up_hori)
        for i in range(well_num_hori):
            alpha_low = allbd_unwfeat_hori[0, 2 * i]
            hL_low = allbd_unwfeat_hori[0, 2 * i + 1]
            alpha_up = allbd_unwfeat_hori[1, 2 * i]
            hL_up = allbd_unwfeat_hori[1, 2 * i + 1]
            hori_p2[0, 2*i] = random.uniform(alpha_low, alpha_up)
            hori_p2[0, 2*i + 1] = round(random.uniform(hL_low, hL_up) / ecD) * ecD
        hori_p = np.hstack((hori_p1, hori_p2))
    if well_num_ver > 0:
        ver_p = np.zeros((1, well_num_ver * 2))
        for i in range(well_num_ver):
            wi_low_ver = allbd_wfeat_ver[0, 2 * i]
            wj_low_ver = allbd_wfeat_ver[0, 2 * i + 1]
            wi_up_ver = allbd_wfeat_ver[1, 2 * i]
            wj_up_ver = allbd_wfeat_ver[1, 2 * i + 1]
            ver_p[0, 2*i] = random.randint(wi_low_ver, wi_up_ver)
            ver_p[0, 2*i + 1] = random.randint(wj_low_ver, wj_up_ver)
    all_p = np.hstack((hori_p, ver_p))
    return hori_p, ver_p, all_p