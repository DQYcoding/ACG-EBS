import boundary
import copy
import numpy as np

def singw_feat(Wij, alpha_low, alpha_up, hL_low, hL_up):
    wi_low_hori = Wij['wi_low_hori']
    wi_up_hori = Wij['wi_up_hori']
    wj_low_hori = Wij['wj_low_hori']
    wj_up_hori = Wij['wj_up_hori']
    wi_low_ver = Wij['wi_low_ver']
    wi_up_ver = Wij['wi_up_ver']
    wj_low_ver = Wij['wj_low_ver']
    wj_up_ver = Wij['wj_up_ver']
    bd_wfeat_hori = boundary.well_bounds(wi_low_hori, wi_up_hori, wj_low_hori, wj_up_hori)
    bd_unwfeat_hori = boundary.unwell_bounds(alpha_low, alpha_up, hL_low, hL_up)
    bd_wfeat_ver = boundary.well_bounds(wi_low_ver, wi_up_ver, wj_low_ver, wj_up_ver)
    return bd_wfeat_hori, bd_unwfeat_hori, bd_wfeat_ver


def allw_feat(well_num_hori, well_num_ver, bd_wfeat_hori, bd_unwfeat_hori, bd_wfeat_ver):
    allbd_wfeat_hori = np.empty((2, 0))
    allbd_unwfeat_hori = np.empty((2, 0))
    allbd_wfeat_ver = np.empty((2, 0))
    allbd_hori = np.empty((2, 0))
    allbd_ver = np.empty((2, 0))
    allbd = np.empty((2, 0))
    if well_num_hori > 0:
        allbd_wfeat_hori = copy.deepcopy(bd_wfeat_hori)
        for i in range(well_num_hori - 1):
            allbd_wfeat_hori = np.hstack([allbd_wfeat_hori, bd_wfeat_hori])
        allbd_unwfeat_hori = copy.deepcopy(bd_unwfeat_hori)
        for i in range(well_num_hori - 1):
            allbd_unwfeat_hori = np.hstack([allbd_unwfeat_hori, bd_unwfeat_hori])
        allbd_hori = np.hstack([allbd_wfeat_hori, allbd_unwfeat_hori])
    if well_num_ver > 0:
        allbd_wfeat_ver = copy.deepcopy(bd_wfeat_ver)
        for i in range(well_num_ver - 1):
            allbd_wfeat_ver = np.hstack((allbd_wfeat_ver, bd_wfeat_ver))
        allbd_ver = allbd_wfeat_ver
    allbd = np.hstack((allbd_hori, allbd_ver))

    return allbd_wfeat_hori, allbd_unwfeat_hori, allbd_hori, allbd_wfeat_ver, allbd_ver, allbd