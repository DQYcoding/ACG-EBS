import numpy as np
from pyDOE import *

def horiver_lhs(pop, n_all, n_hori, n_ver, n1_hori, n2_hori, well_num_hori, well_num_ver, sunwf_num_hori, allbd, ecD):
    init_hori_p1 = np.empty((pop, 0))
    init_hori_p2 = np.empty((pop, 0))
    init_hori = np.empty((pop, 0))
    init_ver = np.empty((pop, 0))
    p_all = np.empty((pop, 0))
    rough_p = np.ones((pop, n_all)) * allbd[0] + lhs(n_all, samples=pop) * (np.ones((pop, n_all)) * (allbd[1] - allbd[0]))
    if well_num_hori > 0:
        init_hori = rough_p[:, :n_hori].reshape((pop, n_hori))
        init_hori_p1 = init_hori[:, :n1_hori].reshape((pop, n1_hori))
        init_hori_p2 = init_hori[:, -n2_hori:].reshape((pop, n2_hori))
    if well_num_ver > 0:
        init_ver = rough_p[:, -n_ver:].reshape((pop, n_ver))
    if well_num_hori > 0:
        init_hori_p1 = np.around(init_hori_p1)
        for i in range(pop):
            for j in range(well_num_hori):
                init_hori_p2[i][(j + 1) * sunwf_num_hori - 1] = round(init_hori_p2[i][(j + 1) * sunwf_num_hori - 1] / ecD) * ecD
    if well_num_ver > 0:
        init_ver = np.around(init_ver)
    if well_num_hori > 0:
        init_hori = np.hstack((init_hori_p1, init_hori_p2))
    p_all = np.hstack((init_hori, init_ver))
    
    return init_hori_p1, init_hori_p2, init_hori, init_ver, p_all


    