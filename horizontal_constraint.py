import numpy as np
import horizontal_judge
import horizontal_operate


def doroughtrial(trial, Num, ecD):
    popsize = trial.shape[0]
    well_num_hori = Num['well_num_hori']
    well_num_ver = Num['well_num_ver']
    n_hori = Num['n_hori']
    n_ver = Num['n_ver']
    n1_hori = Num['n1_hori']
    n2_hori = Num['n2_hori']
    sunwf_num_hori = Num['sunwf_num_hori']
    trial_hori_p1 = np.empty((popsize, 0))
    trial_hori_p2 = np.empty((popsize, 0))
    trial_hori = np.empty((popsize, 0))
    trial_ver = np.empty((popsize, 0))
    trial_all = np.empty((popsize, 0))

    if well_num_hori > 0:
        trial_hori = trial[:, :n_hori].reshape((popsize, n_hori))
        #
        trial_hori_p1 = trial_hori[:, :n1_hori].reshape((popsize, n1_hori))
        trial_hori_p2 = trial_hori[:, -n2_hori:].reshape((popsize, n2_hori))
    if well_num_ver > 0:
        trial_ver = trial[:, -n_ver:].reshape((popsize, n_ver))
    if well_num_hori > 0:
        trial_hori_p1 = np.around(trial_hori_p1)
        for i in range(popsize):
            for j in range(well_num_hori):
                trial_hori_p2[i][(j + 1) * sunwf_num_hori - 1] = round(trial_hori_p2[i][(j + 1) * sunwf_num_hori - 1] / ecD) * ecD
    if well_num_ver > 0:
        trial_ver = np.around(trial_ver)

    if well_num_hori > 0:
        trial_hori = np.hstack((trial_hori_p1, trial_hori_p2))
    trial_all = np.hstack((trial_hori, trial_ver))

    return trial_hori_p1, trial_hori_p2, trial_hori, trial_ver, trial_all