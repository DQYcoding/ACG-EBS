import numpy as np
import os
from os.path import join as fullfile
import horizontal_operate as ho
import file_operate as fo


def rich_i_simulation(simupath, iinit_hori_p1, iinit_hori_p2, pp_ver, Hwtraj, Num, casename, mode, getgrid_mode):
    inj_num_hori = Num['inj_num_hori']
    inj_num_ver = Num['inj_num_ver']
    pro_num_hori = Num['pro_num_hori']
    pro_num_ver = Num['pro_num_ver']
    swf_num_hori = Num['swf_num_hori']
    sunwf_num_hori = Num['sunwf_num_hori']
    sampop = iinit_hori_p1.shape[0]
    well_num_hori = inj_num_hori + pro_num_hori
    hori_X_xi, hori_X_yi = ho.horiw_split1(iinit_hori_p1, well_num_hori, swf_num_hori)
    _, hori_X_L = ho.horiw_split2(iinit_hori_p2, well_num_hori, sunwf_num_hori)
    well_num_ver = inj_num_ver + pro_num_ver
    n_cols = pp_ver.shape[1]
    ver_X_xi = pp_ver[:, 0:n_cols:2].reshape((sampop, well_num_ver))
    ver_X_yi = pp_ver[:, 1:n_cols:2].reshape((sampop, well_num_ver))
    #
    obj_mode = 'NPV_hori'
    nsres = np.zeros([sampop, 1])
    rich_record = []
    for i in range(sampop):
        samp_list = []
        if getgrid_mode == 'center-4':
            hori_samplex = np.empty((1, 0))
            hori_sampley = np.empty((1, 0))
            for j in range(len(Hwtraj[i])):
                hori_samplex = np.hstack((hori_samplex, np.array([[Hwtraj[i][j][0][0]]])))
                hori_sampley = np.hstack((hori_sampley, np.array([[Hwtraj[i][j][0][1]]])))
        #
        hori_sampleL = hori_X_L[i, :].reshape(1, well_num_hori)
        #
        if well_num_hori > 0:
            hori_samtraj = Hwtraj[i]
        else:
            hori_samtraj = []
        ver_samplex = ver_X_xi[i, :].reshape(1, well_num_ver)
        ver_sampley = ver_X_yi[i, :].reshape(1, well_num_ver)
        FOPT, FWPT, FWIT, samp_res = rich_simulation(simupath, hori_samplex, hori_sampley, hori_samtraj, ver_samplex, ver_sampley, inj_num_hori, pro_num_hori, inj_num_ver, pro_num_ver, casename, mode, hori_sampleL)
        samp_list = samp_res
        samp_list.append(FOPT)
        samp_list.append(FWPT)
        samp_list.append(FWIT)
        if obj_mode == 'NPV_hori':
            Po = 80
            Cwp = 4.8
            Cwi = 4.8
            Chori = 7500
            total_L = np.sum(hori_sampleL)
            NPV = Po * FOPT - FWPT * Cwp - FWIT * Cwi - total_L/100*Chori
            nsres[i] = NPV
            samp_list.append(NPV)
        rich_record.append(samp_list)

    return nsres, rich_record


def rich_s_simulation(simupath, iinit_hori_p1, iinit_hori_p2, pp_ver, Hwtraj, Num, casename, mode, getgrid_mode):
    inj_num_hori = Num['inj_num_hori']
    inj_num_ver = Num['inj_num_ver']
    pro_num_hori = Num['pro_num_hori']
    pro_num_ver = Num['pro_num_ver']
    swf_num_hori = Num['swf_num_hori']
    sunwf_num_hori = Num['sunwf_num_hori']
    sampop = iinit_hori_p1.shape[0]
    well_num_hori = inj_num_hori + pro_num_hori
    hori_X_xi, hori_X_yi = ho.horiw_split1(iinit_hori_p1, well_num_hori, swf_num_hori)
    _, hori_X_L = ho.horiw_split2(iinit_hori_p2, well_num_hori, sunwf_num_hori)
    well_num_ver = inj_num_ver + pro_num_ver
    n_cols = pp_ver.shape[1]
    ver_X_xi = pp_ver[:, 0:n_cols:2].reshape((sampop, well_num_ver))
    ver_X_yi = pp_ver[:, 1:n_cols:2].reshape((sampop, well_num_ver))
    #
    obj_mode = 'NPV_hori'
    nsres = np.zeros([sampop, 1])
    rich_record = []
    for i in range(sampop):
        samp_list = []
        if getgrid_mode == 'center-4':
            hori_samplex = np.empty((1, 0))
            hori_sampley = np.empty((1, 0))
            for j in range(len(Hwtraj[i])):
                hori_samplex = np.hstack((hori_samplex, np.array([[Hwtraj[i][j][0][0]]])))
                hori_sampley = np.hstack((hori_sampley, np.array([[Hwtraj[i][j][0][1]]])))
        hori_sampleL = hori_X_L[i, :].reshape(1, well_num_hori)
        if well_num_hori > 0:
            hori_samtraj = Hwtraj[i]
        else:
            hori_samtraj = []
        ver_samplex = ver_X_xi[i, :].reshape(1, well_num_ver)
        ver_sampley = ver_X_yi[i, :].reshape(1, well_num_ver)

        FOPT, FWPT, FWIT, samp_res = rich_simulation(simupath, hori_samplex, hori_sampley, hori_samtraj, ver_samplex, ver_sampley, inj_num_hori, pro_num_hori, inj_num_ver, pro_num_ver, casename, mode, hori_sampleL)
        samp_list = samp_res
        samp_list.append(FOPT)
        samp_list.append(FWPT)
        samp_list.append(FWIT)
        if obj_mode == 'NPV_hori':
            Po = 80
            Cwp = 4.8
            Cwi = 4.8
            Chori = 7500
            total_L = np.sum(hori_sampleL)
            NPV = Po * FOPT - FWPT * Cwp - FWIT * Cwi - total_L / 100 * Chori
            nsres[i] = NPV
            samp_list.append(NPV)
        rich_record.append(samp_list)
    return nsres, rich_record