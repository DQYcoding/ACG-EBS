import numpy as np
from math import *


def horiw_getgrid(num_pop, well_num_hori, X_xi, X_yi, X_alpha, X_L, wi_low_hori, wi_up_hori, wj_low_hori, wj_up_hori, ecD, getgrid_mode):
    if getgrid_mode == 'center-4':
        Hwtraj = []
        for i in range(num_pop):
            straj = []
            for j in range(well_num_hori):
                fixed_mid_xi = X_xi[i, j]
                fixed_mid_yi = X_yi[i, j]
                hstart = 1
                alpha_repair = 0
                k = 0
                while hstart == 1:
                    mid_xi = X_xi[i, j]
                    mid_yi = X_yi[i, j]
                    mid_alpha = X_alpha[i, j]
                    mid_L = X_L[i, j]
                    regen, hwtraj, mid_alpha = horiw_traj_center4(mid_xi, mid_yi, mid_alpha, mid_L, wi_low_hori, wi_up_hori, wj_low_hori, wj_up_hori, ecD)
                    X_alpha[i, j] = mid_alpha
                    if regen == 1:
                        #
                        if alpha_repair < 4:
                            X_alpha[i, j] = X_alpha[i, j] + 0.25*pi
                            alpha_repair = alpha_repair + 1
                            if X_alpha[i, j] >= pi:
                                X_alpha[i, j] = X_alpha[i, j] - pi
                            continue
                        #
                        elif alpha_repair >= 4:
                            center = [(wi_low_hori+wi_up_hori)/2, (wj_low_hori+wj_up_hori)/2]
                            angle = calculate_angle(center, [mid_xi, mid_yi])
                            #
                            if angle >= 0 and angle < 0.5*pi:
                                viL = int(mid_L / ecD)
                                xy_repairlist = generate_coordinates_1([fixed_mid_xi, fixed_mid_yi], viL)
                                X_xi[i, j] = xy_repairlist[k][0]
                                X_yi[i, j] = xy_repairlist[k][1]
                                k = k + 1
                                continue
                            elif angle >= 0.5*pi and angle < pi:
                                viL = int(mid_L / ecD)
                                xy_repairlist = generate_coordinates_2([fixed_mid_xi, fixed_mid_yi], viL)
                                X_xi[i, j] = xy_repairlist[k][0]
                                X_yi[i, j] = xy_repairlist[k][1]
                                k = k + 1
                                continue
                            elif angle >= pi and angle < 1.5*pi:
                                viL = int(mid_L / ecD)
                                xy_repairlist = generate_coordinates_3([fixed_mid_xi, fixed_mid_yi], viL)
                                X_xi[i, j] = xy_repairlist[k][0]
                                X_yi[i, j] = xy_repairlist[k][1]
                                k = k + 1
                                continue
                            elif angle >= 1.5*pi and angle <= 2*pi:
                                viL = int(mid_L / ecD)
                                xy_repairlist = generate_coordinates_4([fixed_mid_xi, fixed_mid_yi], viL)
                                X_xi[i, j] = xy_repairlist[k][0]
                                X_yi[i, j] = xy_repairlist[k][1]
                                k = k + 1
                                continue
                    if regen == 0:
                        hstart = 0
                    straj.append(hwtraj)
            Hwtraj.append(straj)
    return Hwtraj, X_alpha, X_xi, X_yi

     
def horiver_delete(dele_k, Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all):
    if Hwtraj != []:
        for i in reversed(dele_k):
            del Hwtraj[i]
    iinit_hori_p1 = np.delete(iinit_hori_p1, dele_k, axis=0)
    iinit_hori_p2 = np.delete(iinit_hori_p2, dele_k, axis=0)
    pp_hori = np.delete(pp_hori, dele_k, axis=0)
    pp_ver = np.delete(pp_ver, dele_k, axis=0)
    pp_all = np.delete(pp_all, dele_k, axis=0)
    return Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all