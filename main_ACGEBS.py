import copy
import math
import os
import pickle
import numpy as np
import Component
import eucliDist as euc
import feature_operate
import fitness_compute
import horizontal_constraint
import horizontal_judge as hj
import horizontal_operate as ho
import lhs as lhs
import rich_numerical_simulation
import shutil
import generate
import random

cwd = os.getcwd()
initial_path = os.path.join(cwd, 'Iterm')
simulation_path = os.path.join(cwd, 'RunIterm')
result_path = os.path.join(cwd, 'Result')
casename = 'BASE'
mode = 'ecl_jq'
#
getgrid_mode = 'center-4'
if getgrid_mode == 'center-4':
    alpha_low = 0
    alpha_up = math.pi
#
wi_low_hori = 1
wi_up_hori = 100
wj_low_hori = 1
wj_up_hori = 100
#
wi_low_ver = 1
wi_up_ver = 100
wj_low_ver = 1
wj_up_ver = 100
#
hL_low = 300
hL_up = 700
#
old_traj = []
old_range = 2
ver_range = old_range   
hori_range = 1
old_well = []
#
wi_up_union = max(wi_up_hori, wi_up_ver)
wi_low_union = min(wi_low_hori, wi_low_ver)
wj_up_union = max(wj_up_hori, wj_up_ver)
wj_low_union = min(wj_low_hori, wj_low_ver)
#
for i in range(len(old_well)):
    old_coor = old_well[i]
    #
    temp_traj = []
    for j in range(1, old_range+1):
        new_coor1 = [old_coor[0]+j, old_coor[1]+j]
        new_coor2 = [old_coor[0]-j, old_coor[1]-j]
        if wi_low_union <= new_coor1[0] <= wi_up_union and wj_low_union <= new_coor1[1] <= wj_up_union:
            temp_traj.append(new_coor1)
        if wi_low_union <= new_coor2[0] <= wi_up_union and wj_low_union <= new_coor2[1] <= wj_up_union:
            temp_traj.append(new_coor2)
    if temp_traj:
        temp_traj.append(old_coor)
        old_traj.append(np.array(temp_traj))
    #
    temp_traj = []
    for j in range(1, old_range+1):
        new_coor3 = [old_coor[0]+j, old_coor[1]-j]
        new_coor4 = [old_coor[0]-j, old_coor[1]+j]
        if wi_low_union <= new_coor3[0] <= wi_up_union and wj_low_union <= new_coor3[1] <= wj_up_union:
            temp_traj.append(new_coor3)
        if wi_low_union <= new_coor4[0] <= wi_up_union and wj_low_union <= new_coor4[1] <= wj_up_union:
            temp_traj.append(new_coor4)
    if temp_traj:
        temp_traj.append(old_coor)
        old_traj.append(np.array(temp_traj))
#
pop_num_init = 100
totalFEs = 500
pop_evol = 0
inj_num_hori = 0
pro_num_hori = 6
inj_num_ver = 4
pro_num_ver = 0
singlew_dim_hori = 4
well_num_hori = inj_num_hori + pro_num_hori
well_num_ver = inj_num_ver + pro_num_ver
n_hori = well_num_hori * singlew_dim_hori
n_ver = well_num_ver * 2
n_all = n_hori + n_ver
ecD = 100
#
Wij = {}
Wij['wi_low_hori'] = wi_low_hori
Wij['wi_up_hori'] = wi_up_hori
Wij['wj_low_hori'] = wj_low_hori
Wij['wj_up_hori'] = wj_up_hori
Wij['wi_low_ver'] = wi_low_ver
Wij['wi_up_ver'] = wi_up_ver
Wij['wj_low_ver'] = wj_low_ver
Wij['wj_up_ver'] = wj_up_ver
Wij['wi_low_union'] = wi_low_union
Wij['wi_up_union'] = wi_up_union
Wij['wj_low_union'] = wj_low_union
Wij['wj_up_union'] = wj_up_union
#
bd_wfeat_hori, bd_unwfeat_hori, bd_wfeat_ver = feature_operate.singw_feat(Wij, alpha_low, alpha_up, hL_low, hL_up)
#
allbd_wfeat_hori, allbd_unwfeat_hori, allbd_hori, allbd_wfeat_ver, allbd_ver, allbd = feature_operate.allw_feat(well_num_hori, well_num_ver, bd_wfeat_hori, bd_unwfeat_hori, bd_wfeat_ver)
n1_hori = allbd_wfeat_hori.shape[1]
n2_hori = allbd_unwfeat_hori.shape[1]
#
if well_num_ver > 0:
    swf_num_ver = 2
else:
    swf_num_ver = 0
#
if well_num_hori > 0:
    swf_num_hori = int(n1_hori / well_num_hori)
    sunwf_num_hori = int(n2_hori / well_num_hori)
else:
    swf_num_hori = 0
    sunwf_num_hori = 0

Num = {}
Num['well_num_hori'] = well_num_hori
Num['well_num_ver'] = well_num_ver
Num['inj_num_hori'] = inj_num_hori
Num['inj_num_ver'] = inj_num_ver
Num['pro_num_hori'] = pro_num_hori
Num['pro_num_ver'] = pro_num_ver
Num['swf_num_hori'] = swf_num_hori
Num['sunwf_num_hori'] = sunwf_num_hori
Num['n_all'] = n_all
Num['n_hori'] = n_hori
Num['n_ver'] = n_ver
Num['n1_hori'] = n1_hori
Num['n2_hori'] = n2_hori

if os.path.exists(simulation_path):
    shutil.rmtree(simulation_path)
shutil.copytree(initial_path, simulation_path)

#
fake_para = 1
init_hori_p1, init_hori_p2, init_hori, init_ver, p_all = lhs.horiver_lhs(pop_num_init*fake_para, n_all, n_hori, n_ver, n1_hori, n2_hori, well_num_hori, well_num_ver, sunwf_num_hori, allbd, ecD)
#
M_new_ver_traj = feature_operate.get_newver_traj(init_ver, ver_range, Wij)
dele_k, Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori = horizontal_constraint.constraint(init_hori, n1_hori, n2_hori, well_num_hori, well_num_ver, ecD, old_traj, M_new_ver_traj, Wij, hori_range, getgrid_mode)
pp_ver = init_ver
pp_all = p_all
Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all = hj.horiver_delete(dele_k, Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all)
cur_pop = np.size(pp_all, 0)
#
judge = 1
while judge == 1:
    if cur_pop >= 0.8*pop_num_init and cur_pop <= pop_num_init:
        judge = 0
    if cur_pop < 0.8*pop_num_init:
        fake_para = fake_para + 1
        init_hori_p1, init_hori_p2, init_hori, init_ver, p_all = lhs.horiver_lhs(int(pop_num_init*fake_para), n_all, n_hori, n_ver, n1_hori, n2_hori, well_num_hori, well_num_ver, sunwf_num_hori, allbd, ecD)
        #
        M_new_ver_traj = feature_operate.get_newver_traj(init_ver, ver_range, Wij)
        #
        dele_k, Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori = horizontal_constraint.constraint(init_hori, n1_hori, n2_hori, well_num_hori, well_num_ver, ecD, old_traj, M_new_ver_traj, Wij, hori_range, getgrid_mode)
        pp_ver = init_ver
        pp_all = p_all
        Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all = hj.horiver_delete(dele_k, Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all)
        #
        cur_pop = np.size(pp_all, 0)
    if cur_pop > pop_num_init:
        fake_para = fake_para - 0.2
        #
        init_hori_p1, init_hori_p2, init_hori, init_ver, p_all = lhs.horiver_lhs(int(pop_num_init*fake_para), n_all, n_hori, n_ver, n1_hori, n2_hori, well_num_hori, well_num_ver, sunwf_num_hori, allbd, ecD)
        #
        M_new_ver_traj = feature_operate.get_newver_traj(init_ver, ver_range, Wij)
        #
        dele_k, Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori = horizontal_constraint.constraint(init_hori, n1_hori, n2_hori, well_num_hori, well_num_ver, ecD, old_traj, M_new_ver_traj, Wij, hori_range, getgrid_mode)
        #
        pp_ver = init_ver
        pp_all = p_all
        Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all = hj.horiver_delete(dele_k, Hwtraj, iinit_hori_p1, iinit_hori_p2, pp_hori, pp_ver, pp_all)
        #
        cur_pop = np.size(pp_all, 0)
#
while cur_pop < pop_num_init:
    #
    hori_p, ver_p, all_p = generate.generate_randonewithcons(well_num_hori, well_num_ver, Wij, alpha_low, alpha_up, hL_low, hL_up, ecD, old_traj, ver_range, hori_range, getgrid_mode)
    #
    hori_p_p1 = hori_p[:, 0: n1_hori].copy()
    hori_p_p2 = hori_p[:, n1_hori: n1_hori + n2_hori].copy()
    hori_p_Xi, hori_p_Yi = ho.horiw_split1(hori_p_p1, well_num_hori, swf_num_hori)
    hori_p_alpha, hori_p_L = ho.horiw_split2(hori_p_p2, well_num_hori, sunwf_num_hori)
    p_Hwtraj, hori_p_alpha, hori_p_Xi, hori_p_Yi = hj.horiw_getgrid(1, well_num_hori, hori_p_Xi, hori_p_Yi, hori_p_alpha, hori_p_L, wi_low_hori, wi_up_hori, wj_low_hori, wj_up_hori, ecD, getgrid_mode)
    #
    hori_p_p1 = ho.horiw_stack1(np.size(hori_p, 0), well_num_hori, hori_p_Xi, hori_p_Yi)
    hori_p_p2 = ho.horiw_stack2(np.size(hori_p, 0), well_num_hori, hori_p_alpha, hori_p_L)
    hori_p = np.hstack((hori_p_p1, hori_p_p2))
    all_p = np.hstack((hori_p, ver_p))
    #
    pp_hori = np.vstack((pp_hori, hori_p))
    iinit_hori_p1 = np.vstack((iinit_hori_p1, hori_p_p1))
    iinit_hori_p2 = np.vstack((iinit_hori_p2, hori_p_p2))
    Hwtraj.append(p_Hwtraj[0])
    #
    pp_ver = np.vstack((pp_ver, ver_p))
    pp_all = np.vstack((pp_all, all_p))
    #
    cur_pop = cur_pop + 1

#
R_rich_record = []
ns_data, rich_record = rich_numerical_simulation.rich_i_simulation(simulation_path, iinit_hori_p1, iinit_hori_p2, pp_ver, Hwtraj, Num, casename, mode, getgrid_mode)
R_rich_record = rich_record
FEs = pop_num_init
fitvalue = fitness_compute.fitness(ns_data)
fit = np.concatenate((pp_all, fitvalue), axis=1)

#
besty = np.min(fitvalue)
bestindex = np.where(fitvalue == besty)[0]
bestx = pp_all[bestindex[0], :].reshape((1, n_all))
#
init_record_bestX = np.tile(bestx, (pop_num_init, 1))
init_record_bestY = np.tile(besty, (pop_num_init, 1))
bestrecord = np.hstack((init_record_bestX, init_record_bestY))
database = copy.deepcopy(fit)

#
if os.path.exists(init_pkl_path):
    os.remove(init_pkl_path)
#
with open(init_pkl_path, 'wb') as f:
    pickle.dump([ns_data, pp_hori, pp_ver, pp_all, fit, database, iinit_hori_p1, iinit_hori_p2, Hwtraj, bestrecord, R_rich_record], f)

num_fltrain = 60
local_Max_gen = 5
local_NP = 60
local_Mu = 0.5
local_CR = 0.5
local_par_gen_mode = 'cur_best'
local_cons_mode = 'newDE'
num_fgtrain = 100
gl_par = 100
global_Mu = 0.5
global_CR = 0.5
global_cons_mode = 'newDE'

#
localFEs = 0
globalFEs = 0

while FEs < totalFEs:
    #
    flRBFmodel, data_fltrain = Component.nsm_RBF(database, num_fltrain)
    #
    lsur_pop = copy.deepcopy(data_fltrain[:, :-1])
    XL = np.min(lsur_pop, axis=0).reshape((1, n_all))
    XU = np.max(lsur_pop, axis=0).reshape((1, n_all))
    #
    min_error = 0.0001
    best_lsamp = Component.nsm_DE(flRBFmodel, XL, XU, min_error, Num, Wij, ecD, old_traj, ver_range, hori_range, local_Max_gen, local_NP, local_Mu, local_CR, database, local_par_gen_mode, local_cons_mode, getgrid_mode)
    epsilon = 0.0001
    #
    if euc.eucli_pointoset(best_lsamp, database[:, :-1]) >= epsilon:
        bls_p1_hori = best_lsamp[:, 0 : n1_hori].copy()
        bls_p2_hori = best_lsamp[:, n1_hori : n1_hori + n2_hori].copy()
        bls_Xi_hori, bls_Yi_hori = ho.horiw_split1(bls_p1_hori, well_num_hori, swf_num_hori)
        bls_alpha_hori, bls_L_hori = ho.horiw_split2(bls_p2_hori, well_num_hori, sunwf_num_hori)
        bls_Hwtraj, bls_alpha_hori, bls_Xi_hori, bls_Yi_hori = hj.horiw_getgrid(1, well_num_hori, bls_Xi_hori, bls_Yi_hori, bls_alpha_hori, bls_L_hori, wi_low_hori, wi_up_hori, wj_low_hori, wj_up_hori, ecD, getgrid_mode)
        #
        bls_p1_hori = ho.horiw_stack1(np.size(bestx, 0), well_num_hori, bls_Xi_hori, bls_Yi_hori)
        bls_p2_hori = ho.horiw_stack2(np.size(best_lsamp, 0), well_num_hori, bls_alpha_hori, bls_L_hori)
        bls_hori = np.hstack((bls_p1_hori, bls_p2_hori))
        #
        bls_ver = best_lsamp[:, n_hori:(n_hori+n_ver)]
        #
        best_lsamp = np.hstack((bls_hori, bls_ver))
        #
        realevo_back, rich_record = rich_numerical_simulation.rich_s_simulation(simulation_path, bls_p1_hori, bls_p2_hori, bls_ver, bls_Hwtraj, Num, casename, mode, getgrid_mode)
        R_rich_record.append(rich_record[0])
        realevo_fitv = fitness_compute.fitness(realevo_back)
        newpoint = np.append(best_lsamp, np.array(realevo_fitv)).reshape(1, np.size(best_lsamp, 1) + 1)
        database = np.concatenate((database, newpoint), axis = 0)
        FEs = FEs + 1
        localFEs = localFEs + 1
        #
        if realevo_fitv < besty:
            besty = realevo_fitv
            bestx = best_lsamp
        bestrecord = np.append(bestrecord, np.hstack((bestx, besty.reshape((1, 1)))), axis=0)
    
    if FEs >= totalFEs:
        break
    fgRBFmodel = Component.csm_RBF(database, num_fgtrain)
    num_bestpop = gl_par
    epsilon = 0.0001
    best_gsamp = Component.csm_DE(fgRBFmodel, num_bestpop, database, epsilon, Wij, Num, alpha_low, alpha_up, hL_low, hL_up, ecD, old_traj, ver_range, hori_range, global_Mu, global_CR, global_cons_mode, getgrid_mode)
    bgs_p1_hori = best_gsamp[:, 0 : n1_hori].copy()
    bgs_p2_hori = best_gsamp[:, n1_hori : n1_hori + n2_hori].copy()
    bgs_Xi_hori, bgs_Yi_hori = ho.horiw_split1(bgs_p1_hori, well_num_hori, swf_num_hori)
    bgs_alpha_hori, bgs_L_hori = ho.horiw_split2(bgs_p2_hori, well_num_hori, sunwf_num_hori)
    bgs_Hwtraj, bgs_alpha_hori, bgs_Xi_hori, bgs_Yi_hori = hj.horiw_getgrid(1, well_num_hori, bgs_Xi_hori, bgs_Yi_hori, bgs_alpha_hori, bgs_L_hori, wi_low_hori, wi_up_hori, wj_low_hori, wj_up_hori, ecD, getgrid_mode)
    #
    bgs_p1_hori = ho.horiw_stack1(np.size(bestx, 0), well_num_hori, bgs_Xi_hori, bgs_Yi_hori)
    bgs_p2_hori = ho.horiw_stack2(np.size(best_gsamp, 0), well_num_hori, bgs_alpha_hori, bgs_L_hori)
    bgs_hori = np.hstack((bgs_p1_hori, bgs_p2_hori))
    #
    bgs_ver = best_gsamp[:, n_hori:(n_hori+n_ver)]
    #
    best_gsamp = np.hstack((bgs_hori, bgs_ver))
    #
    realevo_back, rich_record = rich_numerical_simulation.rich_s_simulation(simulation_path, bgs_p1_hori, bgs_p2_hori, bgs_ver, bgs_Hwtraj, Num, casename, mode, getgrid_mode)
    R_rich_record.append(rich_record[0])
    realevo_fitv = fitness_compute.fitness(realevo_back)
    newpoint = np.append(best_gsamp, np.array(realevo_fitv)).reshape(1, np.size(best_gsamp, 1) + 1)
    database = np.concatenate((database, newpoint), axis=0)
    FEs = FEs + 1
    globalFEs = globalFEs + 1
    if realevo_fitv < besty:
        besty = realevo_fitv
        bestx = best_gsamp
    bestrecord = np.append(bestrecord, np.hstack((bestx, besty.reshape((1, 1)))), axis=0)

parname = os.path.join(result_path, 'Record_Best_by_FEs.txt')
f = open(parname, 'w')
for i in range(np.size(bestrecord, 0)):
    f.write(str(i+1) + '  ' + str(-bestrecord[i, -1]) + '\n')
f.close()

bestmodel_path = os.path.join(result_path, 'best_model')
aa = os.path.exists(bestmodel_path)
if aa:
    shutil.rmtree(bestmodel_path, ignore_errors=True)
shutil.copytree(initial_path, bestmodel_path)

b_p1_hori = bestx[:, 0 : n1_hori].copy()
b_p2_hori = bestx[:, n1_hori : n1_hori + n2_hori].copy()
b_Xi_hori, b_Yi_hori = ho.horiw_split1(b_p1_hori, well_num_hori, swf_num_hori)
b_alpha_hori, b_L_hori = ho.horiw_split2(b_p2_hori, well_num_hori, sunwf_num_hori)
b_Hwtraj, b_alpha_hori, b_Xi_hori, b_Yi_hori = hj.horiw_getgrid(1, well_num_hori, b_Xi_hori, b_Yi_hori, b_alpha_hori, b_L_hori, wi_low_hori, wi_up_hori, wj_low_hori, wj_up_hori, ecD, getgrid_mode)
b_ver = bestx[:, n_hori:(n_hori+n_ver)]
b_p1_hori = ho.horiw_stack1(np.size(bestx, 0), well_num_hori, b_Xi_hori, b_Yi_hori)
b_p2_hori = ho.horiw_stack2(np.size(bestx, 0), well_num_hori, b_alpha_hori, b_L_hori)
bestobj, rich_record = rich_numerical_simulation.rich_s_simulation(bestmodel_path, b_p1_hori, b_p2_hori, b_ver, b_Hwtraj, Num, casename, mode, getgrid_mode)

f = open('save.pkl', 'wb')
pickle.dump([bestx, bestobj, database, bestrecord, R_rich_record], f)
f.close()
