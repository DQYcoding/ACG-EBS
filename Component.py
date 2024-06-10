import time
import numpy as np
import lhs
import feature_operate as fo
import horizontal_judge as hj
import horizontal_constraint as hc
import random
import generate as gr
import eucliDist as euc
from RBF_newrbe import RBF_newrbe
import horizontal_operate as ho


def DE_MC(Mu, CR, pop, allbd, Num, ecD):
    Nm = np.size(pop, 0)
    D = np.size(pop, 1)
    V = np.zeros((Nm, D))
    U = np.zeros((Nm, D))
    for m in range(Nm):
        #
        r1 = random.randint(0, Nm - 1)
        while r1 == m:
            r1 = random.randint(0, Nm - 1)
        r2 = random.randint(0, Nm - 1)
        while r2 == m or r2 == r1:
            r2 = random.randint(0, Nm - 1)
        r3 = random.randint(0, Nm - 1)
        while r3 == m or r3 == r1 or r3 == r2:
            r3 = random.randint(0, Nm - 1)
        V[m, :] = pop[r1, :] + Mu * (pop[r2, :] - pop[r3, :])

    r = random.randint(0, D - 1)
    for n in range(D):
        cr = random.uniform(0, 1)
        if cr < CR or n == r:
            U[:, n] = V[:, n]
        else:
            U[:, n] = pop[:, n]

    minVar = allbd[0, :].reshape(1, allbd.shape[1])
    maxVar = allbd[1, :].reshape(1, allbd.shape[1])
    minMatrix = np.repeat(minVar, Nm, 0)
    maxMatrix = np.repeat(maxVar, Nm, 0)

    maxnum = np.argwhere(U > maxMatrix)
    for j in maxnum:
        U[j[0], j[1]] = maxMatrix[j[0], j[1]]
    minnum = np.argwhere(U < minMatrix)
    for j in minnum:
        U[j[0], j[1]] = minMatrix[j[0], j[1]]
    trial_hori_p1, trial_hori_p2, trial_hori, trial_ver, trial_all = hc.doroughtrial(U, Num, ecD)
    U = trial_all
    U_hori = trial_hori
    U_ver = trial_ver
    return U, U_hori, U_ver