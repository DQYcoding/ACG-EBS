import numpy as np
import copy

def horiw_split1(matrix, well_num_hori, swf_num_hori):
    m_row = matrix.shape[0]
    xi = np.empty([m_row, 0])
    yi = np.empty([m_row, 0])
    for i in range(well_num_hori):
        x = copy.deepcopy((matrix[:, i * swf_num_hori]).reshape(m_row, 1))
        xi = np.hstack((xi, x))
        y = copy.deepcopy((matrix[:, i * swf_num_hori + 1]).reshape(m_row, 1))
        yi = np.hstack((yi, y))
    return xi, yi


def horiw_split2(matrix, well_num_hori, sunwf_num_hori):
    m_row = matrix.shape[0]
    alpha = np.empty([m_row, 0])
    L = np.empty([m_row, 0])
    for i in range(well_num_hori):
        a = copy.deepcopy(matrix[:, i * sunwf_num_hori]).reshape(m_row, 1)
        alpha = np.hstack((alpha, a))
        l = copy.deepcopy(matrix[:, i * sunwf_num_hori + 1]).reshape(m_row, 1)
        L = np.hstack((L, l))
    return alpha, L
    
    
def horiw_stack1(fakepop_init, well_num_hori, X_xi, X_yi):
    iinit_hori_p1 = np.empty([fakepop_init, 0])
    for i in range(well_num_hori):
        x = copy.deepcopy((X_xi[:, i]).reshape(fakepop_init, 1))
        y = copy.deepcopy((X_yi[:, i]).reshape(fakepop_init, 1))
        singlecoor = np.concatenate((x, y), axis = 1)
        iinit_hori_p1 = np.hstack((iinit_hori_p1, singlecoor))
    return iinit_hori_p1


def horiw_stack2(fakepop_init, well_num_hori, X_alpha, X_L):
    iinit_hori_p2 = np.empty([fakepop_init, 0])
    for i in range(well_num_hori):
        alpha = copy.deepcopy((X_alpha[:, i]).reshape(fakepop_init, 1))
        L = copy.deepcopy((X_L[:, i]).reshape(fakepop_init, 1))
        singlefeat = np.concatenate((alpha, L), axis = 1)
        iinit_hori_p2 = np.hstack((iinit_hori_p2, singlefeat))
    return iinit_hori_p2