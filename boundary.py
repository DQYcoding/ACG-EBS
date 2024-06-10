import numpy as np

def well_bounds(wi_low, wi_up, wj_low, wj_up):
    wellfeat_bd = np.zeros((2,2))
    wellfeat_bd[0][0] = wi_low
    wellfeat_bd[1][0] = wi_up
    wellfeat_bd[0][1] = wj_low
    wellfeat_bd[1][1] = wj_up
    return wellfeat_bd

def unwell_bounds(*args):
    tuple_size = np.size(args)
    unwfeat_num = int(tuple_size / 2)
    unwellfeat_bd = np.zeros((2, unwfeat_num))    
    for i in range(unwfeat_num):
        unwellfeat_bd[0][i] = args[i*2]
        unwellfeat_bd[1][i] = args[i*2 + 1]
    return unwellfeat_bd

