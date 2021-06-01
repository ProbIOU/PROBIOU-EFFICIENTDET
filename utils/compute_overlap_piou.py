# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

from numba import jit
import numpy as np

EPS = 1e-6

@jit(nopython=True)
def bhatacharyya_dist(x1,y1,a1,b1, x2,y2,a2,b2):
    '''
    Db = 1/4*((x1-x2)²/(a1+a2) + (y1-y2)²/(b1+b2))-ln2 \
    1/2*ln((a1+a2)*(b1+b2)) - 1/4*ln(a1*a2*b1*b2)
    '''
    return 1/4.*(np.power(x1-x2, 2.)/(a1+a2+EPS) + np.power(y1-y2, 2.)/(b1+b2+EPS)) - np.log(2.) + \
           1/2.*np.log((a1+a2)*(b1+b2)+EPS) - 1/4.*np.log(a1*a2*b1*b2+EPS)

@jit(nopython=True)
def compute_piou(Db):
    '''
    Dh = sqrt(1 - exp(-Db))
    '''
    return 1. - np.sqrt(1. - np.exp(-Db))

@jit(nopython=True)
def get_piou_values(array):
    x = (array[2] + array[0])/2.
    y = (array[3] + array[1])/2.
    a = np.power(array[2] - array[0], 2.)/12.
    b = np.power(array[3] - array[1], 2.)/12.
    return x,y,a,b

@jit(nopython=True)
def compute_overlap(boxes, query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    
    for k in range(K):
        
        target_vals = get_piou_values(query_boxes[k])
        
        for n in range(N):
            
            overlaps[n, k] = compute_piou(bhatacharyya_dist(
                *target_vals,
                *get_piou_values(boxes[n])
            ))
            
    return overlaps
