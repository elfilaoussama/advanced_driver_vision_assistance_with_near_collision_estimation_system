# File: processing.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
import pandas as pd
cimport pandas as pd
from libc.math cimport isnan
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Define C types for better performance
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t ITYPE_t

def process_bbox_depth_cy(np.ndarray[DTYPE_t, ndim=2] depth_map,
                         int y_min, int y_max, int x_min, int x_max):
    """
    Optimized bbox depth calculations using Cython
    """
    cdef:
        int i, j, count = 0
        double sum_val = 0.0
        double mean_val = 0.0
        np.ndarray[DTYPE_t, ndim=1] flat_vals
        int flat_size = 0
        
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if not isnan(depth_map[i, j]):
                sum_val += depth_map[i, j]
                count += 1
    
    if count > 0:
        mean_val = sum_val / count
        
    # Create array for trimmed mean calculation
    flat_vals = np.zeros(count, dtype=np.float32)
    flat_size = 0
    
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if not isnan(depth_map[i, j]):
                flat_vals[flat_size] = depth_map[i, j]
                flat_size += 1
                
    return mean_val, np.median(flat_vals), np.percentile(flat_vals, [20, 80])

def handle_overlaps_cy(np.ndarray[DTYPE_t, ndim=2] depth_map,
                      np.ndarray[ITYPE_t, ndim=2] boxes):
    """
    Optimized overlap handling using Cython
    """
    cdef:
        int n_boxes = boxes.shape[0]
        int i, j
        int y_min1, y_max1, x_min1, x_max1
        int y_min2, y_max2, x_min2, x_max2
        double area1, area2, area_intersection
        bint* to_remove = <bint*>PyMem_Malloc(n_boxes * sizeof(bint))
        
    if not to_remove:
        raise MemoryError()
        
    try:
        for i in range(n_boxes):
            to_remove[i] = False
            
        for i in range(n_boxes):
            if to_remove[i]:
                continue
                
            y_min1, x_min1, y_max1, x_max1 = boxes[i]
            
            for j in range(i + 1, n_boxes):
                if to_remove[j]:
                    continue
                    
                y_min2, x_min2, y_max2, x_max2 = boxes[j]
                
                # Calculate intersection
                y_min_int = max(y_min1, y_min2)
                y_max_int = min(y_max1, y_max2)
                x_min_int = max(x_min1, x_min2)
                x_max_int = min(x_max1, x_max2)
                
                if y_min_int < y_max_int and x_min_int < x_max_int:
                    area1 = (y_max1 - y_min1) * (x_max1 - x_min1)
                    area2 = (y_max2 - y_min2) * (x_max2 - x_min2)
                    area_intersection = (y_max_int - y_min_int) * (x_max_int - x_min_int)
                    
                    if area_intersection / min(area1, area2) >= 0.70:
                        if area1 < area2:
                            to_remove[i] = True
                            break
                        else:
                            to_remove[j] = True
                            
        return np.array([i for i in range(n_boxes) if not to_remove[i]], dtype=np.int32)
                            
    finally:
        PyMem_Free(to_remove)
