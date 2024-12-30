"""
Created on Sat Apr  9 04:08:02 2022
@author: Admin_with ODD Team

Edited by our team : Sat Oct 12 2024

references: https://github.com/vinvino02/GLPDepth
"""

import numpy as np
import pandas as pd
import torch

from utils.processing_cy import process_bbox_depth_cy, handle_overlaps_cy

class PROCESSING:
    def process_detections(self, scores, boxes, depth_map, detr):
        self.data = pd.DataFrame(columns=['xmin','ymin','xmax','ymax','width', 'height',
                                        'depth_mean_trim','depth_mean','depth_median', 
                                        'class', 'rgb'])
        
        boxes_array = np.array([[int(box[1]), int(box[0]), int(box[3]), int(box[2])] 
                               for box in boxes.tolist()], dtype=np.int32)
        
        # Use Cython-optimized overlap handling
        valid_indices = handle_overlaps_cy(depth_map, boxes_array)
        
        for idx in valid_indices:
            p = scores[idx]
            box = boxes[idx]
            xmin, ymin, xmax, ymax = map(int, box)
            
            detected_class = p.argmax()
            class_label = detr.CLASSES[detected_class]
            
            # Map classes
            if class_label == 'motorcycle':
                class_label = 'bicycle'
            elif class_label == 'bus':
                class_label = 'train'
            elif class_label not in ['person', 'truck', 'car', 'bicycle', 'train']:
                class_label = 'Misc'
            
            if class_label in ['Misc', 'person', 'truck', 'car', 'bicycle', 'train']:
                # Use Cython-optimized depth calculations
                depth_mean, depth_median, (depth_trim_low, depth_trim_high) = \
                    process_bbox_depth_cy(depth_map, ymin, ymax, xmin, xmax)
                
                class_index = ['Misc', 'person', 'truck', 'car', 'bicycle', 'train'].index(class_label)
                r, g, b = detr.COLORS[class_index]
                rgb = (r * 255, g * 255, b * 255)
                
                new_row = pd.DataFrame([[xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin,
                                       (depth_trim_low + depth_trim_high) / 2,
                                       depth_mean, depth_median, class_label, rgb]], 
                                     columns=self.data.columns)
                self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        return self.data