#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:29:01 2020

@author: martarodriguezsampayo
"""

import cv2, csv, os
import numpy as np

csv_file = 'traza_REY.csv'
imagespath = 'REY_scan_anonim'

def main():
    
    images, bboxes = process_data()
    boxes_pred = []
    
    for index, image in enumerate(images):
        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # dilation
        kernel = np.ones((40, 50), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        
        # find contours
        # cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
        cv2MajorVersion = cv2.__version__.split(".")[0]
        # check for contours on thresh
        if int(cv2MajorVersion) >= 4:
            ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        bbox_area = []
        pos_bboxes = []
        for i, ctr in enumerate(sorted_ctrs):
            box = [0] *4
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            bbox_area.append(w*h)
            # Getting ROI
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            box[0] = x
            box[1] = y
            box[2] = x + w
            box[3] = y + h
            pos_bboxes.append(box)
            
        bbox_area = np.array(bbox_area)
        max_area = np.argmax(bbox_area, axis=0)
        final_box = pos_bboxes[max_area]
        boxes_pred.append(final_box)
        
        cv2.rectangle(image, (final_box[0], final_box[1]),(final_box[2], final_box[3]), (0,0,255), 2)
        cv2.imwrite('inference_images/image{}.png'.format(index), image)
        
    print(average_iou(boxes_pred, bboxes))

    
def process_data():
    bboxes = []
    images = []
    with open(csv_file, "r") as file:
            
        reader = csv.reader(file, delimiter=";")
        for index, row in enumerate(reader):
            
            filename = row[2].split(":")[1]
            path = os.path.join(imagespath,filename)
            img = cv2.imread(path)
            if img is None:
                continue
            image_height, image_width, _ = img.shape
        
            bndboxes = row[3].split(":")[1]
            bndboxes = bndboxes.lstrip('[')
            bndboxes = bndboxes.rstrip(']')
        
            x0, y0, h, w = int(bndboxes.split(",")[0]), int(bndboxes.split(",")[1]), int(bndboxes.split(",")[2]), int(
                bndboxes.split(",")[3])
            
            x1 = x0 + w
            y1 = y0  + h
            
            bounding_box = [0]*4
            bounding_box[0] = x0
            bounding_box[1] = y0
            bounding_box[2] = x1
            bounding_box[3] = y1
            
            bboxes.append( bounding_box )
            images.append(img)
    
    return images, bboxes

def average_iou(bboxes_pred, bboxes_true):
    iou = 0
    for i in range(len(bboxes_pred)):
        bbox_true = bboxes_true[i]
        bbox_pred = bboxes_pred[i] 
        iou += calculate_iou(bbox_pred, bbox_true)
    
    return iou/len(bboxes_pred)

def calculate_iou(bbox_pred, bbox_true):

        
    assert bbox_true[0] < bbox_true[2]
    assert bbox_true[1] < bbox_true[3]
    assert bbox_pred[0] < bbox_pred[2]
    assert bbox_pred[1] < bbox_pred[3]

    
    x_left = max(bbox_true[0], bbox_pred[0])
    y_top = max(bbox_true[1], bbox_pred[1])
    x_right = min(bbox_true[2], bbox_pred[2])
    y_bottom = min(bbox_true[3], bbox_pred[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbtrue_area = (bbox_true[2] - bbox_true[0]) * (bbox_true[3] - bbox_true[1])
    bbpred_area = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])

    iou = intersection_area / float(bbtrue_area + bbpred_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

if __name__ == "__main__":
    main()