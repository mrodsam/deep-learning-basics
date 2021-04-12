#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:16:03 2020

@author: martarodriguezsampayo
"""
from PIL import Image , ImageDraw
import os
import glob
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import backend as K
import keras.losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split


input_dim = 228
csv_file = "traza_REY.csv"
pathTrain = "REY_scan_anonim/"
load = False
weights_file = "model-checkpoint-0.47.h5"

input_shape = ( input_dim , input_dim , 3 )
dropout_rate = 0.5
alpha = 0.2

PATIENCE = 20

def main():
    
   images = process_images()
   bboxes, classes_raw = process_annotations()
   
   boxes = np.array(bboxes)

   Y = boxes
   X = np.array(images)
   
   x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)
   
   model = create_model()
   model.summary()
   
   if load:
       model.load_weights(weights_file)
   else:   
       checkpoint = ModelCheckpoint("model-checkpoint-{val_iou_metric:.2f}.h5", monitor='val_iou_metric', 
                                    verbose=1, save_best_only=True,save_weights_only=True, mode="max")
       stop = EarlyStopping(monitor='val_iou_metric', patience=PATIENCE, mode="max")
       reduce_lr = ReduceLROnPlateau(monitor='val_iou_metric', factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
          
       history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                           epochs=100, callbacks=[checkpoint, reduce_lr, stop], batch_size=32, verbose=1)   
       
       plt.plot(history.history['iou_metric'])
       plt.plot(history.history['val_iou_metric'])
       plt.title('Model iou')
       plt.ylabel('IOU')
       plt.xlabel('Epoch')
       plt.legend(['Train', 'Test'], loc='upper left')
       plt.savefig('a_cnn_iou.png')
       plt.show()
        
       plt.plot(history.history['loss'])
       plt.plot(history.history['val_loss'])
       plt.title('Model loss')
       plt.ylabel('Loss')
       plt.xlabel('Epoch')
       plt.legend(['Train', 'Test'], loc='upper left')
       plt.savefig('a_cnn_mse.png')
       plt.show()
       
       
   test(model, x_test)
   evaluation(model, x_test, y_test)
    

def process_images():
    images = []
    for imagefile in glob.glob( pathTrain+'*' ):
        image = Image.open( imagefile ).resize( ( input_dim , input_dim ))
        images.append( np.asarray( image ) / 255.0 )
    
    return images

def process_annotations():
    bboxes = []
    classes_raw = []
    with open(csv_file, "r") as file:
            
        reader = csv.reader(file, delimiter=";")
        for index, row in enumerate(reader):
            
            filename = row[2].split(":")[1]
            path = os.path.join(pathTrain,filename)
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
            
            x0 = int(x0/image_width*input_dim)
            y0 = int(y0/image_height*input_dim)
            x1 = int(x1/image_width*input_dim)
            y1 = int(y1/image_height*input_dim)
            
            bounding_box = [ 0.0 ] * 4
            bounding_box[0] = x0
            bounding_box[1] = y0
            bounding_box[2] = x1
            bounding_box[3] = y1
            
            bboxes.append( bounding_box )
            
                            
            rot = row[4].split(":")[1]
            
            if rot == "0":
                image_class = 0
            if rot == "90":
                image_class = 1
            if rot == "-90":
                image_class = 2
            if rot == "180":
                image_class = 3
                
            classes_raw.append(image_class)
    
    return bboxes, classes_raw
                
def calculate_iou( target_boxes , pred_boxes ):
    xA = K.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = K.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = K.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = K.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = K.maximum( 0.0 , xB - xA ) * K.maximum( 0.0 , yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    return iou


def iou_metric( y_true , y_pred ):
    return calculate_iou( y_true , y_pred )

def create_model():
    pred_vector_length = 4
    
    model_layers = [
			keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Flatten(),

			keras.layers.Dense(1240),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(640),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(480),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(120),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(62),
			keras.layers.LeakyReLU(alpha=alpha),

			keras.layers.Dense( pred_vector_length ),
			keras.layers.LeakyReLU(alpha=alpha),
		]
    
    model = keras.Sequential( model_layers )
    model.compile(
    	optimizer=keras.optimizers.Adam( lr=0.0001 ),
    	loss='mse',
        metrics=[ iou_metric ]
    )

    return model             

def test(model, x_test):

    boxes = model.predict( x_test )
    
    for i in range( boxes.shape[0] ):
        b = boxes[ i , 0 : 4 ]
        img = x_test[i] * 255
        source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
        draw = ImageDraw.Draw( source_img )
        draw.rectangle( b , outline="red" )
        source_img.save( 'inference_images/image_{}.png'.format( i + 1 ) , 'png' )

        
def calculate_avg_iou( target_boxes , pred_boxes ):
    xA = np.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = np.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = np.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = np.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = np.maximum(0.0, xB - xA ) * np.maximum(0.0, yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    
    return iou


def evaluation(model, x_test, y_test):
    target_boxes = y_test
    pred = model.predict( x_test )
    pred_boxes = np.array(pred)
    
    iou_scores = calculate_avg_iou( target_boxes , pred_boxes )
    print( 'Mean IOU score {}'.format( iou_scores.mean() ) )
    
        
if __name__ == "__main__":
    main()