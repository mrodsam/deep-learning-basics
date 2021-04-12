#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:16:03 2020

@author: martarodriguezsampayo
"""
from PIL import Image , ImageDraw
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.losses
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


input_dim = 228
csv_file = "anotaciones1.csv"
pathTrain = "REY_roi_manualselection1"

input_shape = ( input_dim , input_dim , 3 )
dropout_rate = 0.5
alpha = 0.2

load = False
a1 = True #True:a1, False:a4
if a1:
    weights_file = 'modelR2-0.36.h5'
else:
    weights_file = 'modelR2line-0.52.h5'

def main():
   features, names = process_annotations()
   images = process_images(names)   
   
   Y = np.array(features)
   X = np.array(images)
   
   x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)
   
   model = create_model()
   model.summary()
   
   if load:
       model.load_weights(weights_file)
   else:
       checkpoint = ModelCheckpoint("modelR2-{val_coeff_determination:.2f}.h5", monitor='val_coeff_determination', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode="max")
       stop = EarlyStopping(monitor='val_coeff_determination', patience=20, mode="max")
       reduce_lr = ReduceLROnPlateau(monitor='val_coeff_determination', factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
          
       history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                           epochs=100, callbacks=[checkpoint, reduce_lr, stop], batch_size=16, verbose=1)   
       
       
       plt.plot(history.history['mean_squared_error'])
       plt.plot(history.history['val_mean_squared_error'])
       plt.title('Model MSE')
       plt.ylabel('MSE')
       plt.xlabel('Epoch')
       plt.legend(['Train MSE', 'Test MSE'], loc='upper left')
       plt.savefig('c_cnn_mse_line.png')
       plt.show()
       
       plt.plot(history.history['mean_absolute_error'])
       plt.plot(history.history['val_mean_absolute_error'])
       plt.title('Model MAE')
       plt.ylabel('MAE')
       plt.xlabel('Epoch')
       plt.legend(['Train MAE', 'Test MAE'], loc='upper left')
       plt.savefig('c_cnn_mae_line.png')
       plt.show()
       
       plt.plot(history.history['coeff_determination'])
       plt.plot(history.history['val_coeff_determination'])
       plt.title('Model R_squared')
       plt.ylabel('R_Squared')
       plt.xlabel('Epoch')
       plt.legend(['Train RSq', 'Test RSq'], loc='upper left')
       plt.savefig('c_cnn_r2_line.png')
       plt.show()
   
       
   test(model, x_test, y_test)
   
def process_images(names):
    images = []
    for filename in names:
        imagefile = os.path.join(pathTrain,filename)
        image = cv2.imread(imagefile)
        image = cv2.resize(image, (input_dim, input_dim) )
        images.append( np.asarray( image ) / 255.0 )            
    
    return images

def process_annotations():
    features = []
    names = []
    with open(csv_file, "r") as file:
            
        reader = csv.reader(file, delimiter=";")
        for index, row in enumerate(reader):
            
            filename = row[1].strip()
            
            path = os.path.join(pathTrain,filename)
            img = cv2.imread(path)
            if img is None:
                continue
            
            image_height, image_width, _ = img.shape
        
            element = row[2].strip()
            if a1:
                #a1: punto en el centro de la cruz de la izquierda
                if element == "a1":
                    names.append(filename)
                    point = row[3].strip()
                    point = point.lstrip('(')
                    point = point.rstrip(')')
                    
                    x0 = int(point.split(",")[0])
                    y0 = int(point.split(",")[1])
                    
                    feature = [ None ] * 2
                    feature[0] = int(x0/image_width*input_dim)
                    feature[1] = int(y0/image_height*input_dim)
                
                    features.append( feature )
            
            else:
                #a4: inicio y fin de la línea horizontal central
                if element == "a4":
                    names.append(filename)
                    coords = row[3].strip()
                    coordsSplit = coords.split(",")
                    
                    pos = 0
                    dict_coords = {}
                    line_coords = [None] * 4
                    for index,coord in enumerate(coordsSplit): 
                        coord = coord.lstrip("(")
                        coord = coord.rstrip(")")
                        if index%2 == 0:
                            dict_coords['x'+str(pos)] = int(coord)
                        else:
                            dict_coords['y'+str(pos)] = int(coord)
                            pos+=1
                    keys = dict_coords.keys()
                    num_points = len(keys)/2
                    
                    line_coords[0] = int(dict_coords['x0']/image_width*input_dim)
                    line_coords[1] = int(dict_coords['y0']/image_height*input_dim)
                    line_coords[2] = int(dict_coords['x'+str(int(num_points-1))]/image_width*input_dim)
                    line_coords[3] = int(dict_coords['y'+str(int(num_points-1))]/image_height*input_dim)
                    
                    features.append(line_coords)
    
    return features, names
                

def create_model():
    if a1:
        pred_vector_length = 2
    else:
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
    	loss="mean_squared_error",
        metrics=[ 'mean_squared_error', 'mean_absolute_error', coeff_determination]
    )

    return model

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )             

        
def test(model, x_test, y_test):
    if a1:
        points = model.predict( x_test )
        
        for i in range( points.shape[0] ):
            b = points[ i , 0 : 2 ]
            img = x_test[i] * 255
            source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
            draw = ImageDraw.Draw( source_img )
            draw.point( b , fill='red')
            source_img.save( 'point_images/image_{}.png'.format( i + 1 ) , 'png' )
    else:
        lines = model.predict( x_test )
        
        for i in range( points.shape[0] ):
            l = lines[ i , 0 : 4 ]
            img = x_test[i] * 255
            source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
            draw = ImageDraw.Draw( source_img )
            draw.line( l , fill='red', width=0)
            source_img.save( 'line_images/image_{}.png'.format( i + 1 ) , 'png' )
        
    evaluation(model, x_test, y_test)
    
def evaluation(model, x_test, y_test):
    evaluation = model.evaluate(x_test, y_test)
    print(evaluation)
        
if __name__ == "__main__":
    main()