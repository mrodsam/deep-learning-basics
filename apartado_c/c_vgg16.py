#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:55:11 2020

@author: martarodriguezsampayo
"""

import os, cv2, csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image , ImageDraw

from keras.layers import Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model

from sklearn.model_selection import train_test_split

pathTrain = "REY_roi_manualselection1"
csv_file = "anotaciones1.csv"

load = False
a1 = True #True:a1, False:a4
if a1:
    model_file = 'ieeercnn_vgg16_0.78.h5'
else:
    model_file = 'ieeercnn_vgg16_0.86.h5'

#ACABÓ EN LA ÉPOCA 00062 CON 0.78189, [186.82695770263672, 186.82695770263672, 6.397659778594971, 0.5713468591372172]
def main():

    train_images, features = preprocess_data()
    
    X = np.array(train_images)
    Y = np.array(features)
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    
    vggmodel = VGG16(weights='imagenet', include_top=True)
    
    for layers in (vggmodel.layers)[:15]:
        layers.trainable = False
        
    x = vggmodel.layers[-2].output
    predictions = Dense(2, activation="relu")(x)
    model = Model(inputs=vggmodel.input, outputs=predictions)
    opt = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', 
                        optimizer = opt, metrics=['mean_squared_error', 'mean_absolute_error', coeff_determination])
    
    model.summary()    

    if load:
        model = load_model(model_file, custom_objects = {'coeff_determination':coeff_determination})
    else:
        idg1 = ImageDataGenerator()
        idg2 = ImageDataGenerator()
        traindata = idg1.flow(x=x_train, y=y_train)
        testdata = idg2.flow(x=x_test, y=y_test)
        
        checkpoint = ModelCheckpoint("ieeercnn_vgg16_{val_coeff_determination:.2f}.h5", 
                                     monitor='val_coeff_determination', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='max')
        
        stop = EarlyStopping(monitor='val_coeff_determination', patience=20, mode='max')
        
        reduce_lr = ReduceLROnPlateau(monitor='val_coeff_determination', factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
    
        history = model.fit_generator(generator=traindata,
                                         steps_per_epoch=10, epochs=100, validation_data=testdata,
                                         validation_steps=2, callbacks=[checkpoint,stop,reduce_lr],
                                         verbose=1)  
        
            
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('Model MSE')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(['Train MSE', 'Test MSE'], loc='upper left')
        plt.savefig('c_vgg16_mse_line.png')
        plt.show()
       
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train MAE', 'Test MAE'], loc='upper left')
        plt.savefig('c_vgg16_mae_line.png')
        plt.show()
        
        plt.plot(history.history['coeff_determination'])
        plt.plot(history.history['val_coeff_determination'])
        plt.title('Model R_squared')
        plt.ylabel('R_Squared')
        plt.xlabel('Epoch')
        plt.legend(['Train RSq', 'Test RSq'], loc='upper left')
        plt.savefig('c_vgg16_r2_line.png')
        plt.show()
    
    test(model, x_test, y_test)
    

def preprocess_data():
    train_images = []
    features = []
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
                    point = row[3].strip()
                    point = point.lstrip('(')
                    point = point.rstrip(')')
                    
                    x0 = int(point.split(",")[0])
                    y0 = int(point.split(",")[1])
                    
                    feature = [ None ] * 2
                    feature[0] = int(x0/image_width*224)
                    feature[1] = int(y0/image_height*224)
                
                    features.append( feature )
            else:            
                #a4: inicio y fin de la línea horizontal central
                if element == "a4":
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
                    
                    line_coords[0] = int(dict_coords['x0']/image_width*224)
                    line_coords[1] = int(dict_coords['y0']/image_height*224)
                    line_coords[2] = int(dict_coords['x'+str(int(num_points-1))]/image_width*224)
                    line_coords[3] = int(dict_coords['y'+str(int(num_points-1))]/image_height*224)
                    
                    features.append(line_coords)
            
                resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                train_images.append(resized)
            
    return train_images, features

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

if __name__ == '__main__':
    main()
    
