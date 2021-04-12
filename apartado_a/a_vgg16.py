#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:55:11 2020

@author: martarodriguezsampayo
"""

import os, cv2, csv
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

from keras import backend as K

from sklearn.model_selection import train_test_split

pathTrain = "REY_scan_anonim"
csv_file = "traza_REY.csv"

load = False
model_file = "ieeercnn_vgg16_0.86.h5"

def main():

    train_images, bboxes = preprocess_data()
    
    X = np.array(train_images)
    Y = np.array(bboxes)
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    
    vggmodel = VGG16(weights='imagenet', include_top=True)
    
    for layers in (vggmodel.layers)[:15]:
        layers.trainable = False
        
    x = vggmodel.layers[-2].output
    predictions = Dense(4, activation="relu")(x)
    model = Model(inputs=vggmodel.input, outputs=predictions)
    opt = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', 
                        optimizer = opt, metrics=[iou])
    
    model.summary()    
    
    if load:
        model = load_model(model_file, custom_objects = {'iou':iou})
    else:
        idg1 = ImageDataGenerator()
        idg2 = ImageDataGenerator()
        traindata = idg1.flow(x=x_train, y=y_train)
        testdata = idg2.flow(x=x_test, y=y_test)
        
        checkpoint = ModelCheckpoint("ieeercnn_vgg16_{val_iou:.2f}.h5", 
                                     monitor='val_iou', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='max')
        
        stop = EarlyStopping(monitor='val_iou', patience=20, mode='max')
        
        reduce_lr = ReduceLROnPlateau(monitor='val_iou', factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
    
        history = model.fit_generator(generator=traindata,
                                         steps_per_epoch=10, epochs=100, validation_data=testdata,
                                         validation_steps=2, callbacks=[checkpoint,stop,reduce_lr],
                                         verbose=1)  
        
            
        plt.plot(history.history['iou'])
        plt.plot(history.history['val_iou'])
        plt.title('Model iou')
        plt.ylabel('IOU')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('vgg16-bboxes-iou.png')
        plt.show()
    
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model MSE')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('vgg16-bboxes-mse.png')
        plt.show()
    
    test(x_test, y_test, model)
    evaluation(model, x_test, y_test)
    

def preprocess_data():
    train_images = []
    bboxes = []
    with open(csv_file, "r") as file:
    
        reader = csv.reader(file, delimiter=";")
        for index, row in enumerate(reader):
            filename = row[2].split(":")[1]
            
            img = cv2.imread(os.path.join(pathTrain,filename))
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
            
            x0 = int(x0/image_width*224)
            y0 = int(y0/image_height*224)
            x1 = int(x1/image_width*224)
            y1 = int(y1/image_height*224)
            
            bounding_box = [ 0.0 ] * 4
            bounding_box[0] = x0
            bounding_box[1] = y0
            bounding_box[2] = x1
            bounding_box[3] = y1
            
            bboxes.append( bounding_box )
            
            
            resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
            train_images.append(resized)
            
    return train_images, bboxes

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


def iou( y_true , y_pred ):
    return calculate_iou( y_true , y_pred )

def test(testX, testY, model):
    
    i = 0     
    for image in testX:
        i+=1
        box = model.predict(image, verbose=0)
        x0 = int(box[0][0])
        y0 = int(box[0][1])
        x1 = int(box[0][2])
        y1 = int(box[0][3])
        image = np.squeeze(image, axis=0)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.imwrite('inference_images/image'+str(i)+'.png', image)

def evaluation(model, x_test, y_test):
    evaluation = model.evaluate(x_test, y_test)
    print(evaluation)

if __name__ == '__main__':
    main()
    
