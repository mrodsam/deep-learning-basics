#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:55:11 2020

@author: martarodriguezsampayo
"""

import os, cv2, keras, csv
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

pathTrain = "REY_roi_rot0"
csv_file = "traza_REY.csv"

model_file = 'ieeercnn_vgg16_1.00.h5'
load = False

# Clases: 0 = 0, 1 = 90, 2 = -90, 3 = 180

def main():
    
    train_images, train_labels = preprocess_data()
    
    X = np.array(train_images)
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(train_labels)
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    
    vggmodel = VGG16(weights='imagenet', include_top=True)
    
    for layers in (vggmodel.layers)[:15]:
        layers.trainable = False
        
    x = vggmodel.layers[-2].output
    predictions = Dense(4, activation="softmax")(x)
    model = Model(inputs=vggmodel.input, outputs=predictions)
    opt = Adam(lr=0.0001)
    model.compile(loss=keras.losses.categorical_crossentropy, 
                        optimizer = opt, metrics=["categorical_accuracy"])
    
    model.summary()   

    if load:
        model = load_model(model_file)
    
    else:
        idg1 = ImageDataGenerator()
        idg2 = ImageDataGenerator()
        traindata = idg1.flow(x=x_train, y=y_train)
        testdata = idg2.flow(x=x_test, y=y_test)
        
        checkpoint = ModelCheckpoint("ieeercnn_vgg16_{val_categorical_accuracy:.2f}.h5", 
                                     monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='max')
        
        stop = EarlyStopping(monitor='val_categorical_accuracy', patience=20, mode='max')
        
        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
    
        history = model.fit_generator(generator=traindata,
                                         steps_per_epoch=10, epochs=100, validation_data=testdata,
                                         validation_steps=2, callbacks=[checkpoint,stop,reduce_lr],
                                         verbose=1)  
        
            
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('b_cnn_accuracy.png')
        plt.show()
    
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model CCE')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('b_cnn_loss.png')
        plt.show()
    
    evaluation(x_test, y_test, model, encoder)
    

def preprocess_data():
    train_images = []
    train_labels = []
    with open(csv_file, "r") as file:
    
        reader = csv.reader(file, delimiter=";")
        for index, row in enumerate(reader):
            filename = row[1].split(":")[1]
            
            img = cv2.imread(os.path.join(pathTrain,filename))
            if img is None:
                continue
            rot = row[4].split(":")[1]
            
            if rot == "0":
                image_class = 0
            if rot == "90":
                image_class = 1
            if rot == "-90":
                image_class = 2
            if rot == "180":
                image_class = 3
            
            resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
            train_images.append(resized)
            train_labels.append(image_class)
            
    return train_images, train_labels

def evaluation(testX, testY, model, encoder):
    
    yhat_probs = model.predict(testX, verbose=0)
    yhat_classes = np.argmax(yhat_probs, axis=1)
        
    yhat_classes = np.array(yhat_classes)
    testY = encoder.inverse_transform(testY)
    testY = np.array(testY)
        
    print(classification_report(testY,yhat_classes))


if __name__ == '__main__':
    main()
    
