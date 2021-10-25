# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 12:31:25 2021

@author: tlizr
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from numpy import ones, shape, zeros, int8, concatenate, reshape, argmax
from tensorflow.keras.models import load_model
from cv2 import CascadeClassifier, cvtColor, equalizeHist, COLOR_RGB2GRAY, resize, INTER_LINEAR, imread
ROOT = 'C:/Users/tlizr/.spyder-py3'
PATH_VIOLA_JONES = '/Viola-Jones/haarcascade_frontalface_default.xml'
PATH_CNN = '/CNN/efficientnet_learningrate=0.001_3classes_epoch=20_time12.2_filt_balan.h5'
LOFFSET = 0
YCOFFSET = 0
XCOFFSET = 0
RESOLUTION = (1280, 720)
INPUT_SHAPE = 224

x_positivo = []
for root, _, filenames in os.walk('D:/tlizr/Pictures/Camera Roll/positivo'):
    for filename in filenames:
        x_positivo.append(os.path.join(root, filename))
x_negativo = []
for root, _, filenames in os.walk('D:/tlizr/Pictures/Camera Roll/negativo'):
    for filename in filenames:
        x_negativo.append(os.path.join(root, filename))
x_neutro = []
for root, _, filenames in os.walk('D:/tlizr/Pictures/Camera Roll/neutro'):
    for filename in filenames:
        x_neutro.append(os.path.join(root, filename))

y_positivo = zeros(shape(x_positivo), int8, 'C')
y_negativo = ones(shape(x_negativo), int8, 'C')
y_neutro = zeros(shape(x_neutro), int8, 'C')
y_neutro[:] = 2

x_test = concatenate((x_positivo, x_negativo, x_neutro), axis=0)
y_test = concatenate((y_positivo, y_negativo, y_neutro), axis=0)

print("cargando modelo")
model = load_model(ROOT + PATH_CNN)
config = model.get_config()
INPUT_SHAPE = config["layers"][0]["config"]["batch_input_shape"][1]
face_classifier = CascadeClassifier()
face_classifier.load(ROOT + PATH_VIOLA_JONES)
index = []
clasificacion = []
for i in range(shape(x_test)[0]):
    print("cargando imagen")
    x_img = imread(x_test[i])
    gray = cvtColor(x_img, COLOR_RGB2GRAY)
    gray = equalizeHist(gray)
    faces = face_classifier.detectMultiScale(x_img, scaleFactor = 2, minNeighbors = 3)
    for roi in faces:
        crop = resize(x_img[roi[1]:(roi[1]+roi[3]),roi[0]:(roi[0]+roi[2])], (INPUT_SHAPE,INPUT_SHAPE), interpolation= INTER_LINEAR)
        crop = crop / 255
        frame = reshape(crop, (1,INPUT_SHAPE,INPUT_SHAPE,3))
        prediction = model.predict(
                                    frame,
                                    batch_size=1,
                                    verbose=0,
                                    steps=None,
                                    callbacks=None,
                                    workers=1,
                                    use_multiprocessing=False)
        pred = argmax(prediction)
        clasificacion.append(pred)
cm = confusion_matrix(y_test, clasificacion, normalize = None)
disp = ConfusionMatrixDisplay(cm, display_labels=['Positivo', 'Negativo', 'Neutro'])
disp.plot()