# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 18:23:40 2021

@author: tlizr
"""
import matplotlib.pyplot as plt
from cv2 import VideoCapture, destroyAllWindows, imshow, cvtColor, waitKey, \
COLOR_RGB2GRAY, CascadeClassifier, equalizeHist, rectangle, putText, \
TrackerGOTURN_create, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, \
FONT_HERSHEY_COMPLEX,LINE_AA, CAP_DSHOW, resize,INTER_LINEAR, imread, imshow
from tensorflow.keras.models import load_model
from numpy import argmax, reshape

ROOT = 'C:/Users/tlizr/.spyder-py3'                                             # Dirección principal
PATH_VIOLA_JONES = '/Viola-Jones'                                               # Dirección de los clasificadores del algoritmo VIOLA-JONES
VIOLA_JONES = "/haarcascade_frontalface_default.xml" 

print("Iniciando detector")                                                 # Entrada -> Objeto de camara web
face_classifier = CascadeClassifier()                                       # Salida -> Objeto del detector Viola-Jones
face_classifier.load(ROOT + PATH_VIOLA_JONES + VIOLA_JONES)

frame = imread("D:/tlizr/Pictures/Camera Roll/WIN_20210828_18_27_16_Pro.jpg")
gray = cvtColor(frame, COLOR_RGB2GRAY)                                      # Entrada -> objeto del detector Viola-Jones, imagen
gray = equalizeHist(gray)                                                   # Salida -> coordenadas x e y de la esquina superior izquierda, altura, anchura, numero de detecciones
faces = face_classifier.detectMultiScale(gray,
                                             scaleFactor = 2,
                                             minNeighbors = 3)
for roi in faces:
    rectangle(frame,                                                            # Entrada -> Cuadro delimitador, imagen
                  (roi[0], roi[1]),
                  (roi[0] + roi[2], roi[1] + roi[3]),
                  ( 255, 0, 0 ),
                  2,
                  1)
    imshow('Ventana', frame)
waitKey(0)
destroyAllWindows()