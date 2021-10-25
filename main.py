 # -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:37:22 2021

@author: Izri Toufali Lapaz
"""

# --------- MÓDULOS -----------------------------------------------------------
from cv2 import VideoCapture, destroyAllWindows, imshow, cvtColor, waitKey, \
COLOR_RGB2GRAY, CascadeClassifier, equalizeHist, rectangle, putText, \
TrackerGOTURN_create, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, \
FONT_HERSHEY_COMPLEX,LINE_AA, CAP_DSHOW, resize,INTER_LINEAR
from tensorflow.keras.models import load_model
from numpy import argmax, reshape
from sys import exit
# --------- VARIABLES GLOBALES ------------------------------------------------
ROOT = 'C:/Users/tlizr/.spyder-py3'                                             # Dirección principal
PATH_CNN = "/CNN"                                                               # Dirección de los clasificadores convolucionales
PATH_VIOLA_JONES = '/Viola-Jones'                                               # Dirección de los clasificadores del algoritmo VIOLA-JONES
VIOLA_JONES = "/haarcascade_frontalface_default.xml"                            # haarcascade_frontalface_default.xml
                                                                                # haarcascade_frontalface_alt.xml
                                                                                # haarcascade_frontalface_alt2.xml
                                                                                # haarcascade_frontalface_alt_tree.xml
CNN = "/efficientnet_learningrate=0.001_3classes_epoch=20_time12.2_filt_balan.h5"
LOFFSET = 50                                                                    # 80 Offset para el lado del cuadrado
YCOFFSET = 17                                                                   # 27 Negativo hacia arriba
XCOFFSET = 0                                                                    # -13 Negativo hacia izquierda
RESOLUTION = (-1, -1)                                                           # Resolucion de la camara web
FPS = 20                                                                        # Imagenes por segundo de la camara web

# --------- FUNCIONES ---------------------------------------------------------
def liberar_recursos(cap):                                                      # Libera recursos de camara y cerramos la ventana de la camara web
    print("Saliendo")                                                           # Entrada -> objeto de camara web de OpenCV
    cap.release()
    destroyAllWindows()
    exit(0)

def mostrar(frame, prediction, roi):                                            # Muestra la imagen, la clasificación si la hubiera
    putText(frame,                                                              # y caracteristicas de la camara web
            'Resolucion (x(px), y(px)): ' + str(RESOLUTION),                    # Entrada -> Imagen, clasificacion del CNN, cuadro delimitador de la cara
            (0,15),
            FONT_HERSHEY_COMPLEX,
            0.5,
            (0,0,255),
            1,
            LINE_AA,
            False)
    putText(frame,
            'Tasa de imagenes: ' + str(FPS) + " fps",
            (0,30),
            FONT_HERSHEY_COMPLEX,
            0.5,
            (0,0,255),
            1,
            LINE_AA,
            False)
    if prediction != -1:
        if prediction == 0:
            text = "Positivo"
        if prediction == 1:
            text = "Negativo"
        if prediction == 2:
            text = "Neutro"
        putText(frame,
                str(text),
                (roi[0],roi[1] + roi[3]+30),
                FONT_HERSHEY_COMPLEX,
                0.5,
                (0,255,0),
                1,
                LINE_AA,
                False)
    imshow('frame', frame)

def ajustar_region(roi):                                                        # Centrado y limitación del cuadro delimitador
    c = (roi[0] + roi[2]//2 + XCOFFSET, roi[1] + roi[3]//2 + YCOFFSET)          # dentro de la imagen
    l = max(roi[2], roi[3]) + LOFFSET                                           # Entrada -> Cuadro delimitador de la cara
    if c[0] - l//2 < 0:                                                         # Salida -> Coordenadas x e y de la esquina superior izquierda del cuadro delimitador, altura, anchura 
        x0 = 0
    else:
        x0 = c[0] - l//2
    if c[1] - l//2 < 0:
        y0 = 0
    else:
        y0 = c[1] - l//2
    if x0 + l > RESOLUTION[0]:
        x1 = RESOLUTION[0]
        x0 = x1 - l
    else:
        x1 = x0 + l
    if y0 + l > RESOLUTION[1]:
        y1 = RESOLUTION[1]
        y0 = y1 - l
    else:
        y1 = y0 + l
    return (x0, y0, l, l)

def detectar_cara(face_classifier, frame):                                      # Aplicacion del algoritmo Viola-Jones
    gray = cvtColor(frame, COLOR_RGB2GRAY)                                      # Entrada -> objeto del detector Viola-Jones, imagen
    gray = equalizeHist(gray)                                                   # Salida -> coordenadas x e y de la esquina superior izquierda, altura, anchura, numero de detecciones
    faces = face_classifier.detectMultiScale(gray,
                                             scaleFactor = 1.1,
                                             minNeighbors = 5)
    if len(faces) != 0:
        print("Cara detectada!")
    
    return faces

def crear_seguimiento():                                                        # Inicializacion del tracker
    print("Iniciando tracking...")                                              # Salida -> Objeto tracker
    return TrackerGOTURN_create()

def iniciar_seguimiento(tracker, frame, face):                                  # Primera instancia del tracker
    tracker.init(frame,face)                                                    # Entrada -> objeto tracker, imagen, cuadro delimitador de la cara

def iniciar_cam():                                                              # Inicio de la camara web
    print("Iniciando cámara web")                                               # Salida -> objeto de camara web
    cap = VideoCapture(0, CAP_DSHOW)
    if not cap.isOpened():
        print("Error al abrir la cámara, saliendo")
        liberar_recursos(cap)
    else:
        global RESOLUTION
        RESOLUTION = propiedades_video(cap)
#        print("Resolución: " + str(RESOLUTION))
#        print("Tasa de imagenes: " + str(FPS))
#        print("Tasa de bits: " + str(BITRATE))
        return cap

def propiedades_video(cap):                                                     # Extractor de propiedades de la camara web
                                                                                # Entrada -> objeto de camara web
                                                                                # Salida -> Resolucion y FPS de la camara web
    return (int(cap.get(CAP_PROP_FRAME_WIDTH)),\
            int(cap.get(CAP_PROP_FRAME_HEIGHT)))
#            int(cap.get(CAP_PROP_BITRATE))
    
def tomar_muestras(cap):                                                        # Toma de muestra a traves de la camara web
    ret, frame = cap.read()                                                     # Entrada -> objeto de camara web
    if not ret:                                                                 # Salida -> imagen
        print("Error tomando muestras, saliendo")
        liberar_recursos(cap)
    else:
        return frame

def iniciar_detector(cap):                                                      # Inicializacion del detector Viola-Jones
    print("Iniciando detector")                                                 # Entrada -> Objeto de camara web
    face_classifier = CascadeClassifier()                                       # Salida -> Objeto del detector Viola-Jones
    if not face_classifier.load(ROOT + PATH_VIOLA_JONES + VIOLA_JONES):
        liberar_recursos(cap)
    else:
        return face_classifier

def continuar_seguimiento(tracker, frame):                                      # Seguimiento tras la primera instancia del tracker
    tok, bbox = tracker.update(frame)                                           # Entrada -> Objeto tracker, imagen
    if tok:                                                                     # Salida -> Si hay seguimiento, cuadro delimitador
        fixed_bbox = ajustar_region(bbox)
        dibujar_rectangulo(fixed_bbox, frame)
        return True, fixed_bbox
    else:
        return False, ()

def dibujar_rectangulo(roi, frame):                                             # Creacion del cuadro delimitador
    rectangle(frame,                                                            # Entrada -> Cuadro delimitador, imagen
              (roi[0], roi[1]),
              (roi[0] + roi[2], roi[1] + roi[3]),
              ( 255, 0, 0 ),
              2,
              1)

def cargar_modelo():                                                            # Carga del modelo convolucional
    print("Cargando modelo...")                                                 # Salida -> modelo convolucional
    model = load_model(ROOT + PATH_CNN + CNN)
    config = model.get_config()
    global INPUT_SHAPE 
    INPUT_SHAPE = config["layers"][0]["config"]["batch_input_shape"][1]
    return model

def clasificar_accion_facial(model, roi):                                       # Clasificacion de expresion facial
    prediction = model.predict(                                                 # Entrada -> modelo convolucional, cuadro delimitador
                                procesar_imagen(roi),                           # Salida -> clasificacion expresion facial
                                batch_size=1,
                                verbose=0,
                                steps=None,
                                callbacks=None,
                                workers=1,
                                use_multiprocessing=False)
    return argmax(prediction)

def procesar_imagen(roi):                                                       # Reescalado y normalizacion de la region a clasificar
    roi = resize(roi, (INPUT_SHAPE,INPUT_SHAPE), interpolation= INTER_LINEAR)   # Entrada -> cuadro delimitador
    roi = roi / 255                                                             # Salida -> Cuadro delimitador normalizado para clasificar la expresion facial
    return reshape(roi, (1,INPUT_SHAPE,INPUT_SHAPE,3))

# --------- FUNCIÓN PRINCIPAL -------------------------------------------------
# --------- INICIALIZACION ----------------------------------------------------
modelo = cargar_modelo()                                                        # Carga el modelo ya entrenado
video = iniciar_cam()                                                           # Constructor del dispositivo de video
detector_facial = iniciar_detector(video)                                       # Constructor del algoritmo Viola-Jones
seguimiento = crear_seguimiento()                                               # Constructor del tracker

sok = False                                                                     # Variable que indica si se ha realizado el seguimiento
pred = -1                                                                       # Variable para guardar la ultima clasificacion
count = 0

while True:
    
# --------- INICIO ------------------------------------------------------------
    imagen = tomar_muestras(video)                                              # Toma una muestra de la camara web
# --------- PROCESADO ---------------------------------------------------------
    if sok:                                                                     # Si hay seguimiento
        sok, rdi = continuar_seguimiento(seguimiento, imagen)                   # Continua con el tracking
        pred = clasificar_accion_facial(modelo,                                 # Clasifica la expresion facial
                                        imagen[rdi[1]:(rdi[1]+rdi[3]),
                                               rdi[0]:(rdi[0]+rdi[2])])
    if not sok:                                                                 # Si no hay seguimiento
        rdi = detectar_cara(detector_facial, imagen)                            # Detecta una cara en la imagen actual
        if len(rdi) == 0:                                                       # Si no hay deteccion
            sok = False                                                         # No hay seguimiento
        else:                                                                   # Si hay deteccion
            iniciar_seguimiento(seguimiento, imagen, rdi[0])                       # Iniciamos el seguimiento
            sok = True                                                          # Hay seguimiento
            print("Clasificando")
    mostrar(imagen, pred, rdi)                                                  # Muestra la imagen, el cuadro delimitador y la clasificacion

# --------- FIN ---------------------------------------------------------------
    tecla = waitKey(int(1000/FPS))
    if tecla == ord('q'):                                                       # Esperamos 50ms (20fps) o pulsamos "q" para salir
        tecla = None
        break
    elif tecla == ord('u'):
        tecla = None
        print("Actualizando")
        sok = False
        pred = -1

liberar_recursos(video)                                                         # Liberamos recursos
