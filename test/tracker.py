     # -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:37:22 2021

@author: tlizr
"""
from cv2 import TrackerGOTURN_create, VideoCapture, imshow, waitKey, destroyAllWindows, cvtColor, equalizeHist, CascadeClassifier, COLOR_RGB2GRAY, getTickCount, getTickFrequency, rectangle

PATH = '/root'
VJCLASSIFIER = '/haarcascade_frontalface_default.xml'   # haarcascade_frontalface_default.xml
                                                        # haarcascade_frontalface_alt.xml
                                                        # haarcascade_frontalface_alt2.xml
                                                        # haarcascade_frontalface_alt_tree.xml

def iniciar_detector():
    face_classifier = CascadeClassifier()
    if not face_classifier.load(PATH + VJCLASSIFIER):
        liberar_recursos(video)
    else:
        return face_classifier

def detectar_cara(face_classifier, image):
    print("Detectando cara...")
    gray = cvtColor(image, COLOR_RGB2GRAY)
    gray = equalizeHist(gray)
    faces, numDetections = face_classifier.detectMultiScale2(gray, scaleFactor = 1.1, minNeighbors = 5)
    if len(numDetections) == 0:
        print("No se ha detectado ninguna cara...")
        return (), ()
    else:
        print("Detectado ", str(len(numDetections)), " caras con dimensiones (x,y,w,h): ", faces)
        for (x,y,w,h) in faces:
            faces = rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            return faces, numDetections

def iniciar_cam():
    print("Iniciando cámara web...")
    video = VideoCapture(0)
    if not video.isOpened:
        liberar_recursos(video)
    else:
        return video
    
def toma_muestra(video):
    print("Tomando muestra...")
    ret, frame = video.read()

    if ret:
        print("Se ha tomado muestra")
        return ret, frame
    else:
        print("No se ha tomado muestra")
        return False, None

def mostrar_video(frame):
    imshow("Cam", frame)

def liberar_recursos(video):
    print("Saliendo...")
    video.release()
    destroyAllWindows()
    exit(0)

def iniciar_seguimiento():
    print("Iniciando tracking...")
    tracker = TrackerGOTURN_create()
    return tracker

def empezar_seguimiento(tracker, frame, face, fok):
    print("Realizando tracking...")
    tracker.init(frame,face)
    timer = getTickCount()
    tok, bbox = tracker.update(frame)
    fps = getTickFrequency() / (getTickCount() - timer);
    if tok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        return rectangle(frame, p1, p2, (255,0,0), 2, 1), True
    else:
        tok = False
        return None, False

print("Iniciando cámara web...")
video = VideoCapture(0)
if not video.isOpened:
    liberar_recursos(video)
detector_facial = iniciar_detector()
tracker = iniciar_seguimiento()

fok = False

print("loop")
while(True):
    print("Tomando muestra...")
    ret, frame = video.read()
    if not ret:
        liberar_recursos(video)
    else:
        face, numDetections = detectar_cara(detector_facial, frame)
        if len(numDetections) == 0:
            fok = False
        else:
            fok = True
            frame, fok = empezar_seguimiento(tracker, frame, face, fok)
            if fok:
                mostrar_video(frame)
    if waitKey(50) & 0xFF == ord('q'):
        break
