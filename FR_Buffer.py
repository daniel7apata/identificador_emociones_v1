import cv2
import os
import threading
from collections import deque
import time


class FrameBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=3)  # Grado de buffer
        self.lock = threading.Lock()

    def add(self, frame):
        with self.lock:
            self.buffer.appendleft(frame)

    def get(self):
        with self.lock:
            if self.buffer:
                return self.buffer.pop()
            return None

def capture_frames(cap, buffer):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.add(frame)

dataPath = "C:/Users/Daniel/Desktop/datos_entrenamiento"
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model.xml")

#cap = cv2.VideoCapture('dina_recopilacion_1.mp4')

#0 representa la cámara web integrada y 1 la camara virtual de OBS
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

frameBuffer = FrameBuffer()

# Empezar a usar hilo adicional para capturar frames
threading.Thread(target=capture_frames, args=(cap, frameBuffer), daemon=True).start()


intervalo_fps = 1 /24 # Intervalo de FPS

start_time = time.time()

emociones_detectadas = []

while True:
    momento_inicio = time.time()  # Tiempo de inicio para procesar y mostrar el frame
    
    frame = frameBuffer.get()
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        emocion_detectada = face_recognizer.predict(rostro)

        

        #cv2.putText(frame,'{}'.format(emocion_detectada),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
          
        # Sombra de texto blanco
        cv2.putText(frame, '{}'.format(imagePaths[emocion_detectada[0]]), (x, y-25), 1, 1.3, (0, 0, 0), 4, cv2.LINE_AA)
        # Texto blanco
        cv2.putText(frame, '{}'.format(imagePaths[emocion_detectada[0]]), (x, y-25), 1, 1.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


        # Cálculo del tiempo transcurrido
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        # Formateo del texto del tiempo transcurrido
        time_text = '{:02d}:{:02d}'.format(minutes, seconds)

        # Posición para el texto en la esquina inferior derecha
        # Ajusta '20' y '30' para cambiar la distancia desde el borde si es necesario
        text_position = (frame.shape[1] - 100, frame.shape[0] - 20)


        #Agrega la emocion detectada al registro
        emociones_detectadas.append(f"{time_text}-{imagePaths[emocion_detectada[0]]}")

        # Dibujo del texto en el fotograma
        cv2.putText(frame, time_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, time_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)




    cv2.imshow('frame', frame)

    tiempo_procesamiento = time.time() - momento_inicio  # Tiempo desde que se procesa hasta que se muestra el frame
    if tiempo_procesamiento < intervalo_fps:
        time.sleep(intervalo_fps - tiempo_procesamiento)  # Espera para mantener FPS indicados

    k = cv2.waitKey(1)
    if k == 27: 
        break


with open("registro_emociones.txt", "w") as archivo:
    for emocion in emociones_detectadas:
        archivo.write(emocion + "\n")

cap.release()
cv2.destroyAllWindows()