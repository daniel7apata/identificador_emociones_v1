import cv2
import os
import threading
from collections import deque
import time

fotogramas_capturados = 'C:/Users/Daniel/Desktop/fotogramas_capturados'  # Donde se guardan las imágenes de emociones
imagePaths = os.listdir(fotogramas_capturados)
emotions = ['Alegria','Desagrado','Enojo','Miedo','Neutral','Sorpresa','Tristeza'] 

class FrameBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=5)  # Cantidad maxima de elementos en cola del Buffer
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


intervalo_fps = 1/30 # Intervalo de FPS

start_time = time.time()

emociones_detectadas = []
registros_tiempo = []

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
        milliseconds = int((elapsed_time % 1) * 1000)

        # Formateo del texto del tiempo transcurrido
        time_text = '{:02d}:{:02d}'.format(minutes, seconds)

        # Guardar imagen en la carpeta correspondiente
        img_name = f"{'({:02d}.{:02d}.{:03d})'.format(minutes, seconds, milliseconds)}_{imagePaths[emocion_detectada[0]]}.jpg"
        cv2.imwrite(os.path.join(fotogramas_capturados, img_name), rostro)

        # Posición para el texto en la esquina inferior derecha
        # Ajusta '20' y '30' para cambiar la distancia desde el borde si es necesario
        text_position = (frame.shape[1] - 100, frame.shape[0] - 20)


        #Agrega la emocion detectada al registro
        if not any(registro['time_text'] == time_text for registro in registros_tiempo):
        # Si no existe, añadir el nuevo registro
            registros_tiempo.append({'time_text': time_text})
            emociones_detectadas.append(f"{time_text}-{imagePaths[emocion_detectada[0]]}")

        # Dibujo del texto en el fotograma
        cv2.putText(frame, time_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, time_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


    cv2.imshow('Identificador de emociones', frame)

    tiempo_procesamiento = time.time() - momento_inicio  # Tiempo desde que se procesa hasta que se muestra el frame
    if tiempo_procesamiento < intervalo_fps:
        time.sleep(intervalo_fps - tiempo_procesamiento)  # Espera para mantener FPS indicados

    k = cv2.waitKey(1)
    if k == 27: 
        break

total_registros = len(emociones_detectadas)
cantidad_alegria = sum("Alegria" in elemento for elemento in emociones_detectadas)
cantidad_desagrado = sum("Desagrado" in elemento for elemento in emociones_detectadas)
cantidad_enojo = sum("Enojo" in elemento for elemento in emociones_detectadas)
cantidad_miedo = sum("Miedo" in elemento for elemento in emociones_detectadas)
cantidad_neutral = sum("Neutral" in elemento for elemento in emociones_detectadas)
cantidad_sorpresa = sum("Sorpresa" in elemento for elemento in emociones_detectadas)
cantidad_tristeza = sum("Tristeza" in elemento for elemento in emociones_detectadas)

with open("resumen_analisis.html", "w") as archivo:

    archivo.write("<!DOCTYPE html>")
    archivo.write("<html lang=\"en\">")
    archivo.write("<head>")
    archivo.write("<meta charset=\"UTF-8\">")
    archivo.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")
    archivo.write("<title>Resultados del analisis</title>")
    archivo.write("<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>")
    archivo.write("</head>")
    archivo.write("<body>")
    archivo.write("<div style=\"display:flex\">")

    #canva
    archivo.write("<div style=\"width: 600px; height: 600px;\"><canvas id=\"myChart\" width=\"100px\" height=\"100px\"></canvas></div>")
    #Script del canva
    archivo.write("<script>new Chart(document.getElementById(\"myChart\").getContext(\"2d\"),{type:\"pie\",data:{labels:[\"Alegria\",\"Desagrado\",\"Enojo\",\"Miedo\",\"Neutral\",\"Sorpresa\",\"Tristeza\"],datasets:[{label:\"Resultados del analisis\",data:[")
    archivo.write(str(cantidad_alegria) + "," + str(cantidad_desagrado) + "," + str(cantidad_enojo) + "," + str(cantidad_miedo) + "," + str(cantidad_neutral) + "," + str(cantidad_sorpresa) + "," + str(cantidad_tristeza))
    archivo.write("],backgroundColor:[\"#2196f3\",\"#4caf50\",\"#f44336\",\"#dc19fa\",\"#9e9e9e\",\"#ed6b00\",\"#3f51b5\"]}]}});</script>")

    #detalle
    archivo.write("<div style=\"display:block; margin-left: 50px\"><h2>Total de registros: " + str(total_registros) + "</h2>")
    archivo.write("<br><h3>Alegria: {:.2f}%</h3>\n".format(cantidad_alegria/total_registros*100))
    archivo.write("<br><h3>Desagrado: {:.2f}%</h3>\n".format(cantidad_desagrado/total_registros*100))
    archivo.write("<br><h3>Enojo: {:.2f}%</h3>\n".format(cantidad_enojo/total_registros*100))
    archivo.write("<br><h3>Miedo: {:.2f}%</h3>\n".format(cantidad_miedo/total_registros*100))
    archivo.write("<br><h3>Neutral: {:.2f}%</h3>\n".format(cantidad_neutral/total_registros*100))
    archivo.write("<br><h3>Sorpresa: {:.2f}%</h3>\n".format(cantidad_sorpresa/total_registros*100))
    archivo.write("<br><h3>Tristeza: {:.2f}%</h3>\n".format(cantidad_tristeza/total_registros*100))
    archivo.write("</div></div>\n\n")

    archivo.write("\n\n<a href=\"registro_emociones.txt\"><h3 style=\"margin-left:20px\">Clic para ver el registro de emociones</h3></a>\n")
    #añade las emociones detectadas al archivo
    

    archivo.write("</body></html>")

with open("registro_emociones.txt", "w") as archivo:
    for emocion in emociones_detectadas:
        archivo.write(emocion + "\n")


os.startfile("resumen_analisis.html")
cap.release()
cv2.destroyAllWindows()